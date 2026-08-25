# SPDX-License-Identifier: Apache-2.0
"""Resilience tests for Scheduler failure paths.

Covers four independent failure modes:

1. Responses for a BatchGenerator sequence the scheduler has no record of
   (an orphan) used to be dropped silently while holding a batch slot.
2. Cache-corruption recovery was gated on a substring match and never logged
   the triggering exception; an unmatched signature re-raised into a hang.
3. A request whose sampling params differed from the active batch's silently
   decoded with the batch's params.
4. MTP install had no compatibility guard against the mlx-lm BatchGenerator
   shape and raised out of generator creation.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.request import Request, RequestStatus, SamplingParams
from vllm_mlx.scheduler import (
    KNOWN_CACHE_CORRUPTION_SIGNATURES,
    Scheduler,
    SchedulerConfig,
)


def _make_scheduler(**config_kwargs) -> Scheduler:
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(len(x.split())))
    tokenizer.eos_token_id = 0

    config = SchedulerConfig(
        max_num_seqs=4, enable_prefix_cache=False, **config_kwargs
    )
    return Scheduler(model, tokenizer, config)


def _make_request(request_id="req-1", **sampling_kwargs) -> Request:
    params = SamplingParams(**sampling_kwargs)
    request = Request(
        request_id=request_id,
        prompt="hello world",
        sampling_params=params,
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
    )
    return request


def _response(uid, token=7, finish_reason=None):
    resp = MagicMock()
    resp.uid = uid
    resp.token = token
    resp.finish_reason = finish_reason
    return resp


class TestOrphanResponses:
    """A response whose uid has no request mapping must not be dropped."""

    def test_orphan_is_evicted_and_warned_once(self, caplog):
        scheduler = _make_scheduler()
        scheduler.batch_generator = MagicMock()

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler._process_batch_responses([_response(99)])
            scheduler._process_batch_responses([_response(99)])

        # Slot reclaimed on every occurrence...
        assert [c.args[0] for c in scheduler.batch_generator.remove.call_args_list] == [
            [99],
            [99],
        ]
        assert scheduler.orphan_response_count == 2
        # ...but the warning fires once, not once per decoded token.
        orphan_warnings = [
            r.getMessage() for r in caplog.records if "orphan_response" in r.getMessage()
        ]
        assert len(orphan_warnings) == 1
        assert "uid=99" in orphan_warnings[0]

    def test_orphan_eviction_failure_is_not_fatal(self):
        scheduler = _make_scheduler()
        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.remove.side_effect = RuntimeError("gone")

        outputs, finished = scheduler._process_batch_responses([_response(99)])

        assert outputs == []
        assert finished == set()
        assert scheduler.orphan_response_count == 1

    def test_warned_uids_reset_when_generator_closes(self, caplog):
        """mlx-lm counts uids from 0 per generator, so uid 99 can come back."""
        scheduler = _make_scheduler()
        scheduler.batch_generator = MagicMock()
        scheduler._process_batch_responses([_response(99)])
        assert scheduler._orphan_uids == {99}

        scheduler._close_batch_generator()
        assert scheduler._orphan_uids == set()

        scheduler.batch_generator = MagicMock()
        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler._process_batch_responses([_response(99)])
        assert [r for r in caplog.records if "orphan_response" in r.getMessage()]

    def test_stale_mapping_is_dropped(self):
        """uid maps to a request that is no longer running."""
        scheduler = _make_scheduler()
        scheduler.batch_generator = MagicMock()
        scheduler.uid_to_request_id[42] = "req-1"
        scheduler.request_id_to_uid["req-1"] = 42

        scheduler._process_batch_responses([_response(42)])

        assert 42 not in scheduler.uid_to_request_id
        assert "req-1" not in scheduler.request_id_to_uid
        # Not an orphan: the sequence belongs to a request we know about.
        assert scheduler.orphan_response_count == 0
        scheduler.batch_generator.remove.assert_not_called()


class TestStallWatchdog:
    """A batch holding requests while decoding nothing must say so.

    This is the backstop for the original incident: the server held 4 running
    requests emitting zero tokens for 13.5 hours without a single log line.
    """

    def test_idle_scheduler_is_not_stalled(self):
        """Nothing running means nothing to produce -- not a stall."""
        scheduler = _make_scheduler()
        scheduler._last_output_ts -= 10_000
        assert scheduler.stalled_for_s() == 0.0

    def test_stall_grows_while_running_produces_nothing(self):
        scheduler = _make_scheduler()
        scheduler.running["req-1"] = _make_request()
        scheduler._last_output_ts -= 300

        assert scheduler.stalled_for_s() == pytest.approx(300, abs=5)

    def test_warns_once_past_threshold(self, caplog):
        scheduler = _make_scheduler()
        scheduler._stall_warn_threshold_s = 60.0
        scheduler.running["req-1"] = _make_request()
        scheduler._last_output_ts -= 120

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler._check_stall()
            scheduler._check_stall()

        warnings = [
            r.getMessage() for r in caplog.records if "stall_watchdog" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "running" in warnings[0]

    def test_no_warning_below_threshold(self, caplog):
        scheduler = _make_scheduler()
        scheduler._stall_warn_threshold_s = 600.0
        scheduler.running["req-1"] = _make_request()
        scheduler._last_output_ts -= 30

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler._check_stall()

        assert not [r for r in caplog.records if "stall_watchdog" in r.getMessage()]

    def test_threshold_zero_disables_the_watchdog(self, caplog):
        scheduler = _make_scheduler()
        scheduler._stall_warn_threshold_s = 0.0
        scheduler.running["req-1"] = _make_request()
        scheduler._last_output_ts -= 100_000

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler._check_stall()

        assert not [r for r in caplog.records if "stall_watchdog" in r.getMessage()]

    def test_real_token_clears_the_stall(self, caplog):
        """Progress must reset the timer and retract the warning."""
        scheduler = _make_scheduler()
        scheduler.batch_generator = MagicMock()
        scheduler._stall_warn_threshold_s = 60.0
        request = _make_request()
        scheduler.running["req-1"] = request
        scheduler.uid_to_request_id[42] = "req-1"
        scheduler.request_id_to_uid["req-1"] = 42
        scheduler._last_output_ts -= 120
        scheduler._check_stall()
        assert scheduler._stall_warned is True

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler._process_batch_responses([_response(42)])

        assert scheduler._stall_warned is False
        assert scheduler.stalled_for_s() < 5
        assert [r for r in caplog.records if "RECOVERED" in r.getMessage()]

    def test_stats_expose_wedge_signals(self):
        """Both signals were tracked but invisible to /v1/status."""
        scheduler = _make_scheduler()
        scheduler.batch_generator = MagicMock()
        scheduler._process_batch_responses([_response(99)])

        stats = scheduler.get_stats()

        assert stats["orphan_response_count"] == 1
        assert "stalled_for_s" in stats


class TestStreamThreadRecovery:
    """A stream/thread mismatch must not escape step() as a raise.

    engine_core answers a raise by logging, sleeping 100ms and re-entering on
    the same dead stream. Measured 2026-08-25: a mismatch at a model swap
    produced 1742 Metal command-buffer failures over hours while the endpoint
    kept listening and answered /v1/models with 200.
    """

    def _wired(self, scheduler, exc):
        """Drive step() with a batch generator whose next() raises *exc*."""
        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.next.side_effect = exc
        request = _make_request()
        scheduler.running["req-1"] = request
        scheduler.requests["req-1"] = request
        return scheduler

    def test_stream_thread_error_does_not_escape(self):
        scheduler = _make_scheduler()
        self._wired(scheduler, RuntimeError("There is no Stream(gpu, 8) in current thread."))

        output = scheduler.step()  # used to raise

        assert "req-1" in output.finished_request_ids
        assert [o.finish_reason for o in output.outputs] == ["error"]

    def test_client_gets_a_typed_error_not_a_stalled_stream(self):
        """The in-flight request must be failed, not left hanging."""
        scheduler = _make_scheduler()
        self._wired(scheduler, RuntimeError("no Stream(gpu, 3)"))

        scheduler.step()

        assert scheduler.running == {}

    def test_it_rebinds_streams_before_resetting(self, caplog):
        scheduler = _make_scheduler()
        self._wired(scheduler, RuntimeError("no Stream(gpu, 8) in current thread"))

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler.step()

        assert any("stream_thread" in r.getMessage() for r in caplog.records)

    def test_non_stream_errors_keep_their_existing_recovery(self):
        """The generic OOM/Metal path must be unchanged."""
        scheduler = _make_scheduler()
        self._wired(scheduler, RuntimeError("something else entirely"))

        output = scheduler.step()

        assert "req-1" in output.finished_request_ids


class TestCacheRecoveryClassification:
    def test_bare_cache_substring_is_not_a_signature(self):
        """The bare word 'cache' matched almost every TypeError."""
        assert "cache" not in KNOWN_CACHE_CORRUPTION_SIGNATURES

    def test_unknown_typeerror_recovers_and_logs_the_exception(self, caplog):
        scheduler = _make_scheduler()
        scheduler._schedule_waiting = MagicMock(
            side_effect=TypeError("something entirely unrelated")
        )
        scheduler._recover_from_cache_error = MagicMock()
        scheduler._reschedule_running_requests = MagicMock()

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler.step()

        # Recovery is not gated on the message matching a known signature.
        scheduler._recover_from_cache_error.assert_called_once()
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "signature=unknown" in text
        assert "something entirely unrelated" in text

    def test_known_signature_is_labelled(self, caplog):
        scheduler = _make_scheduler()
        scheduler._schedule_waiting = MagicMock(
            side_effect=TypeError("'NoneType' object is not subscriptable")
        )
        scheduler._recover_from_cache_error = MagicMock()
        scheduler._reschedule_running_requests = MagicMock()

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            scheduler.step()

        assert "signature=known" in "\n".join(r.getMessage() for r in caplog.records)

    def test_persistent_failure_fails_requests_instead_of_raising(self):
        """A raise out of step() is a 10Hz spin in engine_core, not a crash."""
        scheduler = _make_scheduler()
        request = _make_request()
        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request
        scheduler._schedule_waiting = MagicMock(side_effect=TypeError("wedged"))
        scheduler._recover_from_cache_error = MagicMock()
        scheduler._reschedule_running_requests = MagicMock()

        output = scheduler.step()

        assert output.finished_request_ids == {"req-1"}
        assert [o.finish_reason for o in output.outputs] == ["error"]
        # The unschedulable request is drained, not left to reproduce the
        # failure on every subsequent step.
        assert len(scheduler.waiting) == 0
        assert request.status == RequestStatus.FINISHED_ABORTED


class TestPerRequestSampler:
    def test_no_override_when_params_match_the_generator(self):
        """Keeps mlx-lm's vectorized sampling path for the common case."""
        scheduler = _make_scheduler()
        params = SamplingParams(temperature=0.7, top_p=0.9, min_p=0.0)
        scheduler._current_sampler_params = scheduler._sampler_params(params)

        assert scheduler._make_request_sampler(params) is None

    def test_override_when_params_differ(self):
        scheduler = _make_scheduler()
        scheduler._current_sampler_params = (0.7, 0.9, 0.0)

        sampler = scheduler._make_request_sampler(
            SamplingParams(temperature=0.0, top_p=1.0, min_p=0.0)
        )
        assert callable(sampler)

    def test_active_batch_blocks_recreation_but_request_keeps_its_params(self):
        scheduler = _make_scheduler()
        generator = MagicMock()
        generator.insert.return_value = [11]
        scheduler.batch_generator = generator
        scheduler._current_sampler_params = (0.7, 0.9, 0.0)
        # An in-flight request pins the generator's sampler.
        scheduler.running["running-req"] = _make_request("running-req")

        request = _make_request("req-cold", temperature=0.0, top_p=1.0)
        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request

        scheduled = scheduler._schedule_waiting()

        assert [r.request_id for r in scheduled] == ["req-cold"]
        # Generator was NOT recreated out from under the active batch...
        assert scheduler.batch_generator is generator
        # ...and the request still decodes with its own sampler.
        samplers = generator.insert.call_args.kwargs["samplers"]
        assert len(samplers) == 1 and callable(samplers[0])

    def test_matching_request_passes_no_samplers(self):
        scheduler = _make_scheduler()
        generator = MagicMock()
        generator.insert.return_value = [12]
        scheduler.batch_generator = generator
        scheduler._current_sampler_params = (0.7, 0.9, 0.0)
        scheduler.running["running-req"] = _make_request("running-req")

        request = _make_request("req-warm", temperature=0.7, top_p=0.9, min_p=0.0)
        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request

        scheduler._schedule_waiting()

        assert "samplers" not in generator.insert.call_args.kwargs


class _FakeBatchGenerator:
    """Stand-in with a controllable attribute surface."""

    def __init__(self, *args, **kwargs):
        self.sampler = kwargs.get("sampler")
        self.logits_processors = []

    def _next(self):  # present on every mlx-lm version
        return [], []


class TestMtpCompatibilityGuard:
    @pytest.fixture
    def scheduler(self):
        sched = _make_scheduler()
        sched.config.enable_mtp = True
        sched.model.mtp = MagicMock()
        return sched

    def test_missing_internals_disable_mtp_instead_of_raising(
        self, scheduler, caplog
    ):
        install = MagicMock()
        with patch("vllm_mlx.scheduler.BatchGenerator", _FakeBatchGenerator), patch(
            "vllm_mlx.scheduler._install_mtp", install
        ), caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            bg = scheduler._create_batch_generator(SamplingParams())

        assert isinstance(bg, _FakeBatchGenerator)
        install.assert_not_called()
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "[MTP] disabled" in text
        assert "_step" in text and "active_batch" in text

    def test_install_failure_restores_prior_patches(self, scheduler, caplog):
        """A half-applied MTP patch must not survive, nor clobber an earlier one."""
        pre_existing_next = MagicMock(name="chunked_prefill_next")

        class _Compatible(_FakeBatchGenerator):
            active_batch = None

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Stand in for the chunked-prefill patch, which binds _next
                # before MTP install runs.
                self._next = pre_existing_next

            def _step(self):
                return None

        def _boom(bg, **kwargs):
            # Fail partway through, after rebinding some internals.
            bg._inner_next = bg._next
            bg._step = "half-applied"
            raise RuntimeError("mtp internals moved")

        with patch("vllm_mlx.scheduler.BatchGenerator", _Compatible), patch(
            "vllm_mlx.scheduler._install_mtp", _boom
        ), caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            bg = scheduler._create_batch_generator(SamplingParams())

        # The partial MTP rebinding is gone; the class method is back.
        assert "_step" not in bg.__dict__
        assert "_inner_next" not in bg.__dict__
        # ...and the unrelated patch that was already there survived.
        assert bg._next is pre_existing_next
        assert "[MTP] install failed" in "\n".join(
            r.getMessage() for r in caplog.records
        )
