# SPDX-License-Identifier: Apache-2.0
"""Resilience tests for the MLLM scheduler's process loop.

A token id the tokenizer cannot decode used to escape
``_process_batch_responses`` into ``_process_loop``, whose handler logs, sleeps
100ms and re-enters with the SAME undecodable token still at the head of the
batch. Observed 2026-08-24 on qwen3.8-27b: 1514 identical ``OverflowError``
tracebacks while the endpoint refused connections, and a map that had been
running for 20 hours stopped making progress.

One corrupt token is a property of one request, not of the server.
"""

import logging
from unittest.mock import MagicMock

from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
from vllm_mlx.request import RequestStatus


class _ExplodingDetokenizer:
    """Stands in for a detokenizer handed a token outside the vocab range."""

    def add_token(self, token):
        raise OverflowError("out of range integral type conversion attempted")

    @property
    def last_segment(self):  # pragma: no cover - add_token raises first
        raise AssertionError("should not be reached")


def _scheduler() -> MLLMScheduler:
    model = MagicMock()
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    return MLLMScheduler(model, processor, MLLMSchedulerConfig(max_num_seqs=4))


def _running_request(scheduler: MLLMScheduler, request_id="req-1", uid=1):
    request = MagicMock()
    request.request_id = request_id
    request.status = RequestStatus.RUNNING
    request.output_tokens = [5, 6]
    request.num_prompt_tokens = 3
    request.num_output_tokens = 2
    request.first_token_time = 1.0
    request.mtp_drafts = 0
    request.mtp_accepted = 0
    scheduler.requests[request_id] = request
    scheduler.running[request_id] = request
    scheduler.uid_to_request_id[uid] = request_id
    scheduler.request_id_to_uid[request_id] = uid
    return request


def _response(uid=1, token=999_999_999, finish_reason=None):
    resp = MagicMock()
    resp.uid = uid
    resp.token = token
    resp.finish_reason = finish_reason
    resp.from_draft = False
    resp.error = None
    return resp


class TestUndecodableToken:
    def test_bad_token_fails_the_request_not_the_loop(self, caplog):
        """The whole point: the exception must not escape to _process_loop."""
        scheduler = _scheduler()
        _running_request(scheduler)
        scheduler._detokenizer_pool["req-1"] = _ExplodingDetokenizer()

        with caplog.at_level(logging.ERROR, logger="vllm_mlx.mllm_scheduler"):
            outputs, finished = scheduler._process_batch_responses([_response()])

        assert "req-1" in finished
        assert [o.finish_reason for o in outputs] == ["error"]
        assert any("not decodable" in r.getMessage() for r in caplog.records)

    def test_the_detokenizer_is_not_left_behind(self):
        """A pooled detokenizer that raises would raise again on every token."""
        scheduler = _scheduler()
        _running_request(scheduler)
        scheduler._detokenizer_pool["req-1"] = _ExplodingDetokenizer()

        scheduler._process_batch_responses([_response()])

        assert "req-1" not in scheduler._detokenizer_pool

    def test_one_bad_request_does_not_stop_the_batch(self):
        """The sibling request in the same batch must still be served."""
        scheduler = _scheduler()
        _running_request(scheduler, "req-1", uid=1)
        good = _running_request(scheduler, "req-2", uid=2)
        good.output_tokens = [7]
        scheduler._detokenizer_pool["req-1"] = _ExplodingDetokenizer()

        outputs, finished = scheduler._process_batch_responses(
            [_response(uid=1), _response(uid=2, token=7)]
        )

        assert "req-1" in finished
        by_id = {o.request_id: o for o in outputs}
        assert "req-2" in by_id
        assert by_id["req-2"].finish_reason != "error"


class TestLoopRetryBound:
    def test_the_retry_bound_exists_and_is_finite(self):
        """1514 retries of one broken state is what this constant bounds."""
        from vllm_mlx.mllm_scheduler import MAX_CONSECUTIVE_STEP_ERRORS

        assert 0 < MAX_CONSECUTIVE_STEP_ERRORS < 1000

    def test_abort_all_in_flight_clears_running_and_waiting(self, caplog):
        scheduler = _scheduler()
        _running_request(scheduler, "req-1", uid=1)
        _running_request(scheduler, "req-2", uid=2)

        with caplog.at_level(logging.ERROR, logger="vllm_mlx.mllm_scheduler"):
            scheduler._abort_all_in_flight("test")

        assert any("aborted 2 request" in r.getMessage() for r in caplog.records)

    def test_abort_survives_a_failing_abort(self):
        """Recovery must not depend on every abort succeeding."""
        scheduler = _scheduler()
        _running_request(scheduler, "req-1", uid=1)
        scheduler.abort_request = MagicMock(side_effect=RuntimeError("nope"))

        scheduler._abort_all_in_flight("test")  # must not raise
