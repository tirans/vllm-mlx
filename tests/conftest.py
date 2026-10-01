# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default=None,
        help="URL of the vllm-mlx server for integration tests",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model loading",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires model loading)"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running server)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    server_url = config.getoption("--server-url")
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            # Supplying an integration server is an explicit opt-in to the
            # integration suite, including tests that are also marked slow.
            if item.get_closest_marker("slow") is not None and not (
                server_url and item.get_closest_marker("integration") is not None
            ):
                item.add_marker(skip_slow)

    # Skip integration tests unless server URL is explicitly provided
    if not server_url:
        skip_integration = pytest.mark.skip(
            reason="Integration tests require --server-url"
        )
        for item in items:
            if item.get_closest_marker("integration") is not None:
                item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def server_url(request):
    """Get server URL from command line."""
    return request.config.getoption("--server-url")


@pytest.fixture(scope="session")
def anyio_backend():
    """Run anyio-marked tests on asyncio only."""
    return "asyncio"
