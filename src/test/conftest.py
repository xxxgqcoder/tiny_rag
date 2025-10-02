import os

import pytest

from common.config import Config, TinyRAGConfig


@pytest.fixture(scope="session", autouse=True)
def setup_config():
    """Load configuration from config.yaml before running tests."""
    # The TinyRAGConfig is initialized on import, but we can re-initialize it
    # for the test session to ensure the correct config is loaded.
    # This is a bit of a workaround for the current config loading strategy.
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")

    # Re-initialize the config object for the test session
    Config(_settings_config=dict(yaml_file=config_path))  # type: ignore

    assert TinyRAGConfig is not None, "Configuration should be loaded"
