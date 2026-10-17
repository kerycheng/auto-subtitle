import pytest


@pytest.fixture(autouse=True)
def isolated_user_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOSUB_SETTINGS_PATH", str(tmp_path / "preferences.json"))
    # Preserve internal repair regressions; public-mode tests explicitly clear it.
    monkeypatch.setenv("AUTOSUB_ENABLE_REPAIR_UI", "1")
