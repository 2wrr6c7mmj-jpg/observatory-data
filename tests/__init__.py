"""On GitHub Actions, also put each failing test into an annotation (::error): the job log lives on blob
storage that some readers cannot reach, while annotations are served by the REST API."""
import os
import unittest

if os.environ.get("GITHUB_ACTIONS") == "true":
    def _esc(s: str) -> str:
        return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")

    def _wrap(name):
        orig = getattr(unittest.TextTestResult, name)

        def hook(self, test, err):
            orig(self, test, err)
            try:
                text = self._exc_info_to_string(err, test)[-3500:]
                print(f"::error title={name[3:]} {test.id()}::{_esc(text)}", flush=True)
            except Exception:  # noqa: BLE001  (never break the run)
                pass
        setattr(unittest.TextTestResult, name, hook)

    for _n in ("addFailure", "addError"):
        _wrap(_n)
