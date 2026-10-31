"""Tests (test.yml) sets OBS_REQUIRE_SAT_DEPS=1 so that the sat tests, which skip
themselves when numpy/Pillow are missing (collect.yml and proto.yml do not install
them), can never be skipped silently in the workflow that is meant to run them."""
import importlib.util
import os
import unittest


@unittest.skipUnless(os.environ.get("OBS_REQUIRE_SAT_DEPS") == "1", "only enforced in the Tests workflow")
class SatDepsPresentTest(unittest.TestCase):
    def test_numpy_and_pillow_are_installed(self):
        for name in ("numpy", "PIL"):
            self.assertIsNotNone(importlib.util.find_spec(name), f"{name} is required in the Tests workflow")
