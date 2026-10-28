"""Pipeline plumbing: window discovery, phase dispatch, options."""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pnavshop import walker  # noqa: E402


class WindowTests(unittest.TestCase):
    def test_invalid_manual_handle_is_rejected(self) -> None:
        self.assertIsNone(walker.find_window(1))

    def test_never_matches_by_title_only(self) -> None:
        # only PetitPlanet.exe windows are accepted
        for window in walker.list_game_windows():
            self.assertEqual(str(window.get("exe", "")).lower(), "petitplanet.exe")


class PhaseTests(unittest.TestCase):
    def test_options_clamp(self) -> None:
        options = walker.ShopOptions(shots=0, turn=10)
        self.assertEqual(options.shots, 1)
        self.assertEqual(options.turn, 100)

    def test_invalid_phase_falls_back_to_both(self) -> None:
        result = walker.run_phase("nonsense", walker.ShopOptions(log=lambda m: None))
        self.assertFalse(result.get("ok"))

    def test_no_game_reports_missing_window(self) -> None:
        for phase in ("nav", "shop", "both"):
            result = walker.run_phase(phase, walker.ShopOptions(log=lambda m: None))
            self.assertFalse(result.get("ok"))
            self.assertIn("game window", result.get("reason", ""))


if __name__ == "__main__":
    unittest.main()
