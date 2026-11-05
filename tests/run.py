"""Run unit tests without downloading or loading model weights."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "clickhouse")]
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
