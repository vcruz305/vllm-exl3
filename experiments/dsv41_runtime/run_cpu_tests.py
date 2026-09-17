"""Run synthetic transport and export checks without importing the bootstrap."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "runtime"))
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
