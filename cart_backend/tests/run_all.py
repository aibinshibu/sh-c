"""Run the whole backend test suite:  python tests/run_all.py [-m 'not slow']"""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-v", *sys.argv[1:]], cwd=root))
