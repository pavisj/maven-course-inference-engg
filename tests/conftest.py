import sys
from pathlib import Path

# Make app/ and control/ importable when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
