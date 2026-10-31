"""Make `prh` and the converter tools importable no matter where pytest is started."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for sub in ('src', 'tools'):
    path = str(root / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
