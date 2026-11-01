"""Small byte-identity helpers shared by fitting and locked evaluation."""
import hashlib
from pathlib import Path


def file_hash(path):
    """Hash file contents, not paths or modification dates."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def software_hashes():
    """Record experiment code, excluding generated results and test fixtures."""
    root=Path(__file__).resolve().parent
    return {str(path.relative_to(root)):file_hash(path) for path in root.rglob('*.py')
            if not any(part in path.parts for part in ['results','tests','__pycache__'])}
