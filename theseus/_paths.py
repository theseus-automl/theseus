import os
from pathlib import Path

CACHE_DIR = os.getenv(
    'THESEUS_CACHE_DIR',
    Path.home() / '.theseus' / 'cache',
)
CACHE_DIR = Path(CACHE_DIR)
CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
