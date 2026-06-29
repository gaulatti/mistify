import sys
from pathlib import Path

# Make the project root importable so tests can import from `src`.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
