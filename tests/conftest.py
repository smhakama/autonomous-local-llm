"""Ensure ``router/`` is importable when pytest runs from anywhere.

``pyproject.toml`` does not declare this project as an installable
package (setup is script-based via ``setup_ai_env.sh``), so we add the
repo root to ``sys.path`` here rather than rely on a ``src/`` layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
