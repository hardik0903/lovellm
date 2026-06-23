"""
Path setup for the backend package.

Import this module BEFORE any other local imports to ensure all
subdirectory modules are discoverable via bare imports.

Usage (in api.py or any entry point):
    import _path_setup  # noqa: F401  (must be first local import)
"""

import os as _os
import sys as _sys

_BACKEND_ROOT = _os.path.dirname(_os.path.abspath(__file__))

_SUBPACKAGE_DIRS = [
    "core",
    "agents",
    "agents/math",
    "agents/code",
    "agents/data",
    "agents/document",
    "agents/knowledge",
    "agents/writing",
    "agents/research",
    "retrieval",
    "answer",
    "web",
    "telemetry",
    "evaluation",
    "tests",
]

for _subdir in _SUBPACKAGE_DIRS:
    _path = _os.path.join(_BACKEND_ROOT, _subdir.replace("/", _os.sep))
    if _path not in _sys.path:
        _sys.path.insert(0, _path)

if _BACKEND_ROOT not in _sys.path:
    _sys.path.insert(0, _BACKEND_ROOT)
