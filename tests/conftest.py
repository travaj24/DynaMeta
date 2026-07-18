"""Make `import dynameta` work when running pytest from anywhere, and select a WORKING numba
threading layer before any parallel kernel launches (dynameta.numba_env: prefer TBB, probe it
in a sacrificial subprocess, fall back to workqueue when the TBB runtime is broken -- its
failure mode is a livelock/hard crash, not an exception)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.numba_env import ensure_working_threading_layer   # noqa: E402

ensure_working_threading_layer()
