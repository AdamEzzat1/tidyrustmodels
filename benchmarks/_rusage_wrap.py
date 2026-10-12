"""POSIX helper for run_benchmarks.py: run a command, then report the exact peak RSS of
its largest descendant (`ru_maxrss` of RUSAGE_CHILDREN) on stderr as
`BENCH_RUSAGE_MAXRSS_KB <n>`. Not used on Windows. Untested until the harness runs on
Linux/macOS (macOS reports bytes, Linux kilobytes; normalised here)."""
from __future__ import annotations

import resource
import subprocess
import sys

rc = subprocess.call(sys.argv[1:])
maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if sys.platform == "darwin":
    maxrss //= 1024
print(f"BENCH_RUSAGE_MAXRSS_KB {maxrss}", file=sys.stderr)
sys.exit(rc)
