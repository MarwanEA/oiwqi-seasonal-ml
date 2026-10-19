"""Run steps 01 to 08 in order with the same options; stops at the first step that fails."""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = sorted(p for p in HERE.glob("0[1-8]_*.py"))

if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):
        subprocess.run([sys.executable, str(STEPS[0]), "--help"])
        sys.exit(0)
    t0 = time.perf_counter()
    for step in STEPS:
        print(f"== {step.name}", flush=True)
        t = time.perf_counter()
        if subprocess.run([sys.executable, str(step), *sys.argv[1:]]).returncode:
            sys.exit(f"{step.name} failed")
        print(f"   {time.perf_counter() - t:.0f} s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s")
