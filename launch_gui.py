#!/usr/bin/env python3
"""
launch_gui.py
=============
Cross-platform launcher for neutron_xray_sim GUI.

Run:
    python launch_gui.py

On Windows you can also double-click this file if Python is in PATH.
"""
import sys
import os
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# ── Make sure the package is findable ────────────────────────────────────────
pkg_dir = HERE / "neutron_xray_sim"
if pkg_dir.is_dir() and str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ── Dependency check ─────────────────────────────────────────────────────────
missing = []
for dep in ["PyQt5", "matplotlib", "numpy"]:
    try:
        __import__(dep)
    except ImportError:
        missing.append(dep)

if missing:
    print("Missing required packages:")
    for m in missing:
        print(f"  pip install {m}")
    print("\nInstall them and re-run.")
    sys.exit(1)

# ── Launch ────────────────────────────────────────────────────────────────────
gui_path = HERE / "neutron_xray_sim_gui.py"
if not gui_path.exists():
    print(f"GUI script not found: {gui_path}")
    sys.exit(1)

# Run as a module to keep imports clean
import runpy
runpy.run_path(str(gui_path), run_name="__main__")
