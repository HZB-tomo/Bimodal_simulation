#!/usr/bin/env python3
"""
launch_gui.py
=============
Cross-platform launcher for the DIANA GUI.

Run from the repository root:
    python launch_gui.py

Equivalent to ``diana-gui`` or ``python -m neutron_xray_sim.gui`` once the package
is installed (``pip install -e ".[gui]"``).
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# Allow running from a plain checkout without installing the package.
if (HERE / "neutron_xray_sim").is_dir() and str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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
    print('\nOr install everything with:  pip install -e ".[gui]"')
    sys.exit(1)

from neutron_xray_sim.gui.app import main  # noqa: E402

main()
