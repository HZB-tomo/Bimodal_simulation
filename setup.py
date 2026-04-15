from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name            = "neutron_xray_sim",
    version         = "1.1.0",
    author          = "neutron_xray_sim contributors",
    description     = "Dual-modality neutron/X-ray CT simulation and bimodal histogram analysis",
    long_description= long_description,
    long_description_content_type = "text/markdown",
    url             = "https://github.com/neutron-xray-sim/neutron_xray_sim",
    packages        = find_packages(),
    python_requires = ">=3.9",
    install_requires = [
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-image>=0.20",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
    ],
    extras_require = {
        "gpu":   ["astra-toolbox>=1.9"],
        "gridrec": ["tomopy>=1.14"],
        "docs":  ["sphinx>=7", "sphinx-rtd-theme", "myst-parser"],
        "dev":   ["pytest", "jupyter"],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
