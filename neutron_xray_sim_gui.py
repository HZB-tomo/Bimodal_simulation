"""
neutron_xray_sim_gui.py
========================
Desktop GUI for the neutron_xray_sim package.

Launch:
    python neutron_xray_sim_gui.py

Requirements (beyond neutron_xray_sim):
    pip install PyQt5 matplotlib numpy

Two top-level modes:
  Simulate  — configure phantom / X-ray / neutron / artifacts / reconstruction,
              run the pipeline, save every stage as .npy via SimCache.
  Analysis  — load any number of saved (vol_xray, vol_neutron) pairs, compute
              bimodal histograms, run cluster quality metrics, compare algorithms.
"""

from __future__ import annotations

import sys
import os
import json
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── PyQt5 ────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QLineEdit, QFileDialog, QProgressBar,
    QScrollArea, QFrame, QSizePolicy, QListWidget, QListWidgetItem,
    QTextEdit, QDialog, QDialogButtonBox, QMessageBox, QToolButton,
    QAbstractItemView, QStatusBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

# ── Matplotlib embedded in Qt ─────────────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens — dark scientific aesthetic
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg0":        "#0e1117",   # deepest background
    "bg1":        "#161b27",   # panel background
    "bg2":        "#1e2535",   # card / group background
    "bg3":        "#252d42",   # input / hover background
    "border":     "#2e3a55",   # subtle borders
    "border_hi":  "#3d4f72",   # highlighted border
    "accent":     "#4f8ef7",   # primary blue accent
    "accent2":    "#7c5cfc",   # secondary violet accent
    "accent3":    "#00d4aa",   # success / neutron green-teal
    "warn":       "#f59e0b",   # warning amber
    "danger":     "#ef4444",   # error red
    "text0":      "#f0f4ff",   # primary text
    "text1":      "#8fa3cc",   # secondary text
    "text2":      "#556080",   # muted text
    "xray_col":   "#60a5fa",   # X-ray channel colour
    "neut_col":   "#34d399",   # Neutron channel colour
}

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0e1117;
    color: #f0f4ff;
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #1e2535;
    border: 1px solid #2e3a55;
    border-radius: 8px;
    margin-top: 20px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    font-size: 12px;
    color: #8fa3cc;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #8fa3cc;
    background-color: #1e2535;
}
QLabel {
    color: #8fa3cc;
    font-size: 12px;
}
QLabel#heading {
    color: #f0f4ff;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
QLabel#subheading {
    color: #8fa3cc;
    font-size: 13px;
}
QLabel#value_label {
    color: #f0f4ff;
    font-size: 12px;
    font-weight: 600;
}
QLabel#channel_x {
    color: #60a5fa;
    font-weight: 700;
    font-size: 12px;
}
QLabel#channel_n {
    color: #34d399;
    font-weight: 700;
    font-size: 12px;
}
QPushButton {
    background-color: #252d42;
    color: #f0f4ff;
    border: 1px solid #2e3a55;
    border-radius: 6px;
    padding: 7px 18px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2e3a55;
    border-color: #4f8ef7;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #1e2535;
}
QPushButton#primary {
    background-color: #4f8ef7;
    color: #ffffff;
    border: none;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #6ba3f9;
}
QPushButton#primary:pressed {
    background-color: #3a7be5;
}
QPushButton#danger {
    background-color: #7f1d1d;
    color: #fecaca;
    border: 1px solid #ef4444;
}
QPushButton#danger:hover {
    background-color: #991b1b;
}
QPushButton#success {
    background-color: #064e3b;
    color: #6ee7b7;
    border: 1px solid #34d399;
}
QPushButton#success:hover {
    background-color: #065f46;
}
QComboBox {
    background-color: #252d42;
    color: #f0f4ff;
    border: 1px solid #2e3a55;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 13px;
    min-height: 28px;
}
QComboBox:hover { border-color: #4f8ef7; }
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #8fa3cc;
    width: 0; height: 0;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1e2535;
    color: #f0f4ff;
    border: 1px solid #2e3a55;
    selection-background-color: #4f8ef7;
    outline: none;
}
QSpinBox, QDoubleSpinBox {
    background-color: #252d42;
    color: #f0f4ff;
    border: 1px solid #2e3a55;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 13px;
    min-height: 28px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #4f8ef7; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #2e3a55;
    border: none;
    width: 18px;
    border-radius: 3px;
}
QLineEdit {
    background-color: #252d42;
    color: #f0f4ff;
    border: 1px solid #2e3a55;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 13px;
    min-height: 28px;
}
QLineEdit:hover, QLineEdit:focus { border-color: #4f8ef7; }
QCheckBox {
    color: #8fa3cc;
    spacing: 8px;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #2e3a55;
    border-radius: 4px;
    background-color: #252d42;
}
QCheckBox::indicator:checked {
    background-color: #4f8ef7;
    border-color: #4f8ef7;
    image: none;
}
QCheckBox::indicator:hover { border-color: #4f8ef7; }
QProgressBar {
    background-color: #1e2535;
    border: 1px solid #2e3a55;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4f8ef7, stop:1 #7c5cfc);
    border-radius: 4px;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: #1e2535;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2e3a55;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #4f8ef7; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QListWidget {
    background-color: #1e2535;
    border: 1px solid #2e3a55;
    border-radius: 8px;
    padding: 4px;
    font-size: 12px;
}
QListWidget::item {
    color: #8fa3cc;
    padding: 6px 10px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #252d42;
    color: #f0f4ff;
    border-left: 2px solid #4f8ef7;
}
QListWidget::item:hover { background-color: #252d42; color: #f0f4ff; }
QTextEdit {
    background-color: #1e2535;
    color: #8fa3cc;
    border: 1px solid #2e3a55;
    border-radius: 8px;
    padding: 8px;
    font-family: "Fira Code", "Consolas", "Courier New", monospace;
    font-size: 12px;
}
QTabWidget::pane {
    border: none;
    background-color: transparent;
}
QTabBar::tab {
    background-color: transparent;
    color: #556080;
    padding: 8px 20px;
    border-bottom: 2px solid transparent;
    font-size: 13px;
    font-weight: 500;
}
QTabBar::tab:selected {
    color: #f0f4ff;
    border-bottom: 2px solid #4f8ef7;
}
QTabBar::tab:hover { color: #8fa3cc; }
QSplitter::handle {
    background-color: #2e3a55;
    width: 2px;
}
QFrame#sidebar {
    background-color: #161b27;
    border-right: 1px solid #2e3a55;
}
QFrame#card {
    background-color: #1e2535;
    border: 1px solid #2e3a55;
    border-radius: 10px;
}
QFrame#topbar {
    background-color: #161b27;
    border-bottom: 1px solid #2e3a55;
}
QStatusBar {
    background-color: #161b27;
    color: #556080;
    border-top: 1px solid #2e3a55;
    font-size: 12px;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread — runs simulation / analysis without blocking the UI
# ─────────────────────────────────────────────────────────────────────────────

class Worker(QThread):
    progress  = pyqtSignal(int, str)   # (percent, message)
    finished  = pyqtSignal(object)     # result object
    error     = pyqtSignal(str)        # traceback string

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn   = fn
        self._args = args
        self._kw   = kwargs

    def run(self):
        try:
            result = self._fn(self.progress.emit, *self._args, **self._kw)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Reusable widgets
# ─────────────────────────────────────────────────────────────────────────────

class SectionHeader(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("heading")

class SubHeader(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("subheading")

class ChannelLabel(QLabel):
    def __init__(self, text: str, channel: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName(f"channel_{channel}")

class Separator(QFrame):
    def __init__(self, orientation=QFrame.HLine, parent=None):
        super().__init__(parent)
        self.setFrameShape(orientation)
        self.setStyleSheet("color: #2e3a55; background-color: #2e3a55;")
        self.setFixedHeight(1) if orientation == QFrame.HLine else None

class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")

class LogPanel(QTextEdit):
    """Scrolling log output panel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(180)
        self.setPlaceholderText("Output will appear here…")

    def append_line(self, msg: str, colour: str = "#8fa3cc"):
        self.append(f'<span style="color:{colour}">{msg}</span>')
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def ok(self,   msg): self.append_line(msg, "#34d399")
    def warn(self, msg): self.append_line(msg, "#f59e0b")
    def err(self,  msg): self.append_line(msg, "#ef4444")
    def info(self, msg): self.append_line(msg, "#8fa3cc")

class MplCanvas(QWidget):
    """Matplotlib figure wrapped in a Qt widget with nav toolbar."""
    def __init__(self, ncols=1, nrows=1, parent=None):
        super().__init__(parent)
        self.fig = Figure(facecolor="#161b27", tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background: #161b27;")
        toolbar = NavToolbar(self.canvas, self)
        toolbar.setStyleSheet(
            "QToolBar { background:#161b27; border:none; }"
            "QToolButton { background:#1e2535; border:1px solid #2e3a55;"
            "  border-radius:4px; color:#8fa3cc; padding:3px; }"
            "QToolButton:hover { background:#252d42; color:#f0f4ff; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        self.fig.clear()
        self.canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Simulate tab
# ─────────────────────────────────────────────────────────────────────────────

class SimulatePanel(QWidget):
    """Full simulation configuration and run panel."""

    simulation_done = pyqtSignal(object)  # emits SimulationResult

    def __init__(self, log: LogPanel, parent=None):
        super().__init__(parent)
        self.log    = log
        self.worker = None
        self._result = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = QHBoxLayout()
        title = SectionHeader("Simulation Setup")
        topbar.addWidget(title)
        topbar.addStretch()
        self.run_btn  = QPushButton("  Run Simulation")
        self.run_btn.setObjectName("primary")
        self.run_btn.setFixedHeight(38)
        self.run_btn.clicked.connect(self._run)
        topbar.addWidget(self.run_btn)
        root.addLayout(topbar)

        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        root.addWidget(self.progress_bar)
        root.addWidget(self.progress_label)

        # ── Scrollable config area ─────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        col = QVBoxLayout(container)
        col.setSpacing(12)
        col.setContentsMargins(0, 0, 8, 0)
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        # ── Phantom config ────────────────────────────────────────────────────
        g_ph = QGroupBox("Phantom")
        g_ph_l = QFormLayout(g_ph)
        self.phantom_combo = QComboBox()
        self.phantom_combo.addItems(["composite", "battery", "bone_implant", "industrial"])
        self.N_spin = QSpinBox()
        self.N_spin.setRange(32, 1024)
        self.N_spin.setValue(64)
        self.N_spin.setSingleStep(32)
        self.N_spin.setSuffix(" voxels")
        g_ph_l.addRow("Preset:", self.phantom_combo)
        g_ph_l.addRow("Grid size N³:", self.N_spin)
        col.addWidget(g_ph)

        # ── Projection config ─────────────────────────────────────────────────
        g_proj = QGroupBox("Forward Projection")
        g_proj_l = QGridLayout(g_proj)
        g_proj_l.setColumnStretch(1, 1)
        g_proj_l.setColumnStretch(3, 1)

        g_proj_l.addWidget(ChannelLabel("X-ray", "x"), 0, 0, 1, 2)
        g_proj_l.addWidget(ChannelLabel("Neutron", "n"), 0, 2, 1, 2)

        g_proj_l.addWidget(QLabel("Angles:"), 1, 0)
        self.n_angles_spin = QSpinBox()
        self.n_angles_spin.setRange(30, 3600)
        self.n_angles_spin.setValue(120)
        g_proj_l.addWidget(self.n_angles_spin, 1, 1)

        g_proj_l.addWidget(QLabel("Tube voltage (kVp):"), 2, 0)
        self.kvp_spin = QDoubleSpinBox()
        self.kvp_spin.setRange(40, 300)
        self.kvp_spin.setValue(120)
        self.kvp_spin.setSuffix(" kV")
        g_proj_l.addWidget(self.kvp_spin, 2, 1)

        g_proj_l.addWidget(QLabel("Al pre-filter:"), 3, 0)
        self.al_spin = QDoubleSpinBox()
        self.al_spin.setRange(0, 20)
        self.al_spin.setValue(2.0)
        self.al_spin.setSuffix(" mm")
        g_proj_l.addWidget(self.al_spin, 3, 1)

        g_proj_l.addWidget(QLabel("I₀ X-ray:"), 2, 2)
        self.i0x_spin = QDoubleSpinBox()
        self.i0x_spin.setRange(100, 1e8)
        self.i0x_spin.setValue(1e5)
        self.i0x_spin.setDecimals(0)
        self.i0x_spin.setSingleStep(10000)
        g_proj_l.addWidget(self.i0x_spin, 2, 3)

        g_proj_l.addWidget(QLabel("I₀ Neutron:"), 3, 2)
        self.i0n_spin = QDoubleSpinBox()
        self.i0n_spin.setRange(100, 1e8)
        self.i0n_spin.setValue(1e5)
        self.i0n_spin.setDecimals(0)
        self.i0n_spin.setSingleStep(10000)
        g_proj_l.addWidget(self.i0n_spin, 3, 3)

        col.addWidget(g_proj)

        # ── X-ray reconstruction config ───────────────────────────────────────
        g_rx = QGroupBox("X-ray Reconstruction")
        g_rx_l = QFormLayout(g_rx)

        self.alg_x_combo = QComboBox()
        self.alg_x_combo.addItems([
            "FBP", "GRIDREC", "SIRT", "SART", "CGLS",
            "EM", "OSSART", "TV_MIN", "NESTEROV_SIRT"
        ])
        self.alg_x_combo.setCurrentText("SART")

        self.filter_x_combo = QComboBox()
        self.filter_x_combo.addItems(["shepp-logan", "ram-lak", "cosine", "hann", "hamming"])

        self.n_iter_x_spin = QSpinBox()
        self.n_iter_x_spin.setRange(1, 500)
        self.n_iter_x_spin.setValue(30)

        self.lambda_tv_x = QDoubleSpinBox()
        self.lambda_tv_x.setRange(0.001, 1.0)
        self.lambda_tv_x.setValue(0.02)
        self.lambda_tv_x.setDecimals(3)
        self.lambda_tv_x.setSingleStep(0.005)

        g_rx_l.addRow("Algorithm:", self.alg_x_combo)
        g_rx_l.addRow("FBP filter:", self.filter_x_combo)
        g_rx_l.addRow("Iterations:", self.n_iter_x_spin)
        g_rx_l.addRow("TV lambda:", self.lambda_tv_x)
        col.addWidget(g_rx)

        # ── Neutron reconstruction config ─────────────────────────────────────
        g_rn = QGroupBox("Neutron Reconstruction")
        g_rn_l = QFormLayout(g_rn)

        self.alg_n_combo = QComboBox()
        self.alg_n_combo.addItems([
            "FBP", "GRIDREC", "SIRT", "SART", "CGLS",
            "EM", "OSSART", "TV_MIN", "NESTEROV_SIRT"
        ])
        self.alg_n_combo.setCurrentText("SART")

        self.filter_n_combo = QComboBox()
        self.filter_n_combo.addItems(["shepp-logan", "ram-lak", "cosine", "hann", "hamming"])

        self.n_iter_n_spin = QSpinBox()
        self.n_iter_n_spin.setRange(1, 500)
        self.n_iter_n_spin.setValue(30)

        self.lambda_tv_n = QDoubleSpinBox()
        self.lambda_tv_n.setRange(0.001, 1.0)
        self.lambda_tv_n.setValue(0.02)
        self.lambda_tv_n.setDecimals(3)
        self.lambda_tv_n.setSingleStep(0.005)

        g_rn_l.addRow("Algorithm:", self.alg_n_combo)
        g_rn_l.addRow("FBP filter:", self.filter_n_combo)
        g_rn_l.addRow("Iterations:", self.n_iter_n_spin)
        g_rn_l.addRow("TV lambda:", self.lambda_tv_n)
        col.addWidget(g_rn)

        # ── Artifact config ───────────────────────────────────────────────────
        g_art = QGroupBox("Artifacts")
        g_art_l = QVBoxLayout(g_art)

        row1 = QHBoxLayout()
        self.noise_chk  = QCheckBox("Photon noise")
        self.bh_chk     = QCheckBox("Beam hardening (uncorrected)")
        self.bhc_chk    = QCheckBox("BH correction")
        row1.addWidget(self.noise_chk)
        row1.addWidget(self.bh_chk)
        row1.addWidget(self.bhc_chk)
        row1.addStretch()
        g_art_l.addLayout(row1)

        row2 = QHBoxLayout()
        self.nscatter_chk = QCheckBox("Neutron scatter")
        self.xscatter_chk = QCheckBox("X-ray scatter")
        self.psf_chk      = QCheckBox("Detector PSF")
        row2.addWidget(self.nscatter_chk)
        row2.addWidget(self.xscatter_chk)
        row2.addWidget(self.psf_chk)
        row2.addStretch()
        g_art_l.addLayout(row2)

        row3 = QHBoxLayout()
        self.rings_chk    = QCheckBox("Ring artifacts")
        self.misalign_chk = QCheckBox("Misalignment")
        row3.addWidget(self.rings_chk)
        row3.addWidget(self.misalign_chk)
        row3.addStretch()
        g_art_l.addLayout(row3)

        g_art_l.addWidget(Separator())

        # Artifact parameter sub-grid
        art_params = QGridLayout()
        art_params.setColumnStretch(1, 1); art_params.setColumnStretch(3, 1)

        art_params.addWidget(QLabel("I₀ noise X-ray:"), 0, 0)
        self.noise_i0x = QDoubleSpinBox()
        self.noise_i0x.setRange(100, 1e8)
        self.noise_i0x.setValue(5e4)
        self.noise_i0x.setDecimals(0)
        self.noise_i0x.setSingleStep(5000)
        art_params.addWidget(self.noise_i0x, 0, 1)

        art_params.addWidget(QLabel("I₀ noise Neutron:"), 0, 2)
        self.noise_i0n = QDoubleSpinBox()
        self.noise_i0n.setRange(100, 1e8)
        self.noise_i0n.setValue(5e4)
        self.noise_i0n.setDecimals(0)
        self.noise_i0n.setSingleStep(5000)
        art_params.addWidget(self.noise_i0n, 0, 3)

        art_params.addWidget(QLabel("Neutron scatter f:"), 1, 0)
        self.ns_frac = QDoubleSpinBox()
        self.ns_frac.setRange(0, 0.5)
        self.ns_frac.setValue(0.06)
        self.ns_frac.setSingleStep(0.01)
        art_params.addWidget(self.ns_frac, 1, 1)

        art_params.addWidget(QLabel("X-ray scatter f:"), 1, 2)
        self.xs_frac = QDoubleSpinBox()
        self.xs_frac.setRange(0, 0.5)
        self.xs_frac.setValue(0.04)
        self.xs_frac.setSingleStep(0.01)
        art_params.addWidget(self.xs_frac, 1, 3)

        art_params.addWidget(QLabel("PSF sigma X-ray:"), 2, 0)
        self.psf_x_sig = QDoubleSpinBox()
        self.psf_x_sig.setRange(0.1, 10)
        self.psf_x_sig.setValue(0.8)
        self.psf_x_sig.setSingleStep(0.1)
        art_params.addWidget(self.psf_x_sig, 2, 1)

        art_params.addWidget(QLabel("PSF sigma Neutron:"), 2, 2)
        self.psf_n_sig = QDoubleSpinBox()
        self.psf_n_sig.setRange(0.1, 10)
        self.psf_n_sig.setValue(1.5)
        self.psf_n_sig.setSingleStep(0.1)
        art_params.addWidget(self.psf_n_sig, 2, 3)

        art_params.addWidget(QLabel("Ring columns:"), 3, 0)
        self.ring_cols = QSpinBox()
        self.ring_cols.setRange(1, 20)
        self.ring_cols.setValue(3)
        art_params.addWidget(self.ring_cols, 3, 1)

        art_params.addWidget(QLabel("Misalign (vox):"), 3, 2)
        self.misalign_vx = QDoubleSpinBox()
        self.misalign_vx.setRange(0, 20)
        self.misalign_vx.setValue(3.0)
        self.misalign_vx.setSingleStep(0.5)
        art_params.addWidget(self.misalign_vx, 3, 3)

        g_art_l.addLayout(art_params)
        col.addWidget(g_art)

        # ── Save options ──────────────────────────────────────────────────────
        g_save = QGroupBox("Save Options")
        g_save_l = QVBoxLayout(g_save)

        self.save_chk = QCheckBox("Save all pipeline stages to disk")
        self.save_chk.setChecked(True)
        g_save_l.addWidget(self.save_chk)

        save_path_row = QHBoxLayout()
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setPlaceholderText("Output directory…")
        self.save_path_edit.setText(str(Path.home() / "neutron_xray_results"))
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_save)
        save_path_row.addWidget(self.save_path_edit)
        save_path_row.addWidget(browse_btn)
        g_save_l.addLayout(save_path_row)

        save_format_row = QHBoxLayout()
        self.save_phantom_chk  = QCheckBox("Phantom (.npy)")
        self.save_sino_chk     = QCheckBox("Sinograms (.npy)")
        self.save_vol_chk      = QCheckBox("Volumes (.npy)")
        self.save_hist_chk     = QCheckBox("Histogram (.npy)")
        for chk in [self.save_phantom_chk, self.save_sino_chk,
                    self.save_vol_chk, self.save_hist_chk]:
            chk.setChecked(True)
            save_format_row.addWidget(chk)
        save_format_row.addStretch()
        g_save_l.addLayout(save_format_row)
        col.addWidget(g_save)

        col.addStretch()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse_save(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory",
                                             self.save_path_edit.text())
        if d:
            self.save_path_edit.setText(d)

    def _run(self):
        try:
            import neutron_xray_sim as nxs
        except ImportError:
            QMessageBox.critical(self, "Import Error",
                "neutron_xray_sim package not found.\n"
                "Make sure it is in the same folder or on PYTHONPATH.")
            return

        cfg = self._build_config()
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.log.info("Starting simulation…")

        self.worker = Worker(self._sim_fn, cfg)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _build_config(self):
        import neutron_xray_sim as nxs
        return {
            "preset":        self.phantom_combo.currentText(),
            "N":             self.N_spin.value(),
            "n_angles":      self.n_angles_spin.value(),
            "kVp":           self.kvp_spin.value(),
            "filter_mm_Al":  self.al_spin.value(),
            "I0_xray":       self.i0x_spin.value(),
            "I0_neutron":    self.i0n_spin.value(),
            "alg_x":         self.alg_x_combo.currentText(),
            "alg_n":         self.alg_n_combo.currentText(),
            "filter_x":      self.filter_x_combo.currentText(),
            "filter_n":      self.filter_n_combo.currentText(),
            "n_iter_x":      self.n_iter_x_spin.value(),
            "n_iter_n":      self.n_iter_n_spin.value(),
            "lambda_tv_x":   self.lambda_tv_x.value(),
            "lambda_tv_n":   self.lambda_tv_n.value(),
            "artifact": nxs.ArtifactConfig(
                photon_noise             = self.noise_chk.isChecked(),
                I0_xray                  = self.noise_i0x.value(),
                I0_neutron               = self.noise_i0n.value(),
                apply_bh_correction      = self.bhc_chk.isChecked(),
                neutron_scatter          = self.nscatter_chk.isChecked(),
                scatter_fraction         = self.ns_frac.value(),
                xray_scatter             = self.xscatter_chk.isChecked(),
                xray_scatter_fraction    = self.xs_frac.value(),
                detector_psf             = self.psf_chk.isChecked(),
                psf_sigma_xray_pixels    = self.psf_x_sig.value(),
                psf_sigma_neutron_pixels = self.psf_n_sig.value(),
                ring_artifacts           = self.rings_chk.isChecked(),
                n_bad_columns            = self.ring_cols.value(),
                misalignment             = self.misalign_chk.isChecked(),
                translation_voxels       = (self.misalign_vx.value(), 0.0, 0.0),
            ),
            "save":       self.save_chk.isChecked(),
            "save_path":  self.save_path_edit.text(),
            "save_phantom": self.save_phantom_chk.isChecked(),
            "save_sino":    self.save_sino_chk.isChecked(),
            "save_vol":     self.save_vol_chk.isChecked(),
            "save_hist":    self.save_hist_chk.isChecked(),
        }

    @staticmethod
    def _sim_fn(report, cfg):
        """Run inside the Worker thread."""
        import neutron_xray_sim as nxs
        from neutron_xray_sim.artifacts import (
            inject_sinogram_artifacts, inject_volume_artifacts
        )
        from neutron_xray_sim.histogram import compute_bimodal_histogram
        from neutron_xray_sim.io import SimCache

        report(5, "Building phantom…")
        phantom = nxs.make_phantom(cfg["preset"], cfg["N"])

        report(15, "Computing sinograms…")
        xray_sino, neut_sino = nxs.make_sinogram_pair(
            phantom,
            n_angles       = cfg["n_angles"],
            kVp            = cfg["kVp"],
            filter_mm_Al   = cfg["filter_mm_Al"],
            I0_xray        = cfg["I0_xray"],
            I0_neutron     = cfg["I0_neutron"],
            use_astra      = True,
        )

        report(35, "Injecting artifacts…")
        x_art = {k: v.copy() if isinstance(v, np.ndarray) else v
                 for k, v in xray_sino.items()}
        n_art = {k: v.copy() if isinstance(v, np.ndarray) else v
                 for k, v in neut_sino.items()}
        x_art, n_art = inject_sinogram_artifacts(x_art, n_art, cfg["artifact"])

        report(50, f"Reconstructing X-ray ({cfg['alg_x']})…")
        vol_x = nxs.reconstruct(
            x_art, algorithm=cfg["alg_x"],
            filter_name=cfg["filter_x"],
            n_iter=cfg["n_iter_x"],
            lambda_tv=cfg["lambda_tv_x"],
            use_astra=True,
        )

        report(70, f"Reconstructing Neutron ({cfg['alg_n']})…")
        vol_n = nxs.reconstruct(
            n_art, algorithm=cfg["alg_n"],
            filter_name=cfg["filter_n"],
            n_iter=cfg["n_iter_n"],
            lambda_tv=cfg["lambda_tv_n"],
            use_astra=True,
        )

        report(80, "Applying volume artifacts…")
        vol_x, vol_n = inject_volume_artifacts(vol_x, vol_n, cfg["artifact"])

        report(88, "Computing bimodal histogram…")
        hist = compute_bimodal_histogram(vol_x, vol_n, bins=128)

        result = {
            "phantom":   phantom,
            "xray_sino": xray_sino,
            "neut_sino": neut_sino,
            "vol_x":     vol_x,
            "vol_n":     vol_n,
            "hist":      hist,
            "cfg":       cfg,
        }

        if cfg.get("save"):
            report(92, "Saving to disk…")
            tag  = f"{cfg['preset']}_N{cfg['N']}"
            root = Path(cfg["save_path"]) / tag
            cache = SimCache(str(root), overwrite=True)
            if cfg["save_phantom"]:
                try: cache.save_phantom(phantom)
                except Exception: pass
            if cfg["save_sino"]:
                try: cache.save_raw_sinograms(xray_sino, neut_sino)
                except Exception: pass

            import dataclasses
            from neutron_xray_sim.io import tag_to_slug

            class _FakeResult:
                pass
            r = _FakeResult()
            r.tag          = tag
            r.cfg          = cfg["artifact"]
            r.phantom      = phantom
            r.vol_xray     = vol_x
            r.vol_neutron  = vol_n
            r.xray_sino    = x_art
            r.neutron_sino = n_art
            r.histogram    = hist
            r.elapsed_s    = 0.0

            if cfg["save_vol"] or cfg["save_hist"]:
                try: cache.save_run(r)
                except Exception: pass

            result["save_path"] = str(root)

        report(100, "Done.")
        return result

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)
        self.log.info(f"[{pct:3d}%] {msg}")

    def _on_done(self, result):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._result = result
        self.log.ok("Simulation complete.")
        if "save_path" in result:
            self.log.ok(f"Saved to: {result['save_path']}")
        self.simulation_done.emit(result)

    def _on_error(self, tb: str):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.log.err("Simulation failed:")
        for line in tb.strip().splitlines():
            self.log.err(f"  {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis tab
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisPanel(QWidget):
    """Load volumes, compute histograms, run metrics, compare pairs."""

    def __init__(self, log: LogPanel, parent=None):
        super().__init__(parent)
        self.log    = log
        self.worker = None
        self._volumes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # label -> (vol_x, vol_n)
        self._histograms: Dict[str, object] = {}
        self._phantom = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left sidebar: volume management ──────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(300)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(14, 14, 14, 14)
        sb_layout.setSpacing(10)

        sb_layout.addWidget(SectionHeader("Volumes"))
        sb_layout.addWidget(SubHeader("Load (vol_xray, vol_neutron) pairs"))

        # Volume list
        self.vol_list = QListWidget()
        self.vol_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection)
        self.vol_list.setMinimumHeight(180)
        sb_layout.addWidget(self.vol_list)

        # Load buttons
        btn_row1 = QHBoxLayout()
        load_pair_btn = QPushButton("Load .npy pair…")
        load_pair_btn.clicked.connect(self._load_npy_pair)
        load_cache_btn = QPushButton("Load from cache…")
        load_cache_btn.clicked.connect(self._load_from_cache)
        btn_row1.addWidget(load_pair_btn)
        btn_row1.addWidget(load_cache_btn)
        sb_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn = QPushButton("Clear all")
        clear_btn.setObjectName("danger")
        clear_btn.clicked.connect(self._clear_all)
        btn_row2.addWidget(remove_btn)
        btn_row2.addWidget(clear_btn)
        sb_layout.addLayout(btn_row2)

        sb_layout.addWidget(Separator())

        # Optional phantom for GT overlay
        sb_layout.addWidget(QLabel("Ground-truth phantom (optional):"))
        ph_row = QHBoxLayout()
        self.phantom_path_edit = QLineEdit()
        self.phantom_path_edit.setPlaceholderText("Cache dir or skip…")
        load_ph_btn = QPushButton("…")
        load_ph_btn.setFixedWidth(32)
        load_ph_btn.clicked.connect(self._load_phantom)
        ph_row.addWidget(self.phantom_path_edit)
        ph_row.addWidget(load_ph_btn)
        sb_layout.addLayout(ph_row)

        sb_layout.addWidget(Separator())

        # Analysis options
        sb_layout.addWidget(QLabel("Histogram bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(32, 512)
        self.bins_spin.setValue(128)
        self.bins_spin.setSingleStep(32)
        sb_layout.addWidget(self.bins_spin)

        self.log_scale_chk = QCheckBox("Log colour scale")
        self.log_scale_chk.setChecked(True)
        sb_layout.addWidget(self.log_scale_chk)

        self.show_gt_chk = QCheckBox("Show GT markers")
        self.show_gt_chk.setChecked(True)
        sb_layout.addWidget(self.show_gt_chk)

        self.metrics_chk = QCheckBox("Compute quality metrics")
        self.metrics_chk.setChecked(True)
        sb_layout.addWidget(self.metrics_chk)

        sb_layout.addStretch()

        analyse_btn = QPushButton("  Analyse Selected")
        analyse_btn.setObjectName("primary")
        analyse_btn.setFixedHeight(38)
        analyse_btn.clicked.connect(self._analyse)
        sb_layout.addWidget(analyse_btn)

        compare_btn = QPushButton("  Compare All Pairs")
        compare_btn.setObjectName("success")
        compare_btn.setFixedHeight(38)
        compare_btn.clicked.connect(self._compare_all)
        sb_layout.addWidget(compare_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        sb_layout.addWidget(self.progress_bar)

        root.addWidget(sidebar)

        # ── Right: result tabs ────────────────────────────────────────────────
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(12, 12, 12, 12)
        right_l.setSpacing(8)

        self.result_tabs = QTabWidget()

        # Histogram viewer tab
        self.hist_canvas = MplCanvas()
        self.result_tabs.addTab(self.hist_canvas, "Histograms")

        # Volumes viewer tab
        self.vol_canvas = MplCanvas()
        self.result_tabs.addTab(self.vol_canvas, "Volume Slices")

        # Metrics table tab
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.result_tabs.addTab(self.metrics_text, "Metrics")

        # Per-material bar chart tab
        self.bar_canvas = MplCanvas()
        self.result_tabs.addTab(self.bar_canvas, "Centroid Errors")

        right_l.addWidget(self.result_tabs)
        root.addWidget(right, stretch=1)

    # ── Volume loading ────────────────────────────────────────────────────────

    def _load_npy_pair(self):
        """User selects a vol_xray.npy; we auto-find vol_neutron.npy alongside."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select X-ray volume (.npy)",
            str(Path.home()), "NumPy (*.npy)")
        if not path:
            return
        p = Path(path)
        # Try to find the neutron file
        n_path = p.parent / p.name.replace("xray", "neutron").replace("_x.", "_n.")
        if not n_path.exists():
            n_path_str, _ = QFileDialog.getOpenFileName(
                self, "Select paired Neutron volume (.npy)",
                str(p.parent), "NumPy (*.npy)")
            if not n_path_str:
                return
            n_path = Path(n_path_str)
        try:
            vol_x = np.load(str(p))
            vol_n = np.load(str(n_path))
            label = p.parent.name + "/" + p.stem.replace("vol_xray_", "")
            self._add_volume(label, vol_x, vol_n)
            self.log.ok(f"Loaded: {label}  shape={vol_x.shape}")
        except Exception as e:
            self.log.err(f"Load failed: {e}")

    def _load_from_cache(self):
        """Load all runs from a SimCache directory."""
        d = QFileDialog.getExistingDirectory(self, "Select SimCache directory",
                                             str(Path.home()))
        if not d:
            return
        try:
            from neutron_xray_sim.io import SimCache
            cache = SimCache(d)
            tags  = cache.list_run_tags()
            if not tags:
                self.log.warn(f"No runs found in {d}")
                return
            for tag in tags:
                try:
                    vol_x = cache.load_run_volume(tag, "xray")
                    vol_n = cache.load_run_volume(tag, "neutron")
                    label = tag.replace("\n", " ")
                    self._add_volume(label, vol_x, vol_n)
                    self.log.ok(f"Loaded '{label}' from cache")
                except Exception as e:
                    self.log.warn(f"Skipping '{tag}': {e}")
        except Exception as e:
            self.log.err(f"Cache load failed: {e}")

    def _load_phantom(self):
        d = QFileDialog.getExistingDirectory(self, "Select SimCache directory",
                                             str(Path.home()))
        if not d:
            return
        self.phantom_path_edit.setText(d)
        try:
            from neutron_xray_sim.io import SimCache
            cache = SimCache(d)
            meta  = cache.load_phantom_meta()
            from neutron_xray_sim import make_phantom
            self._phantom = make_phantom(meta["name"], meta["N"])
            self.log.ok(f"Phantom loaded: {meta['name']}  N={meta['N']}")
        except Exception as e:
            self.log.warn(f"Could not load phantom: {e}")

    def _add_volume(self, label: str, vol_x: np.ndarray, vol_n: np.ndarray):
        # Avoid duplicates
        if label in self._volumes:
            label = label + "_2"
        self._volumes[label] = (vol_x, vol_n)
        item = QListWidgetItem(label)
        item.setCheckState(Qt.Checked)
        self.vol_list.addItem(item)

    def _remove_selected(self):
        for item in self.vol_list.selectedItems():
            label = item.text()
            self._volumes.pop(label, None)
            self._histograms.pop(label, None)
            self.vol_list.takeItem(self.vol_list.row(item))

    def _clear_all(self):
        self._volumes.clear()
        self._histograms.clear()
        self.vol_list.clear()
        self.hist_canvas.clear()
        self.vol_canvas.clear()
        self.metrics_text.clear()
        self.bar_canvas.clear()
        self.log.info("Cleared all volumes.")

    # ── Accept result from simulation tab ─────────────────────────────────────
    def receive_simulation(self, result: dict):
        """Called when the Simulate tab completes a run."""
        label = f"sim_{result['cfg']['preset']}_N{result['cfg']['N']}"
        self._add_volume(label, result["vol_x"], result["vol_n"])
        self._phantom = result.get("phantom")
        self.log.ok(f"Received simulation result as '{label}'")

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _get_checked_labels(self) -> List[str]:
        checked = []
        for i in range(self.vol_list.count()):
            item = self.vol_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def _analyse(self):
        labels = self._get_checked_labels()
        if not labels:
            self.log.warn("No volumes checked. Check at least one in the list.")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.log.info(f"Analysing {len(labels)} volume pair(s)…")

        vols   = {l: self._volumes[l] for l in labels if l in self._volumes}
        opts   = {
            "bins":        self.bins_spin.value(),
            "log_scale":   self.log_scale_chk.isChecked(),
            "show_gt":     self.show_gt_chk.isChecked(),
            "metrics":     self.metrics_chk.isChecked(),
            "phantom":     self._phantom,
        }
        self.worker = Worker(self._analysis_fn, vols, opts)
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    @staticmethod
    def _analysis_fn(report, vols, opts):
        from neutron_xray_sim.histogram import (
            compute_bimodal_histogram, evaluate_histogram_quality
        )
        results = {}
        n = len(vols)
        for i, (label, (vx, vn)) in enumerate(vols.items()):
            report(int(i / n * 80), f"Computing histogram: {label}")
            hist = compute_bimodal_histogram(vx, vn, bins=opts["bins"])
            m    = None
            if opts["metrics"] and opts["phantom"] is not None:
                try:
                    m = evaluate_histogram_quality(hist, opts["phantom"])
                except Exception:
                    pass
            results[label] = {"hist": hist, "metrics": m}
        report(100, "Done.")
        return results

    def _on_analysis_done(self, results: dict):
        self.progress_bar.setVisible(False)
        self._histograms.update({l: v["hist"] for l, v in results.items()})
        self.log.ok(f"Analysed {len(results)} pair(s).")
        self._plot_histograms(results)
        self._plot_volumes(list(results.keys()))
        self._render_metrics(results)
        self._plot_centroid_bars(results)

    def _plot_histograms(self, results: dict):
        self.hist_canvas.fig.clear()
        n    = len(results)
        ncols = min(n, 3)
        nrows = max(1, -(-n // ncols))
        axes  = self.hist_canvas.fig.subplots(nrows, ncols)
        if n == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]

        log_scale = self.log_scale_chk.isChecked()
        phantom   = self._phantom
        show_gt   = self.show_gt_chk.isChecked() and phantom is not None

        for idx, (label, data) in enumerate(results.items()):
            r, c   = divmod(idx, ncols)
            ax     = axes[r][c] if nrows > 1 else axes[0][c]
            hist   = data["hist"]
            H      = hist.H.T
            H_plot = np.log1p(H) if log_scale else H
            H_plot = np.ma.masked_where(H == 0, H_plot)
            ax.imshow(H_plot, origin="lower",
                      extent=[hist.x_edges[0], hist.x_edges[-1],
                               hist.n_edges[0], hist.n_edges[-1]],
                      aspect="auto", cmap="inferno", interpolation="bilinear")
            ax.set_xlabel(r"$\mu_x$ [cm$^{-1}$]", fontsize=8, color="#8fa3cc")
            ax.set_ylabel(r"$\mu_n$ [cm$^{-1}$]", fontsize=8, color="#8fa3cc")
            ax.set_title(label[:40], fontsize=8, color="#f0f4ff", pad=4)
            ax.tick_params(labelsize=7, colors="#8fa3cc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2e3a55")
            ax.set_facecolor("#0e1117")

            if show_gt:
                try:
                    mu_x_gt = [m._mu_x_table[6] for m in phantom.materials]
                    mu_n_gt = [m.mu_n for m in phantom.materials]
                    cols = plt.cm.Set1(np.linspace(0, 0.9, len(mu_x_gt)))
                    for mx, mn, col in zip(mu_x_gt, mu_n_gt, cols):
                        ax.plot(mx, mn, "D", color="white", markersize=4,
                                markeredgecolor=col, markeredgewidth=1.2, zorder=5)
                except Exception:
                    pass

            m = data.get("metrics")
            if m is not None:
                db_s = f"{m.davies_bouldin:.3f}" if m.davies_bouldin == m.davies_bouldin else "n/a"
                ce_s = f"{m.mean_centroid_error:.4f}" if m.mean_centroid_error == m.mean_centroid_error else "n/a"
                ax.set_title(f"{label[:32]}\nDB={db_s}  CE={ce_s}",
                             fontsize=7, color="#f0f4ff", pad=4, linespacing=1.4)

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            ax   = axes[r][c] if nrows > 1 else axes[0][c]
            ax.set_visible(False)

        self.hist_canvas.fig.patch.set_facecolor("#161b27")
        self.hist_canvas.canvas.draw()

    def _plot_volumes(self, labels: List[str]):
        self.vol_canvas.fig.clear()
        vols = [(l, self._volumes[l]) for l in labels if l in self._volumes]
        if not vols:
            return
        n    = len(vols)
        # 2 rows (X-ray top, Neutron bottom), n cols
        axes = self.vol_canvas.fig.subplots(2, n)
        if n == 1:
            axes = [[axes[0]], [axes[1]]]

        for col, (label, (vx, vn)) in enumerate(vols):
            si = vx.shape[0] // 2
            for row, (vol, mod, col_str) in enumerate([
                (vx, "X-ray",   "#60a5fa"),
                (vn, "Neutron", "#34d399"),
            ]):
                ax = axes[row][col]
                vmin, vmax = np.percentile(vol, [1, 99])
                ax.imshow(vol[si], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
                ax.set_title(f"{mod}\n{label[:20]}", fontsize=7,
                             color=col_str, pad=3)
                ax.axis("off")
                ax.set_facecolor("#0e1117")

        self.vol_canvas.fig.patch.set_facecolor("#161b27")
        self.vol_canvas.canvas.draw()

    def _render_metrics(self, results: dict):
        lines = []
        lines.append("<h3 style='color:#f0f4ff'>Cluster Quality Metrics</h3>")
        has_any = False
        for label, data in results.items():
            m = data.get("metrics")
            if m is None:
                continue
            has_any = True
            lines.append(
                f"<h4 style='color:#60a5fa;margin-bottom:2px'>{label}</h4>"
                f"<table style='color:#8fa3cc;font-size:12px;border-collapse:collapse'>"
                f"<tr><td style='padding:2px 12px 2px 0'><b>Davies-Bouldin</b></td>"
                f"<td style='color:#f0f4ff'>{m.davies_bouldin:.4f}</td></tr>"
                f"<tr><td style='padding:2px 12px 2px 0'><b>Mean centroid error</b></td>"
                f"<td style='color:#f0f4ff'>{m.mean_centroid_error:.4f} cm\u207b\u00b9</td></tr>"
                f"<tr><td style='padding:2px 12px 2px 0'><b>Matched materials</b></td>"
                f"<td style='color:#f0f4ff'>{m.n_matched}</td></tr>"
                f"</table>"
                f"<table style='color:#8fa3cc;font-size:11px;border-collapse:collapse;margin-top:4px'>"
                f"<tr><th style='text-align:left;padding:2px 10px 2px 0'>Material</th>"
                f"<th>CE</th><th>sigma_x</th><th>sigma_n</th></tr>"
            )
            for mat, err in sorted(m.centroid_errors.items()):
                sx = m.sigma_x.get(mat, float("nan"))
                sn = m.sigma_n.get(mat, float("nan"))
                lines.append(
                    f"<tr><td style='padding:1px 10px 1px 0;color:#f0f4ff'>{mat}</td>"
                    f"<td style='padding:1px 6px'>{err:.4f}</td>"
                    f"<td style='padding:1px 6px'>{sx:.4f}</td>"
                    f"<td style='padding:1px 6px'>{sn:.4f}</td></tr>"
                )
            lines.append("</table><br>")

        if not has_any:
            lines.append(
                "<p style='color:#556080'>No metrics available.<br>"
                "Load a phantom and enable 'Compute quality metrics'.</p>"
            )
        self.metrics_text.setHtml("".join(lines))

    def _plot_centroid_bars(self, results: dict):
        self.bar_canvas.fig.clear()
        metrics = {l: d["metrics"] for l, d in results.items()
                   if d.get("metrics") is not None}
        if not metrics:
            self.bar_canvas.canvas.draw()
            return

        all_mats = sorted({name for m in metrics.values()
                           for name in m.centroid_errors})
        n_labels = len(metrics)
        n_mats   = len(all_mats)
        if n_mats == 0:
            return

        x    = np.arange(n_mats)
        w    = 0.7 / max(n_labels, 1)
        ax   = self.bar_canvas.fig.add_subplot(111)
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#8fa3cc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3a55")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#2e3a55", linewidth=0.6)

        colours = plt.cm.tab10(np.linspace(0, 1, n_labels))
        for i, (label, m) in enumerate(metrics.items()):
            errs = [m.centroid_errors.get(mat, float("nan")) for mat in all_mats]
            bars = ax.bar(x + i * w - (n_labels - 1) * w / 2,
                          errs, width=w * 0.88,
                          color=colours[i], alpha=0.85,
                          label=label[:20])

        ax.set_xticks(x)
        ax.set_xticklabels(all_mats, rotation=30, ha="right",
                           fontsize=8, color="#8fa3cc")
        ax.set_ylabel("Centroid error [cm\u207b\u00b9]",
                      fontsize=9, color="#8fa3cc")
        ax.set_title("Per-material centroid errors",
                     fontsize=10, color="#f0f4ff", pad=8)
        ax.legend(fontsize=8, facecolor="#1e2535",
                  edgecolor="#2e3a55", labelcolor="#8fa3cc")

        self.bar_canvas.fig.patch.set_facecolor("#161b27")
        self.bar_canvas.canvas.draw()

    def _compare_all(self):
        """Plot all loaded volumes in a cross-comparison grid."""
        all_labels = [self.vol_list.item(i).text()
                      for i in range(self.vol_list.count())]
        if len(all_labels) < 2:
            self.log.warn("Need at least 2 volume pairs to compare.")
            return
        checked = set(self._get_checked_labels())
        labels  = [l for l in all_labels if l in checked and l in self._volumes]
        if len(labels) < 2:
            self.log.warn("Check at least 2 volumes for comparison.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        vols = {l: self._volumes[l] for l in labels}
        opts = {"bins": self.bins_spin.value()}
        self.worker = Worker(self._compare_fn, vols, opts)
        self.worker.finished.connect(self._on_compare_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    @staticmethod
    def _compare_fn(report, vols, opts):
        from neutron_xray_sim.histogram import compute_bimodal_histogram
        results = {}
        n = len(vols)
        for i, (label, (vx, vn)) in enumerate(vols.items()):
            report(int(i / n * 100), f"Histogram: {label}")
            results[label] = compute_bimodal_histogram(vx, vn, bins=opts["bins"])
        return results

    def _on_compare_done(self, hists: dict):
        self.progress_bar.setVisible(False)
        n     = len(hists)
        ncols = min(n, 4)
        nrows = -(-n // ncols)

        self.hist_canvas.fig.clear()
        axes = self.hist_canvas.fig.subplots(nrows, ncols)
        if n == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [list(axes)]

        log_scale = self.log_scale_chk.isChecked()

        for idx, (label, hist) in enumerate(hists.items()):
            r, c   = divmod(idx, ncols)
            ax     = axes[r][c] if nrows > 1 else axes[0][c]
            H      = hist.H.T
            H_plot = np.log1p(H) if log_scale else H
            H_plot = np.ma.masked_where(H == 0, H_plot)
            ax.imshow(H_plot, origin="lower",
                      extent=[hist.x_edges[0], hist.x_edges[-1],
                               hist.n_edges[0], hist.n_edges[-1]],
                      aspect="auto", cmap="inferno", interpolation="bilinear")
            ax.set_title(label[:36], fontsize=7, color="#f0f4ff", pad=3)
            ax.set_xlabel(r"$\mu_x$", fontsize=7, color="#8fa3cc")
            ax.set_ylabel(r"$\mu_n$", fontsize=7, color="#8fa3cc")
            ax.tick_params(labelsize=6, colors="#8fa3cc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2e3a55")
            ax.set_facecolor("#0e1117")

        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            ax   = axes[r][c] if nrows > 1 else axes[0][c]
            ax.set_visible(False)

        self.hist_canvas.fig.patch.set_facecolor("#161b27")
        self.hist_canvas.fig.suptitle(
            "Multi-volume histogram comparison",
            color="#f0f4ff", fontsize=11)
        self.hist_canvas.canvas.draw()
        self.result_tabs.setCurrentIndex(0)
        self.log.ok(f"Comparison grid: {n} volumes plotted.")

    def _on_error(self, tb: str):
        self.progress_bar.setVisible(False)
        self.log.err("Analysis failed:")
        for line in tb.strip().splitlines():
            self.log.err(f"  {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("neutron_xray_sim  \u2014  Dual-modality CT")
        self.resize(1320, 860)
        self.setMinimumSize(980, 640)
        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(52)
        tb_l = QHBoxLayout(topbar)
        tb_l.setContentsMargins(20, 0, 20, 0)

        brand = QLabel("\u25c6  neutron_xray_sim")
        brand.setStyleSheet(
            "color: #4f8ef7; font-size: 16px; font-weight: 700; letter-spacing: 0.5px;")
        tb_l.addWidget(brand)

        version = QLabel("v1.1.0")
        version.setStyleSheet("color: #2e3a55; font-size: 12px; margin-left: 8px;")
        tb_l.addWidget(version)
        tb_l.addStretch()

        # Mode switcher
        self.mode_tabs = QTabWidget()
        self.mode_tabs.setDocumentMode(True)
        self.mode_tabs.tabBar().setStyleSheet(
            "QTabBar::tab { padding: 6px 28px; font-size: 14px; font-weight: 600; }"
        )
        tb_l.addWidget(self.mode_tabs)
        layout.addWidget(topbar)

        # ── Log panel at bottom ───────────────────────────────────────────────
        self.log = LogPanel()

        # ── Simulate tab ──────────────────────────────────────────────────────
        self.simulate_panel = SimulatePanel(self.log)
        self.simulate_panel.simulation_done.connect(self._on_simulation_done)
        self.mode_tabs.addTab(self.simulate_panel, "  Simulate  ")

        # ── Analysis tab ──────────────────────────────────────────────────────
        self.analysis_panel = AnalysisPanel(self.log)
        self.mode_tabs.addTab(self.analysis_panel, "  Analysis  ")

        # ── Splitter: content above, log below ────────────────────────────────
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        content = QWidget()
        content_l = QVBoxLayout(content)
        content_l.setContentsMargins(0, 0, 0, 0)
        content_l.addWidget(self.mode_tabs)

        log_card = QWidget()
        log_l = QVBoxLayout(log_card)
        log_l.setContentsMargins(12, 6, 12, 6)
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(QLabel("  Console Output"))
        clr = QPushButton("Clear")
        clr.setFixedWidth(60)
        clr.setFixedHeight(22)
        clr.clicked.connect(self.log.clear)
        log_hdr.addWidget(clr)
        log_hdr.addStretch()
        log_l.addLayout(log_hdr)
        log_l.addWidget(self.log)

        splitter.addWidget(content)
        splitter.addWidget(log_card)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 180])

        layout.addWidget(splitter, stretch=1)

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._check_package()

    def _check_package(self):
        try:
            import neutron_xray_sim as nxs
            self.status_bar.showMessage(
                f"  neutron_xray_sim v{nxs.__version__} loaded  \u2014  "
                f"Algorithms: {', '.join(nxs.AVAILABLE_ALGORITHMS)}")
            self.log.ok(f"Package loaded: neutron_xray_sim v{nxs.__version__}")
        except ImportError:
            self.status_bar.showMessage(
                "  WARNING: neutron_xray_sim not found — "
                "place the package folder next to this script.")
            self.log.err("neutron_xray_sim not found. "
                         "Place the package folder next to neutron_xray_sim_gui.py")

    def _on_simulation_done(self, result: dict):
        """Forward simulation result to the analysis panel."""
        self.analysis_panel.receive_simulation(result)
        self.log.ok("Result forwarded to Analysis tab — switch to Analysis to explore.")

    def _apply_theme(self):
        self.setStyleSheet(STYLESHEET)
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,      QColor(PALETTE["bg0"]))
        pal.setColor(QPalette.ColorRole.WindowText,  QColor(PALETTE["text0"]))
        pal.setColor(QPalette.ColorRole.Base,        QColor(PALETTE["bg1"]))
        pal.setColor(QPalette.ColorRole.Text,        QColor(PALETTE["text0"]))
        pal.setColor(QPalette.ColorRole.Highlight,   QColor(PALETTE["accent"]))
        QApplication.setPalette(pal)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("neutron_xray_sim GUI")
    app.setApplicationVersion("1.1.0")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
