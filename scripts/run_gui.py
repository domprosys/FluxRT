import argparse
import json
import os
import platform
import random
import threading
import time

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QSizePolicy,
)

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle

# ── Spout (Windows-only) ──────────────────────────────────────────────────────
_SPOUT_AVAILABLE = False
if platform.system() == "Windows":
    try:
        import SpoutGL

        _SPOUT_AVAILABLE = True
    except ImportError:
        pass

POLL_MS = 40
MAX_CAM_INDEX = 8
if platform.system() == "Windows":
    CAM_BACKEND = cv2.CAP_DSHOW
    CAM_BACKEND_FALLBACK = cv2.CAP_MSMF
else:
    CAM_BACKEND = cv2.CAP_V4L2
    CAM_BACKEND_FALLBACK = None
DEFAULT_CONFIG = "configs/config_with_reference.json"

# ── colour tokens ────────────────────────────────────────────────────────────
BG = "#1e1e1e"
CTRL_BG = "#252526"
ENTRY_BG = "#3c3c3c"
FG = "#cccccc"
DIM_FG = "#666666"
BTN_BG = "#3a3a3a"
BTN_HOVER = "#4e4e4e"
ERR_FG = "#f44747"
VIDEO_BG = "#111111"
STATUS_BG = "#007acc"
STATUS_FG = "#ffffff"
ACCENT = "#007acc"
BORDER = "#4a4a4a"
SEP = "#2d2d2d"

_sp_lock = threading.Lock()


def log(msg: str) -> None:
    print(f"[FluxRT] {msg}", flush=True)


def enumerate_cameras() -> list[tuple[int, bool]]:
    """Return (index, is_live) for each openable camera.

    is_live is True when the camera delivers a non-black frame. Idle NDI /
    virtual cameras open successfully but only yield pure-black frames
    (brightness ~0), so this lets the GUI prefer a real physical camera.
    """
    found = []
    for i in range(MAX_CAM_INDEX):
        cap = cv2.VideoCapture(i, CAM_BACKEND)
        if not cap.isOpened() and CAM_BACKEND_FALLBACK is not None:
            cap = cv2.VideoCapture(i, CAM_BACKEND_FALLBACK)
        if cap.isOpened():
            # Grab a few frames so auto-exposure can warm up, then judge
            # liveness by peak brightness (idle NDI cams stay pure black).
            brightness = 0.0
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    brightness = max(brightness, float(frame.mean()))
            found.append((i, brightness > 2.0))
            cap.release()
    return found


# ── stylesheet ────────────────────────────────────────────────────────────────
STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Noto Sans", "SF Pro Text", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background-color: {BG};
}}

/* generic widget base */
QWidget {{
    background-color: {BG};
    color: {FG};
}}

/* video area */
QWidget#video_root,
QWidget#video_panel {{
    background-color: {VIDEO_BG};
}}

/* control panel and its row containers */
QWidget#ctrl_area,
QWidget#ctrl_row {{
    background-color: {CTRL_BG};
}}

QLabel {{
    background-color: transparent;
    color: {FG};
}}

QLabel#video_title {{
    background-color: {VIDEO_BG};
    color: {DIM_FG};
    font-size: 11px;
    padding: 4px 0px;
}}

QLabel#video_lbl {{
    background-color: {VIDEO_BG};
    color: #3a3a3a;
}}

QLabel#dim {{
    color: {DIM_FG};
}}

QLabel#err {{
    color: {ERR_FG};
}}

QLineEdit, QTextEdit {{
    background-color: {ENTRY_BG};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 7px;
    selection-background-color: #264f78;
    selection-color: {FG};
}}

QLineEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT};
}}

QPushButton {{
    background-color: {BTN_BG};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 14px;
    min-height: 26px;
    min-width: 60px;
}}

QPushButton:hover {{
    background-color: {BTN_HOVER};
    border-color: #666666;
}}

QPushButton:pressed {{
    background-color: #5a5a5a;
}}

QPushButton:disabled {{
    background-color: #262626;
    color: #555555;
    border-color: #333333;
}}

QPushButton#accent {{
    background-color: {ACCENT};
    color: {STATUS_FG};
    border: none;
    font-weight: 600;
}}

QPushButton#accent:hover {{
    background-color: #1a8cd8;
}}

QPushButton#accent:pressed {{
    background-color: #005fa3;
}}

QPushButton#accent:disabled {{
    background-color: #1a3d55;
    color: #6a9ab5;
    border: none;
}}

QComboBox {{
    background-color: {ENTRY_BG};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 7px;
    min-height: 26px;
}}

QComboBox:focus {{
    border-color: {ACCENT};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border-left: 1px solid {BORDER};
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}}

QComboBox::down-arrow {{
    border-left:  4px solid transparent;
    border-right: 4px solid transparent;
    border-top:   5px solid {DIM_FG};
    width: 0;
    height: 0;
}}

QComboBox QAbstractItemView {{
    background-color: {ENTRY_BG};
    color: {FG};
    border: 1px solid {BORDER};
    selection-background-color: #264f78;
    selection-color: {FG};
    outline: none;
    padding: 2px;
}}

QStatusBar {{
    background-color: {STATUS_BG};
    color: {STATUS_FG};
    font-size: 12px;
    padding: 0 8px;
}}

QStatusBar QLabel {{
    color: {STATUS_FG};
    background-color: transparent;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #555555;
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
}}
QScrollBar::handle:horizontal {{
    background: #555555;
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}
"""


# ── cross-thread signals ───────────────────────────────────────────────────────
class _Signals(QObject):
    launch_capture = Signal(
        object, int, str
    )  # (cv2.VideoCapture | None, cam_idx, spout_name)
    sp_error = Signal(str)
    camera_error = Signal()
    vcam_error = Signal(str)


# ── main window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, config_path: str, use_int8: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("FluxRT")
        self.resize(1200, 780)

        self.config_path = config_path
        self._use_int8 = use_int8

        self._sp = None
        self._input_tensor = None
        self._output_tensor = None
        self._resolution: dict | None = None
        self._out_resolution: dict | None = None
        self._use_ref_image = False
        self._lip_transfer_in_config = False
        self._lip_active = False
        self._sp_loading = False

        # Prompt-cycle mode: prompts pre-encoded at startup, switched on a timer.
        self._prompt_cycle: list = []
        self._cycle_interval_ms = 6000
        self._cycle_index = 0
        self._cycle_timer: QTimer | None = None
        self._cycle_paused = False
        self._cfg_w = 576
        self._cfg_h = 320

        self._latest_input: np.ndarray | None = None
        self._latest_output: np.ndarray | None = None
        self._latest_output_bgr: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # Temporal input smoothing (EMA): blended = alpha*new + (1-alpha)*history.
        # alpha=1.0 disables it; lower = dreamier ghost-trails on motion.
        self._smooth_alpha = 1.0
        self._smoothed_frame: np.ndarray | None = None

        self._capture_thread: threading.Thread | None = None
        self._capture_stop = threading.Event()
        self._running = False

        self._vcam_thread: threading.Thread | None = None
        self._vcam_stop = threading.Event()
        self._vcam_cam = None
        self._show_vcam = True  # overridden by config "show_virtual_cam"
        self._show_advanced = False  # overridden by config "show_advanced_controls"

        self._spout_sender = None
        self._spout_sender_thread: threading.Thread | None = None
        self._spout_sender_stop = threading.Event()

        self._ref_full_path: str | None = None

        self._sig = _Signals()
        self._build_ui()

        # Fullscreen toggle: F11 or F to toggle, Esc to exit.
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence("F"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, activated=self._exit_fullscreen)

        # Connect cross-thread signals after UI exists
        self._sig.launch_capture.connect(self._on_launch_capture)
        self._sig.sp_error.connect(self._on_sp_error)
        self._sig.camera_error.connect(self._on_camera_error)
        self._sig.vcam_error.connect(self._on_vcam_error)

        self._load_config_meta(config_path)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_frames)
        self._poll_timer.start(POLL_MS)

        QTimer.singleShot(0, self._begin_start)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── video panels ──────────────────────────────────────────────────────
        video_root = QWidget()
        video_root.setObjectName("video_root")
        video_layout = QHBoxLayout(video_root)
        video_layout.setContentsMargins(6, 6, 6, 6)
        video_layout.setSpacing(6)

        self._input_panel, self._input_lbl = self._make_video_panel(
            video_layout, "Input"
        )
        self._output_panel, self._output_lbl = self._make_video_panel(
            video_layout, "Output"
        )

        # ── control panel ─────────────────────────────────────────────────────
        ctrl_area = QWidget()
        self._ctrl_area = ctrl_area
        ctrl_area.setObjectName("ctrl_area")
        ctrl_layout = QGridLayout(ctrl_area)
        ctrl_layout.setContentsMargins(14, 10, 14, 12)
        ctrl_layout.setHorizontalSpacing(8)
        ctrl_layout.setVerticalSpacing(6)
        ctrl_layout.setColumnStretch(1, 1)

        row = 0

        # Camera row
        ctrl_layout.addWidget(
            QLabel("Camera:"),
            row,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        cam_row = self._ctrl_row()
        cam_row_l = cam_row.layout()
        self._cam_combo = QComboBox()
        self._cam_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        cam_row_l.addWidget(self._cam_combo)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_cameras)
        cam_row_l.addWidget(refresh_btn)
        self._cam_err_lbl = QLabel()
        self._cam_err_lbl.setObjectName("err")
        cam_row_l.addWidget(self._cam_err_lbl)
        cam_row_l.addStretch()
        ctrl_layout.addWidget(cam_row, row, 1, 1, 2)
        row += 1

        # Spout input row (Windows / SpoutGL only)
        self._spout_lbl = QLabel("Spout Input:")
        ctrl_layout.addWidget(
            self._spout_lbl,
            row,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        spout_row = self._ctrl_row()
        spout_row_l = spout_row.layout()
        self._spout_input_edit = QLineEdit()
        self._spout_input_edit.setPlaceholderText(
            "Sender name (leave blank to use camera)"
        )
        self._spout_input_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        spout_row_l.addWidget(self._spout_input_edit)
        spout_row_l.addStretch()
        ctrl_layout.addWidget(spout_row, row, 1, 1, 2)
        # Spout input form removed from the UI (webcam-only setup). The field
        # stays present but empty, so capture always uses the camera.
        self._spout_lbl.setVisible(False)
        spout_row.setVisible(False)
        row += 1

        # Prompt row
        ctrl_layout.addWidget(
            QLabel("Prompt:"),
            row,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setFixedHeight(72)
        self._prompt_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._prompt_edit.textChanged.connect(self._on_prompt_changed)
        ctrl_layout.addWidget(self._prompt_edit, row, 1, 1, 2)
        row += 1

        # Reference image row (conditionally visible)
        self._ref_widget = self._ctrl_row()
        ref_l = self._ref_widget.layout()
        ref_l.addWidget(QLabel("Reference image:"))
        self._ref_path_lbl = QLabel("(none)")
        self._ref_path_lbl.setObjectName("dim")
        self._ref_path_lbl.setMinimumWidth(260)
        ref_l.addWidget(self._ref_path_lbl)
        browse_ref_btn = QPushButton("Browse")
        browse_ref_btn.clicked.connect(self._browse_reference)
        ref_l.addWidget(browse_ref_btn)
        clear_ref_btn = QPushButton("Clear")
        clear_ref_btn.clicked.connect(self._clear_reference)
        ref_l.addWidget(clear_ref_btn)
        ref_l.addStretch()
        ctrl_layout.addWidget(self._ref_widget, row, 0, 1, 3)
        self._ref_widget.setVisible(False)
        row += 1

        # Lip transfer toggle row
        lip_row = self._ctrl_row()
        lip_l = lip_row.layout()
        self._lip_btn = QPushButton("Enable Lip Transfer")
        self._lip_btn.setEnabled(False)
        self._lip_btn.clicked.connect(self._toggle_lip)
        lip_l.addWidget(self._lip_btn)
        lip_l.addStretch()
        ctrl_layout.addWidget(lip_row, row, 0, 1, 3)
        row += 1

        # Action buttons row
        btn_row = self._ctrl_row()
        btn_row.layout().setContentsMargins(0, 4, 0, 0)
        btn_l = btn_row.layout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("accent")
        self._start_btn.setMinimumWidth(90)
        self._start_btn.clicked.connect(self._toggle_start)
        btn_l.addWidget(self._start_btn)
        self._vcam_btn = QPushButton("Enable Virtual Cam")
        self._vcam_btn.setEnabled(False)
        self._vcam_btn.clicked.connect(self._toggle_vcam)
        btn_l.addWidget(self._vcam_btn)
        self._fs_btn = QPushButton("Fullscreen (F11)")
        self._fs_btn.clicked.connect(self._toggle_fullscreen)
        btn_l.addWidget(self._fs_btn)
        self._vcam_err_lbl = QLabel()
        self._vcam_err_lbl.setObjectName("err")
        btn_l.addWidget(self._vcam_err_lbl)
        btn_l.addStretch()
        ctrl_layout.addWidget(btn_row, row, 0, 1, 3)
        row += 1

        # Cycle controls (prev / pause / next) — shown only when prompt_cycle is set
        self._cycle_ctrl_row = self._ctrl_row()
        cyc_l = self._cycle_ctrl_row.layout()
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.clicked.connect(self._cycle_prev)
        cyc_l.addWidget(self._prev_btn)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._toggle_cycle_pause)
        cyc_l.addWidget(self._pause_btn)
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.clicked.connect(self._cycle_next)
        cyc_l.addWidget(self._next_btn)
        self._cycle_lbl = QLabel("")
        cyc_l.addWidget(self._cycle_lbl)
        cyc_l.addStretch()
        self._cycle_ctrl_row.setVisible(False)
        ctrl_layout.addWidget(self._cycle_ctrl_row, row, 0, 1, 3)
        row += 1

        # Advanced generation controls — hidden unless show_advanced_controls
        self._advanced_group = self._build_advanced_controls()
        ctrl_layout.addWidget(self._advanced_group, row, 0, 1, 3)
        row += 1

        self.statusBar().showMessage("Ready.")

        root_layout.addWidget(video_root, stretch=1)
        root_layout.addWidget(ctrl_area, stretch=0)

        self._refresh_cameras()

    @staticmethod
    def _ctrl_row() -> QWidget:
        """A transparent-background row container for the control panel."""
        w = QWidget()
        w.setObjectName("ctrl_row")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        return w

    def _make_video_panel(
        self, parent_layout: QHBoxLayout, title: str
    ) -> tuple[QWidget, QLabel]:
        panel = QWidget()
        panel.setObjectName("video_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("video_title")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setFixedHeight(22)
        layout.addWidget(title_lbl)

        img_lbl = QLabel("No signal")
        img_lbl.setObjectName("video_lbl")
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        img_lbl.setMinimumSize(240, 135)
        layout.addWidget(img_lbl)

        parent_layout.addWidget(panel)
        panel._title_lbl = title_lbl
        return panel, img_lbl

    # ── config ─────────────────────────────────────────────────────────────────

    def _load_config_meta(self, path: str) -> None:  # noqa: C901
        try:
            with open(path) as f:
                cfg = json.load(f)
            self._use_ref_image = cfg.get("use_reference_image", False)
            self._lip_transfer_in_config = cfg.get("lip_transfer", {}).get(
                "enable", False
            )
            res = cfg.get("resolution", {})
            self._cfg_w = res.get("width", 576)
            self._cfg_h = res.get("height", 320)
            log(
                f"Config loaded: {path}  res={self._cfg_w}x{self._cfg_h}  use_ref={self._use_ref_image}  lip_transfer={self._lip_transfer_in_config}"
            )
            self._ref_widget.setVisible(self._use_ref_image)
            self._lip_btn.setEnabled(self._lip_transfer_in_config)
            self._prompt_cycle = cfg.get("prompt_cycle", []) or []
            self._cycle_interval_ms = int(cfg.get("prompt_cycle_interval_s", 6) * 1000)
            self._cycle_ctrl_row.setVisible(bool(self._prompt_cycle))
            self._smooth_alpha = float(cfg.get("input_smoothing_alpha", 1.0))
            # Virtual webcam (pyvirtualcam/OBS) output. Off for Spout-only / kiosk
            # setups — hides the button and the OBS-not-found error label, and
            # skips the auto-start that produces that error.
            self._show_vcam = bool(cfg.get("show_virtual_cam", True))
            self._vcam_btn.setVisible(self._show_vcam)
            self._vcam_err_lbl.setVisible(self._show_vcam)
            # Advanced generation controls (dev/experiment only).
            self._show_advanced = bool(cfg.get("show_advanced_controls", False))
            self._advanced_group.setVisible(self._show_advanced)
            if self._show_advanced:
                self._adv_steps.blockSignals(True)
                self._adv_steps.setValue(int(cfg.get("default_steps", 2)))
                self._adv_steps.blockSignals(False)
                self._adv_seed.blockSignals(True)
                self._adv_seed.setValue(int(cfg.get("default_seed", 52)))
                self._adv_seed.blockSignals(False)
                # RIFE enable + factor reflect the config's interpolation_exp.
                cfg_interp = int(cfg.get("interpolation_exp", 2))
                rife_on = cfg_interp > 0
                self._adv_rife_on.blockSignals(True)
                self._adv_rife_on.setChecked(rife_on)
                self._adv_rife_on.blockSignals(False)
                self._adv_interp.blockSignals(True)
                self._adv_interp.setValue(cfg_interp if rife_on else 2)
                self._adv_interp.setEnabled(rife_on)
                self._adv_interp.blockSignals(False)
                # Flow-upscaler toggle only matters if the capability is loaded.
                self._adv_flow_up.setVisible(bool(cfg.get("enable_flow_upscaler", False)))
            default_prompt = cfg.get("default_prompt", "")
            if self._prompt_cycle:
                # Cycle mode: prompt box mirrors the active cycle prompt (read-only).
                default_prompt = self._prompt_cycle[0]
                self._prompt_edit.setReadOnly(True)
            if default_prompt and not self._prompt_edit.toPlainText().strip():
                self._prompt_edit.blockSignals(True)
                self._prompt_edit.setPlainText(default_prompt)
                self._prompt_edit.blockSignals(False)
        except Exception as exc:
            log(f"Config read error: {exc}")
            self._cfg_w, self._cfg_h = 576, 320

    # ── cameras ────────────────────────────────────────────────────────────────

    def _refresh_cameras(self) -> None:
        log("Scanning for cameras…")
        cams = enumerate_cameras()
        self._cam_combo.clear()
        # Prefer cameras that actually deliver a picture; idle NDI / virtual
        # cams (black frames) are skipped. If nothing is live, fall back to
        # listing everything so the app stays usable.
        live = [(i, is_live) for (i, is_live) in cams if is_live]
        entries = live if live else cams
        if entries:
            for idx, is_live in entries:
                label = f"Camera {idx}" if is_live else f"Camera {idx} (no signal)"
                self._cam_combo.addItem(label, idx)
            self._cam_combo.setCurrentIndex(0)  # first live camera
            self._cam_err_lbl.setText("")
            skipped = [i for (i, is_live) in cams if not is_live] if live else []
            msg = f"Cameras found (live): {[i for (i, _) in entries]}"
            if skipped:
                msg += f"  | skipped no-signal: {skipped}"
            log(msg)
        else:
            self._cam_err_lbl.setText("No cameras found")
            log("No cameras found")

    def _selected_cam_index(self) -> int | None:
        idx = self._cam_combo.currentData()
        if idx is not None:
            return int(idx)
        # Fallback: parse a trailing integer from the label.
        val = self._cam_combo.currentText()
        if not val:
            return None
        for tok in reversed(val.split()):
            if tok.isdigit():
                return int(tok)
        return None

    # ── prompt / reference ─────────────────────────────────────────────────────

    # ── fullscreen ───────────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self._exit_fullscreen()
        else:
            # Show ONLY the processed output: hide controls, the input panel, the
            # "Output" caption, and the status bar (the blue cycle x/N bar).
            self._ctrl_area.hide()
            self._input_panel.hide()
            self._output_panel._title_lbl.hide()
            self.statusBar().hide()
            self.showFullScreen()
            self._fs_btn.setText("Exit Fullscreen (Esc)")

    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        self._ctrl_area.show()
        self._input_panel.show()
        self._output_panel._title_lbl.show()
        self.statusBar().show()
        self._fs_btn.setText("Fullscreen (F11)")

    # ── advanced generation controls ─────────────────────────────────────────

    def _send_gen(self, name: str, value) -> None:
        """Push a live advanced-control change to the running stream processor
        (no-op until a stream is running)."""
        if self._sp is not None:
            self._sp.set_gen_param(name, value)

    def _build_advanced_controls(self) -> QGroupBox:
        # Checkable group title acts as a collapse/expand toggle; the inner
        # widget (all the knobs) is hidden by default so the panel stays tidy.
        group = QGroupBox("Advanced (generation)")
        group.setVisible(False)
        group.setCheckable(True)
        group.setChecked(False)
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 4, 8, 8)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        outer.addWidget(inner)
        inner.setVisible(False)
        group.toggled.connect(inner.setVisible)
        self._adv_row = 0

        def add_row(label: str, widget) -> None:
            grid.addWidget(
                QLabel(label),
                self._adv_row,
                0,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            grid.addWidget(widget, self._adv_row, 1)
            self._adv_row += 1

        # Steps
        self._adv_steps = QSpinBox()
        self._adv_steps.setRange(1, 8)
        self._adv_steps.setValue(2)
        self._adv_steps.valueChanged.connect(lambda v: self._send_gen("steps", int(v)))
        add_row("Steps", self._adv_steps)

        # Seed + reroll
        seed_row = QWidget()
        seed_l = QHBoxLayout(seed_row)
        seed_l.setContentsMargins(0, 0, 0, 0)
        self._adv_seed = QSpinBox()
        self._adv_seed.setRange(0, 2_147_483_647)
        self._adv_seed.setValue(52)
        self._adv_seed.valueChanged.connect(lambda v: self._send_gen("seed", int(v)))
        seed_l.addWidget(self._adv_seed, 1)
        reroll = QPushButton("Reroll")
        reroll.clicked.connect(self._reroll_seed)
        seed_l.addWidget(reroll)
        add_row("Seed", seed_row)

        # Dynamic shift toggle (when on, shift is auto from resolution/steps)
        self._adv_dynamic = QCheckBox("Dynamic shift (auto)")
        self._adv_dynamic.setChecked(True)
        self._adv_dynamic.toggled.connect(self._on_dynamic_shift_toggled)
        add_row("", self._adv_dynamic)

        # Static shift value (active only when dynamic shift is off)
        self._adv_shift = QDoubleSpinBox()
        self._adv_shift.setRange(0.1, 12.0)
        self._adv_shift.setSingleStep(0.1)
        self._adv_shift.setValue(3.0)
        self._adv_shift.setEnabled(False)
        self._adv_shift.valueChanged.connect(
            lambda v: self._send_gen("shift", float(v))
        )
        add_row("Shift", self._adv_shift)

        # Stochastic sampling
        self._adv_stochastic = QCheckBox("Stochastic sampling")
        self._adv_stochastic.toggled.connect(
            lambda c: self._send_gen("stochastic_sampling", bool(c))
        )
        add_row("", self._adv_stochastic)

        # Time shift type
        self._adv_timeshift = QComboBox()
        self._adv_timeshift.addItems(["exponential", "linear"])
        self._adv_timeshift.currentTextChanged.connect(
            lambda t: self._send_gen("time_shift_type", t)
        )
        add_row("Time shift", self._adv_timeshift)

        # Beta sigma spacing (community suggests Euler + Beta for Klein)
        self._adv_beta = QCheckBox("Beta sigmas")
        self._adv_beta.toggled.connect(
            lambda c: self._send_gen("use_beta_sigmas", bool(c))
        )
        add_row("", self._adv_beta)

        # Custom sigmas (comma-separated, descending; overrides steps)
        self._adv_sigmas = QLineEdit()
        self._adv_sigmas.setPlaceholderText(
            "auto — e.g. 1.0, 0.75, 0.5, 0.25 (overrides steps)"
        )
        self._adv_sigmas.editingFinished.connect(self._apply_sigmas)
        add_row("Sigmas", self._adv_sigmas)

        # ── RIFE frame interpolation ─────────────────────────────────────────
        # Interpolation factor: output = 2^exp displayed frames per generated
        # frame. Higher = smoother but more latency/warp. Buffer supports up to 8 (exp 3).
        # RIFE on/off + factor. "Enable RIFE" toggles interpolation entirely
        # (off = interp 0 = raw generated frames, no warp artifacts but choppier);
        # the spinbox sets the factor (2^exp output frames) when enabled.
        self._adv_rife_on = QCheckBox("Enable RIFE")
        self._adv_rife_on.setChecked(True)
        self._adv_rife_on.toggled.connect(self._on_rife_enabled)
        add_row("", self._adv_rife_on)

        self._adv_interp = QSpinBox()
        self._adv_interp.setRange(1, 3)
        self._adv_interp.setValue(2)
        self._adv_interp.valueChanged.connect(self._on_interp_changed)
        add_row("Interp 2^exp", self._adv_interp)

        # RIFE optical-flow scale — power-of-2 only (other values don't align
        # with the multi-scale pyramid). Lower = coarser flow (better for
        # large/fast motion). NOTE: recompiles RIFE on change when compile is on.
        self._adv_rife_scale = QComboBox()
        self._adv_rife_scale.addItems(["0.25", "0.5", "1.0", "2.0"])
        self._adv_rife_scale.setCurrentText("1.0")
        self._adv_rife_scale.currentTextChanged.connect(
            lambda t: self._send_gen("rife_scale", float(t))
        )
        add_row("Flow scale", self._adv_rife_scale)

        # Flow upscaler — live 2x output super-resolution. Only has an effect when
        # the config loads the capability (enable_flow_upscaler). Off by default.
        self._adv_flow_up = QCheckBox("Flow upscaler (2x output)")
        self._adv_flow_up.toggled.connect(
            lambda c: self._send_gen("flow_upscale_on", bool(c))
        )
        add_row("", self._adv_flow_up)

        return group

    def _reroll_seed(self) -> None:
        self._adv_seed.setValue(random.randint(0, 2_147_483_647))

    def _on_rife_enabled(self, on: bool) -> None:
        self._adv_interp.setEnabled(on)
        self._send_gen("interpolation_exp", self._adv_interp.value() if on else 0)

    def _on_interp_changed(self, v: int) -> None:
        if self._adv_rife_on.isChecked():
            self._send_gen("interpolation_exp", int(v))

    def _on_dynamic_shift_toggled(self, checked: bool) -> None:
        self._adv_shift.setEnabled(not checked)
        self._send_gen("use_dynamic_shifting", bool(checked))

    def _apply_sigmas(self) -> None:
        text = self._adv_sigmas.text().strip()
        if not text:
            self._send_gen("sigmas", None)
            return
        try:
            sigmas = [
                float(x) for x in text.replace(";", ",").split(",") if x.strip()
            ]
        except ValueError:
            self.statusBar().showMessage(
                "Invalid sigmas — use comma-separated numbers (e.g. 1.0, 0.5)"
            )
            return
        self._send_gen("sigmas", sigmas if sigmas else None)

    # ── prompt cycle ─────────────────────────────────────────────────────────────

    def _start_cycle_timer(self) -> None:
        if self._cycle_timer is None:
            self._cycle_timer = QTimer(self)
            self._cycle_timer.timeout.connect(self._advance_cycle)
        self._cycle_index = 0
        self._cycle_paused = False
        if hasattr(self, "_pause_btn"):
            self._pause_btn.setText("Pause")
        self._cycle_timer.start(self._cycle_interval_ms)
        self._update_cycle_label()
        log(f"Prompt cycle started: {len(self._prompt_cycle)} prompts, "
            f"{self._cycle_interval_ms/1000:.0f}s each")

    def _stop_cycle_timer(self) -> None:
        if self._cycle_timer is not None:
            self._cycle_timer.stop()

    def _update_cycle_label(self) -> None:
        if hasattr(self, "_cycle_lbl") and self._prompt_cycle:
            paused = "  (paused)" if self._cycle_paused else ""
            self._cycle_lbl.setText(
                f"[{self._cycle_index + 1}/{len(self._prompt_cycle)}]{paused}"
            )

    def _apply_cycle_index(self, idx: int) -> None:
        """Switch to cycle prompt `idx` (instant, pre-encoded) and sync the UI."""
        if self._sp is None or not self._prompt_cycle:
            return
        self._cycle_index = idx % len(self._prompt_cycle)
        prompt = self._prompt_cycle[self._cycle_index]
        self._sp.set_prompt_index(self._cycle_index)  # instant: swaps cached embeds
        self._prompt_edit.blockSignals(True)
        self._prompt_edit.setPlainText(prompt)
        self._prompt_edit.blockSignals(False)
        self._update_cycle_label()
        self.statusBar().showMessage(
            f"Cycle [{self._cycle_index + 1}/{len(self._prompt_cycle)}]: {prompt[:60]}"
        )
        log(f"cycle -> [{self._cycle_index}] {prompt}")

    def _advance_cycle(self) -> None:
        # Wait until the model is ready (prompts pre-encoded) before switching.
        if self._sp is None:
            return
        try:
            if not self._sp.is_ready():
                return
        except Exception:
            return
        self._apply_cycle_index(self._cycle_index + 1)

    def _restart_cycle_interval(self) -> None:
        # After a manual step, restart the timer so the next auto-advance is a
        # full interval away (no effect while paused).
        if self._cycle_timer is not None and not self._cycle_paused:
            self._cycle_timer.start(self._cycle_interval_ms)

    def _cycle_next(self) -> None:
        self._apply_cycle_index(self._cycle_index + 1)
        self._restart_cycle_interval()

    def _cycle_prev(self) -> None:
        self._apply_cycle_index(self._cycle_index - 1)
        self._restart_cycle_interval()

    def _toggle_cycle_pause(self) -> None:
        if self._cycle_timer is None:
            return
        if self._cycle_paused:
            self._cycle_paused = False
            self._pause_btn.setText("Pause")
            self._cycle_timer.start(self._cycle_interval_ms)
            self.statusBar().showMessage("Cycle resumed")
        else:
            self._cycle_paused = True
            self._pause_btn.setText("Resume")
            self._cycle_timer.stop()
            self.statusBar().showMessage("Cycle paused")
        self._update_cycle_label()

    def _on_prompt_changed(self) -> None:
        prompt = self._prompt_edit.toPlainText()
        if self._sp is not None:
            self._sp.set_prompt(prompt)
        preview = prompt[:70] + ("…" if len(prompt) > 70 else "")
        log(f"Prompt: {preview!r}")

    def _browse_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)",
        )
        if not path:
            return
        self._ref_full_path = path
        self._ref_path_lbl.setText(os.path.basename(path))
        log(f"Reference image selected: {path}")
        if self._sp is not None:
            self._apply_reference(path)

    def _clear_reference(self) -> None:
        self._ref_full_path = None
        self._ref_path_lbl.setText("(none)")
        if self._sp is not None and self._use_ref_image:
            try:
                self._sp.set_reference_image(None)
                log("Reference image cleared")
            except Exception as exc:
                log(f"Error clearing reference image: {exc}")

    def _apply_reference(self, path: str) -> None:
        try:
            arr = np.array(Image.open(path).convert("RGB"))
            self._sp.set_reference_image(arr)
            log(f"Reference image applied: {path}")
        except Exception as exc:
            log(f"Reference image load error: {exc}")

    def _toggle_lip(self) -> None:
        self._lip_active = not self._lip_active
        self._lip_btn.setText(
            "Disable Lip Transfer" if self._lip_active else "Enable Lip Transfer"
        )
        if self._sp is not None:
            self._sp.set_lip_transfer(self._lip_active)
        log(f"Lip transfer: {'on' if self._lip_active else 'off'}")

    # ── start / stop ───────────────────────────────────────────────────────────

    def _toggle_start(self) -> None:
        if self._running or self._sp_loading:
            self._stop_capture()
        else:
            self._begin_start()

    def _begin_start(self) -> None:
        spout_name = self._spout_input_edit.text().strip() if _SPOUT_AVAILABLE else ""

        if spout_name:
            # Spout input path — no camera needed
            cap, cam_idx = None, -1
            self._cam_err_lbl.setText("")
        else:
            # Camera path
            cam_idx = self._selected_cam_index()
            if cam_idx is None:
                self._cam_err_lbl.setText("No camera selected")
                log("Start aborted: no camera selected")
                return
            cap = cv2.VideoCapture(cam_idx, CAM_BACKEND)
            if not cap.isOpened():
                cap.release()
                self._cam_err_lbl.setText("Cannot open camera")
                log(f"Start aborted: cannot open camera {cam_idx}")
                return
            self._cam_err_lbl.setText("")

        if self._sp is None:
            self._sp_loading = True
            self._start_btn.setText("Loading…")
            self._start_btn.setEnabled(False)
            self.statusBar().showMessage("Loading model…")
            log("Loading StreamProcessor (this may take a while)")
            prompt = self._prompt_edit.toPlainText()
            threading.Thread(
                target=self._init_sp_thread,
                args=(cap, cam_idx, prompt, spout_name),
                daemon=True,
            ).start()
        else:
            self._on_launch_capture(cap, cam_idx, spout_name)

    def _init_sp_thread(
        self, cap, cam_idx: int, prompt: str, spout_name: str = ""
    ) -> None:
        try:
            sp = StreamProcessor(self.config_path)
            if self._use_int8:
                sp.enable_quantization()
            sp.start()
            # In cycle mode the subprocess pre-encodes the cycle prompts and starts
            # at index 0 — don't trigger an extra live encode here.
            if not self._prompt_cycle:
                sp.set_prompt(prompt)
            self._sp = sp
            self._input_tensor = sp.get_input_tensor()
            self._output_tensor = sp.get_output_tensor()
            self._resolution = sp.get_resolution()
            self._out_resolution = sp.get_out_resolution()
            log(
                f"StreamProcessor ready — resolution={self._resolution}  out_resolution={self._out_resolution}"
            )
            if self._ref_full_path and self._use_ref_image:
                self._apply_reference(self._ref_full_path)
            if self._lip_active:
                sp.set_lip_transfer(True)
            self._sig.launch_capture.emit(cap, cam_idx, spout_name)
        except Exception as exc:
            log(f"StreamProcessor init error: {exc}")
            if cap is not None:
                cap.release()
            self._sig.sp_error.emit(str(exc))

    @Slot(object, int, str)
    def _on_launch_capture(self, cap, cam_idx: int, spout_name: str = "") -> None:
        self._sp_loading = False
        self._capture_stop.clear()
        self._running = True
        if spout_name:
            self._capture_thread = threading.Thread(
                target=self._spout_input_loop, args=(spout_name,), daemon=True
            )
            status_msg = f"Running — Spout input: {spout_name}"
        else:
            self._capture_thread = threading.Thread(
                target=self._capture_loop, args=(cap,), daemon=True
            )
            status_msg = f"Running — camera {cam_idx}"
        self._capture_thread.start()
        self._start_btn.setText("Stop")
        self._start_btn.setEnabled(True)
        self._vcam_btn.setEnabled(True)
        self.statusBar().showMessage(status_msg)
        log(f"Capture started: {status_msg}")
        self._start_spout_output()
        if self._show_vcam:
            self._start_vcam()
        if self._prompt_cycle:
            self._start_cycle_timer()

    @Slot(str)
    def _on_sp_error(self, _err: str) -> None:
        self._sp_loading = False
        self._start_btn.setText("Start")
        self._start_btn.setEnabled(True)
        self.statusBar().showMessage("Model error — see terminal")

    def _stop_capture(self) -> None:
        self._stop_cycle_timer()
        self._capture_stop.set()
        if self._vcam_cam is not None:
            self._stop_vcam()
        if self._spout_sender is not None:
            self._stop_spout_output()
        self._running = False
        self._sp_loading = False
        self._start_btn.setText("Start")
        self._start_btn.setEnabled(True)
        self._vcam_btn.setEnabled(False)
        self.statusBar().showMessage("Stopped.")
        with self._frame_lock:
            self._latest_input = self._latest_output = self._latest_output_bgr = None
        for lbl in (self._input_lbl, self._output_lbl):
            lbl.clear()
            lbl.setText("No signal")
        log("Capture stopped")

    # ── capture loop ───────────────────────────────────────────────────────────

    def _smooth_input(self, frame: np.ndarray) -> np.ndarray:
        """Exponential moving average over input frames for a dreamy ghost-trail.
        alpha=1.0 → passthrough; lower → more smoothing/trailing on motion."""
        if self._smooth_alpha >= 1.0:
            return frame
        f = frame.astype(np.float32)
        if self._smoothed_frame is None:
            self._smoothed_frame = f
        else:
            self._smoothed_frame = (
                self._smooth_alpha * f + (1.0 - self._smooth_alpha) * self._smoothed_frame
            )
        return np.ascontiguousarray(self._smoothed_frame.astype(np.uint8))

    def _capture_loop(self, cap) -> None:
        h = self._resolution["height"]
        w = self._resolution["width"]
        self._smoothed_frame = None  # reset trail history on each capture start
        try:
            while not self._capture_stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log("Camera read error — stopping capture")
                    self._sig.camera_error.emit()
                    break
                cropped = crop_maximal_rectangle(frame, h, w)
                cropped = self._smooth_input(cropped)
                with _sp_lock:
                    self._input_tensor.copy_from(cropped)
                    output_bgr = self._output_tensor.to_numpy()
                input_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                with self._frame_lock:
                    self._latest_input = input_rgb
                    self._latest_output = output_rgb
                    self._latest_output_bgr = output_bgr
        finally:
            cap.release()

    @Slot()
    def _on_camera_error(self) -> None:
        self._cam_err_lbl.setText("Camera disconnected")
        self._running = False
        self._start_btn.setText("Start")
        self._start_btn.setEnabled(True)
        self._vcam_btn.setEnabled(False)
        self.statusBar().showMessage("Camera error — see terminal")

    # ── spout input loop ───────────────────────────────────────────────────────

    def _spout_input_loop(self, sender_name: str) -> None:
        h = self._resolution["height"]
        w = self._resolution["width"]
        receiver = SpoutGL.SpoutReceiver()
        receiver.setReceiverName(sender_name)
        recv_w, recv_h = w, h
        frame = np.empty((recv_h, recv_w, 3), dtype=np.uint8)
        try:
            while not self._capture_stop.is_set():
                if receiver.isUpdated():
                    recv_w = receiver.getSenderWidth()
                    recv_h = receiver.getSenderHeight()
                    frame = np.empty((recv_h, recv_w, 3), dtype=np.uint8)
                result = receiver.receiveImage(frame, 0x1907, False, 0)
                if result:
                    frame = np.ascontiguousarray(frame)
                    # Spout delivers RGB; pipeline expects BGR
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cropped = crop_maximal_rectangle(frame_bgr, h, w)
                    with _sp_lock:
                        self._input_tensor.copy_from(cropped)
                        output_bgr = self._output_tensor.to_numpy()
                    input_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                    with self._frame_lock:
                        self._latest_input = input_rgb
                        self._latest_output = output_rgb
                        self._latest_output_bgr = output_bgr
                else:
                    time.sleep(0.005)
        finally:
            receiver.releaseReceiver()

    # ── spout output ───────────────────────────────────────────────────────────

    def _start_spout_output(self) -> None:
        if not _SPOUT_AVAILABLE or self._spout_sender is not None:
            return
        try:
            sender = SpoutGL.SpoutSender()
            sender.setSenderName("FluxRTOutput")
            self._spout_sender = sender
            self._spout_sender_stop.clear()
            self._spout_sender_thread = threading.Thread(
                target=self._spout_output_loop, daemon=True
            )
            self._spout_sender_thread.start()
            log("Spout output started: FluxRTOutput")
        except Exception as exc:
            log(f"Spout output init error: {exc}")

    def _stop_spout_output(self) -> None:
        self._spout_sender_stop.set()
        if self._spout_sender_thread is not None:
            self._spout_sender_thread.join(timeout=2)
            self._spout_sender_thread = None
        if self._spout_sender is not None:
            try:
                self._spout_sender.releaseSender()
            except Exception:
                pass
            self._spout_sender = None
        log("Spout output stopped")

    def _spout_output_loop(self) -> None:
        sender = self._spout_sender
        try:
            while not self._spout_sender_stop.is_set():
                with self._frame_lock:
                    frame_bgr = self._latest_output_bgr
                if frame_bgr is not None:
                    fh, fw = frame_bgr.shape[:2]
                    frame_rgb = np.ascontiguousarray(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    )
                    try:
                        sender.sendImage(frame_rgb, fw, fh, 0x1907, False, 0)
                    except Exception as exc:
                        log(f"Spout send error: {exc}")
                        break
                else:
                    time.sleep(0.01)
        finally:
            pass

    # ── virtual camera ─────────────────────────────────────────────────────────

    def _toggle_vcam(self) -> None:
        if self._vcam_cam is not None:
            self._stop_vcam()
        else:
            self._start_vcam()

    def _start_vcam(self) -> None:
        if self._out_resolution is None:
            return
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat

            vcam = pyvirtualcam.Camera(
                width=self._out_resolution["width"],
                height=self._out_resolution["height"],
                fps=30,
                fmt=PixelFormat.BGR,
            )
            self._vcam_cam = vcam
            self._vcam_err_lbl.setText("")
            self._vcam_btn.setText("Disable Virtual Cam")
            self._vcam_stop.clear()
            self._vcam_thread = threading.Thread(target=self._vcam_loop, daemon=True)
            self._vcam_thread.start()
            log(f"Virtual camera started: {vcam.device}")
            self.statusBar().showMessage(f"Virtual cam active: {vcam.device}")
        except Exception as exc:
            self._vcam_err_lbl.setText(f"VCam error: {exc}")
            log(f"Virtual camera error: {exc}")

    def _stop_vcam(self) -> None:
        self._vcam_stop.set()
        if self._vcam_thread is not None:
            self._vcam_thread.join(timeout=2)
            self._vcam_thread = None
        if self._vcam_cam is not None:
            try:
                self._vcam_cam.__exit__(None, None, None)
            except Exception:
                pass
            self._vcam_cam = None
        self._vcam_btn.setText("Enable Virtual Cam")
        self._vcam_err_lbl.setText("")
        self.statusBar().showMessage("Running.")
        log("Virtual camera stopped")

    def _vcam_loop(self) -> None:
        vcam = self._vcam_cam
        try:
            while not self._vcam_stop.is_set():
                with self._frame_lock:
                    frame = self._latest_output_bgr
                if frame is not None:
                    try:
                        vcam.send(frame)
                        vcam.sleep_until_next_frame()
                    except Exception as exc:
                        log(f"VCam send error: {exc}")
                        self._sig.vcam_error.emit("VCam send error — see terminal")
                        break
                else:
                    time.sleep(0.01)
        finally:
            pass

    @Slot(str)
    def _on_vcam_error(self, msg: str) -> None:
        self._vcam_err_lbl.setText(msg)

    # ── frame rendering ────────────────────────────────────────────────────────

    def _poll_frames(self) -> None:
        with self._frame_lock:
            inp = self._latest_input
            out = self._latest_output
        self._render_frame(self._input_lbl, inp)
        self._render_frame(self._output_lbl, out)

    def _render_frame(self, label: QLabel, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        lw = label.width()
        lh = label.height()
        if lw < 10:
            lw = self._cfg_w
        if lh < 10:
            lh = self._cfg_h
        h, w = frame.shape[:2]
        scale = min(lw / w, lh / h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        arr = np.ascontiguousarray(frame)
        if (nw, nh) != (w, h):
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # .copy() detaches QImage from the numpy buffer before it's GC'd
        qimg = QImage(arr.data, nw, nh, nw * 3, QImage.Format.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(qimg))

    # ── close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        log("Shutting down")
        self._stop_cycle_timer()
        self._poll_timer.stop()
        self._capture_stop.set()
        self._vcam_stop.set()
        self._spout_sender_stop.set()
        if self._vcam_cam is not None:
            try:
                self._vcam_cam.__exit__(None, None, None)
            except Exception:
                pass
        if self._spout_sender is not None:
            try:
                self._spout_sender.releaseSender()
            except Exception:
                pass
        if self._sp is not None:
            try:
                self._sp.stop()
            except Exception:
                pass
        event.accept()


# ── entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="FluxRT GUI")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to StreamProcessor config JSON (default: %(default)s)",
    )
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args, _ = parser.parse_known_args()

    app = QApplication([])
    app.setStyleSheet(STYLESHEET)
    win = MainWindow(config_path=args.config, use_int8=args.int8)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
