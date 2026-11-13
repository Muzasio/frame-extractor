#!/usr/bin/env python3
"""FrameExtract: pick one exact frame from a video and save it as a lossless PNG."""
import os
import sys
import threading
import traceback

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QSlider, QVBoxLayout,
                             QWidget)

APP_ID = "frameextract"
DEFAULT_FPS = 30.0
BESTSOURCE_TIMEOUT = 6      # seconds before falling back to OpenCV
MIN_ZOOM_SPAN = 10          # smallest timeline window, in frames
VIDEO_FILTER = "Video files (*.mp4 *.mkv *.avi *.mov *.webm)"


def safe_fps(value):
    """Return a usable frame rate; bad values (0, negative, NaN) become DEFAULT_FPS."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_FPS
    return value if value > 0 and value == value else DEFAULT_FPS


def cache_dir():
    """Per-user cache folder for BestSource index files (~/.cache/frameextract/bsindex)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(base, APP_ID, "bsindex")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def clamp(value, low, high):
    return max(low, min(high, value))


def zoom_span(span, wheel_delta, total_frames):
    """New timeline width after one wheel step (positive delta = zoom in)."""
    last = total_frames - 1
    if wheel_delta > 0:
        return min(last, max(MIN_ZOOM_SPAN, int(span * 0.6)))
    return min(last, int(span * 1.6) if span > 0 else 200)


def window_around(center, span, total_frames):
    """(start, end) of a timeline window of width `span` centred on `center`, kept inside the video."""
    last = total_frames - 1
    span = clamp(span, 0, last)
    start = clamp(center - span // 2, 0, last)
    end = min(last, start + span)
    start = max(0, end - span)
    return start, end


class BestSourceBackend:
    """Frame-exact reader (VapourSynth + BestSource). Indexes the file on first open."""

    def __init__(self, path):
        import vapoursynth as vs
        clip = vs.core.bs.VideoSource(source=path, cachepath=cache_dir())
        # Tagged videos use their own colour matrix (the frame tag wins).
        # Untagged ones get the usual guess: BT.709 for HD, BT.601 for SD.
        guess = "709" if clip.height > 576 else "170m"
        self.clip = vs.core.resize.Bicubic(clip, format=vs.RGB24, matrix_in_s=guess)
        self.total_frames = len(self.clip)
        self.fps = safe_fps(clip.fps_num / clip.fps_den) if clip.fps_den else DEFAULT_FPS
        self.width = self.clip.width
        self.height = self.clip.height

    def get_frame_rgb(self, idx):
        idx = clamp(idx, 0, self.total_frames - 1)
        vsframe = self.clip.get_frame(idx)
        return np.dstack([np.asarray(vsframe[p]) for p in range(3)])

    def release(self):
        self.clip = None


class OpenCVBackend:
    """Fallback reader. Seeks, checks which frame it landed on, and corrects the drift."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = safe_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame_rgb(self, idx, max_retries=3):
        last = self.total_frames - 1
        idx = clamp(idx, 0, last)
        target = idx
        frame = None
        for _ in range(max_retries):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = self.cap.read()
            if not ok:
                return None
            # POS_FRAMES after read() reports the *next* frame to be read
            actual = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if actual == idx:
                break
            target = clamp(target + (idx - actual), 0, last)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


def open_backend(path, bestsource_timeout=BESTSOURCE_TIMEOUT):
    """Try BestSource first; fall back to OpenCV if it is missing, fails or is too slow.

    A timed-out BestSource thread cannot be stopped; it is a daemon thread, so it
    finishes its index in the background (next open is fast) or dies with the app.
    """
    result = {}

    def target():
        try:
            result["backend"] = BestSourceBackend(path)
        except Exception as e:  # missing plugin, unsupported file, unwritable cache...
            result["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=bestsource_timeout)

    if t.is_alive():
        backend, name = OpenCVBackend(path), "OpenCV (BestSource timed out, fell back)"
    elif "backend" in result:
        backend, name = result["backend"], "BestSource (frame-accurate)"
    else:
        backend, name = OpenCVBackend(path), "OpenCV (seek-verify fallback)"

    if backend.total_frames < 1:
        backend.release()
        raise RuntimeError("the video has no readable frames")
    return backend, name


class LoaderThread(QThread):
    loaded = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            backend, name = open_backend(self.path)
            self.loaded.emit(backend, name)
        except Exception as e:
            self.failed.emit(str(e))


class ZoomSlider(QSlider):
    """Slider whose mouse wheel zooms the timeline instead of moving the handle."""
    wheel_zoom = pyqtSignal(int)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self.wheel_zoom.emit(delta)
        event.accept()


class FrameExtractor(QWidget):
    def __init__(self, video_path=None):
        super().__init__()
        self.setAcceptDrops(True)

        self.backend = None
        self.backend_name = ""
        self.video_name = ""
        self.total_frames = 0
        self.fps = DEFAULT_FPS
        self.view_start = 0
        self.view_end = 0
        self.current_frame = 0
        self.last_frame_rgb = None
        self.output_dir = ""
        self._loader = None
        self._pending_path = None

        self.setWindowTitle("FrameExtract - drop a video or click Open Video")
        self.resize(1000, 700)

        self.preview = QLabel("Drag a video here, or click Open Video (Ctrl+O)")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background-color: black; color: #888; font-size: 16px;")
        self.preview.setMinimumSize(1, 1)   # let the window shrink below the pixmap size

        self.open_btn = QPushButton("Open Video")
        self.open_btn.clicked.connect(self.browse_video)

        self.output_btn = QPushButton("Output Folder")
        self.output_btn.clicked.connect(self.choose_output_dir)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.open_btn)
        top_bar.addWidget(self.output_btn)
        top_bar.addStretch()

        self.slider = ZoomSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.wheel_zoom.connect(self.on_wheel_zoom)
        self.slider.setEnabled(False)

        self.status = QLabel("no video loaded")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.preview, stretch=1)
        layout.addWidget(self.slider)
        layout.addWidget(self.status)
        self.setLayout(layout)

        shortcuts = [
            (Qt.Key.Key_Left, lambda: self.step(-1)),
            (Qt.Key.Key_Right, lambda: self.step(1)),
            ("Shift+Left", lambda: self.step(-10)),
            ("Shift+Right", lambda: self.step(10)),
            (Qt.Key.Key_S, self.save_frame),
            (Qt.Key.Key_Return, self.save_frame),
            ("Ctrl+O", self.browse_video),
        ]
        for key, action in shortcuts:
            QShortcut(QKeySequence(key), self).activated.connect(action)

        if video_path:
            self.load_new_video(video_path)

    # ---- loading ---------------------------------------------------------

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", VIDEO_FILTER)
        if path:
            self.load_new_video(path)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.load_new_video(urls[0].toLocalFile())

    def load_new_video(self, path):
        if self._loader is not None and self._loader.isRunning():
            return
        self._pending_path = path
        self.open_btn.setEnabled(False)
        self.preview.setText(f"Loading {os.path.basename(path)}...")
        self.status.setText("indexing video, this can take a few seconds on first open...")
        self._loader = LoaderThread(path)
        self._loader.loaded.connect(self._on_backend_loaded)
        self._loader.failed.connect(self._on_backend_failed)
        self._loader.start()

    def _on_backend_loaded(self, new_backend, backend_name):
        path = self._pending_path
        if self.backend is not None:
            self.backend.release()
        self.backend = new_backend
        self.backend_name = backend_name
        self.video_name = os.path.splitext(os.path.basename(path))[0]
        self.total_frames = self.backend.total_frames
        self.fps = self.backend.fps
        self.view_start = 0
        self.view_end = self.total_frames - 1
        self.current_frame = 0
        # Folder is only created when a frame is saved (the video folder may be read-only).
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "extracted_frames")
        self.setWindowTitle(f"FrameExtract - {os.path.basename(path)} [{backend_name}]")
        self.open_btn.setEnabled(True)
        self.slider.setEnabled(True)
        self._set_slider_range(self.view_start, self.view_end, value=0)
        self.show_frame(0)

    def _on_backend_failed(self, message):
        self.open_btn.setEnabled(True)
        self.preview.setText("Drag a video here, or click Open Video (Ctrl+O)")
        self.status.setText("no video loaded")
        QMessageBox.critical(self, "Error", f"Cannot open video: {self._pending_path}\n{message}")

    # ---- timeline --------------------------------------------------------

    def _set_slider_range(self, start, end, value=None):
        self.view_start, self.view_end = start, end
        self.slider.blockSignals(True)
        self.slider.setRange(start, end)
        if value is not None:
            self.slider.setValue(value)
        self.slider.blockSignals(False)

    def on_wheel_zoom(self, delta):
        if self.backend is None:
            return
        span = zoom_span(self.view_end - self.view_start, delta, self.total_frames)
        start, end = window_around(self.current_frame, span, self.total_frames)
        self._set_slider_range(start, end, value=self.current_frame)
        self.update_status()

    def on_slider_change(self, value):
        self.current_frame = value
        self.show_frame(value)

    def step(self, delta):
        if self.backend is None:
            return
        new_frame = clamp(self.current_frame + delta, 0, self.total_frames - 1)
        if not self.view_start <= new_frame <= self.view_end:
            start, end = window_around(new_frame, self.view_end - self.view_start, self.total_frames)
            self._set_slider_range(start, end)
        self.slider.setValue(new_frame)

    # ---- preview ---------------------------------------------------------

    def show_frame(self, frame_idx):
        if self.backend is None:
            return
        rgb = self.backend.get_frame_rgb(frame_idx)
        if rgb is None:
            self.status.setText(f"could not read frame {frame_idx}")
            return
        self.last_frame_rgb = np.ascontiguousarray(rgb)
        self._render_preview()
        self.update_status()

    def _render_preview(self):
        """Scale the cached frame to the preview size (no new decode)."""
        rgb = self.last_frame_rgb
        if rgb is None:
            return
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pixmap)

    def update_status(self, note=""):
        mm, ss = divmod(self.current_frame / self.fps, 60)
        parts = [
            f"frame {self.current_frame}/{self.total_frames - 1}",
            f"{int(mm):02d}:{ss:05.2f}",
            f"zoom [{self.view_start}-{self.view_end}]",
            self.backend_name,
            f"out: {self.output_dir}",
        ]
        if note:
            parts.insert(0, note)
        self.status.setText("  |  ".join(parts))

    # ---- saving ----------------------------------------------------------

    def save_frame(self):
        if self.backend is None or self.last_frame_rgb is None:
            return
        filename = f"{self.video_name}_frame{self.current_frame:06d}.png"
        path = os.path.join(self.output_dir, filename)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            bgr = cv2.cvtColor(self.last_frame_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                raise OSError("the image writer returned an error")
        except OSError as e:
            QMessageBox.warning(self, "Save failed",
                                f"Could not save {path}\n{e}\n\nPick another folder with Output Folder.")
            return
        h, w = self.last_frame_rgb.shape[:2]
        self.update_status(note=f"saved {filename} ({w}x{h}px)")

    def choose_output_dir(self):
        start_dir = self.output_dir or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "Output folder", start_dir)
        if d:
            self.output_dir = d
            if self.backend is not None:
                self.update_status()

    # ---- window events ---------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_preview()

    def closeEvent(self, event):
        if self._loader is not None and self._loader.isRunning():
            # A QThread destroyed while running crashes Python; let it finish first.
            self.hide()
            self._loader.loaded.disconnect()
            self._loader.failed.disconnect()
            self._loader.wait()
        if self.backend is not None:
            self.backend.release()
            self.backend = None
        super().closeEvent(event)


def _excepthook(exc_type, exc, tb):
    """Print errors instead of letting PyQt abort the whole app on an uncaught exception."""
    traceback.print_exception(exc_type, exc, tb)


def main():
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    app.setApplicationName("FrameExtract")
    app.setDesktopFileName(APP_ID)   # Wayland: links the window to frameextract.desktop (taskbar icon)
    app.setWindowIcon(QIcon.fromTheme(APP_ID))
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    win = FrameExtractor(video_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
