# scripts/draw_fence_gui.py
"""
Interactive GUI tool to draw no-parking zones on a video frame.

Supports local video files and RTSP URLs. RTSP auto-loading is asynchronous so
the window appears immediately while the first frame is being decoded.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional, Tuple

from PIL import Image, ImageTk


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_FILE = os.path.join(PROJECT_ROOT, "no_parking_config.json")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")


class FenceDrawingApp:
    def __init__(
        self,
        root: tk.Tk,
        auto_load_source: Optional[str] = None,
        zone_key: Optional[str] = None,
    ):
        self.root = root
        self.root.title("Draw No-Parking Zone")

        self.canvas_width = 800
        self.canvas_height = 600

        self.video_path = None
        self.zone_key = zone_key
        self.frame = None
        self.display_photo = None
        self.polygons: List[List[Tuple[int, int]]] = []
        self.points: List[Tuple[int, int]] = []
        self.drawing = False
        self.auto_loaded = False
        self.loading_source = False
        self.auto_start_after_load = False
        self.load_result_queue = queue.Queue()
        self._load_poll_after_id = None

        self.scale_ratio = 1.0
        self.display_width = self.canvas_width
        self.display_height = self.canvas_height
        self.offset_x = 0
        self.offset_y = 0
        self.last_cursor_pos = (0, 0)

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#777777",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=(12, 8))

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0, 4))

        tk.Button(btn_frame, text="Select Video", command=self.load_video).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Start Drawing", command=self.start_drawing).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Save Zone", command=self.save_zone).grid(row=0, column=2, padx=5)
        tk.Button(
            btn_frame,
            text="Finish & Exit",
            command=self.finish_and_exit,
            bg="#4CAF50",
            fg="white",
        ).grid(row=0, column=3, padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear_points).grid(row=0, column=4, padx=5)

        self.status_label = tk.Label(root, text="Please select a video.", fg="blue")
        self.status_label.pack(pady=(0, 10))

        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Button-2>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        if auto_load_source:
            self._set_status("Opening source...", "blue")
            self.root.after(
                50,
                lambda: self._start_async_load(auto_load_source, auto_start=True),
            )

    def _set_status(self, text: str, color: str = "blue", bold: bool = False):
        font = ("Arial", 10, "bold") if bold else ("Arial", 10)
        self.status_label.config(text=text, fg=color, font=font)

    def _display_name(self, source: str) -> str:
        return self.zone_key or os.path.basename(source) or source

    def _start_async_load(self, source: str, auto_start: bool):
        self.video_path = source
        self.loading_source = True
        self.auto_start_after_load = auto_start
        self.polygons = []
        self.points = []
        self.drawing = False

        while not self.load_result_queue.empty():
            try:
                self.load_result_queue.get_nowait()
            except queue.Empty:
                break

        display_name = self._display_name(source)
        self._set_status(f"Opening source for {display_name}...", "blue", bold=True)
        threading.Thread(
            target=self._load_source_worker,
            args=(source,),
            daemon=True,
        ).start()
        self._poll_load_result()

    def _load_source_worker(self, source: str):
        started = time.perf_counter()
        try:
            frame = self._read_first_frame(source)
            elapsed = time.perf_counter() - started
            self.load_result_queue.put(("ok", source, frame, elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - started
            self.load_result_queue.put(("error", source, str(e), elapsed))

    def _poll_load_result(self):
        try:
            result_type, source, payload, elapsed = self.load_result_queue.get_nowait()
        except queue.Empty:
            if self.loading_source:
                self._load_poll_after_id = self.root.after(100, self._poll_load_result)
            return

        self._load_poll_after_id = None
        if result_type == "ok":
            self._apply_loaded_frame(source, payload, elapsed)
        else:
            self._handle_load_error(payload)

    def _is_rtsp_source(self, source: str) -> bool:
        return source.lower().startswith(("rtsp://", "rtsps://"))

    def _open_capture(self, source: str):
        import cv2

        if not self._is_rtsp_source(source):
            return cv2.VideoCapture(source)

        timeout_ms = int(os.getenv("FENCE_GUI_RTSP_TIMEOUT_MS", "8000"))
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;tcp|stimeout;{timeout_ms * 1000}|max_delay;500000",
        )

        params = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])

        if params:
            try:
                return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
            except Exception:
                pass
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    def _read_first_frame(self, source: str):
        cap = self._open_capture(source)
        timeout_sec = float(os.getenv("FENCE_GUI_FRAME_TIMEOUT_SEC", "12"))
        deadline = time.perf_counter() + max(timeout_sec, 1.0)

        try:
            while time.perf_counter() < deadline:
                ret, frame = cap.read()
                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                    return frame
                time.sleep(0.05)
        finally:
            cap.release()

        raise RuntimeError(f"Cannot read first frame within {timeout_sec:.1f}s: {source}")

    def _configure_frame(self, frame):
        import cv2

        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.orig_height, self.orig_width = self.frame.shape[:2]

        scale_w = self.canvas_width / self.orig_width
        scale_h = self.canvas_height / self.orig_height
        self.scale_ratio = min(scale_w, scale_h)

        self.display_width = int(self.orig_width * self.scale_ratio)
        self.display_height = int(self.orig_height * self.scale_ratio)
        self.offset_x = (self.canvas_width - self.display_width) // 2
        self.offset_y = (self.canvas_height - self.display_height) // 2

    def _apply_loaded_frame(self, source: str, frame, elapsed: float):
        self.loading_source = False
        self._configure_frame(frame)
        display_name = self._display_name(source)
        self._set_status(
            f"Loaded {display_name} in {elapsed:.1f}s | Draw zone, then Finish & Exit",
            "purple",
            bold=True,
        )
        self.auto_loaded = True
        self.display_frame()
        if self.auto_start_after_load:
            self.start_drawing()

    def _handle_load_error(self, error: str):
        self.loading_source = False
        messagebox.showerror("Error", error)
        self._set_status("Failed to load auto-specified source", "red")

    def _auto_load_source(self, source: str):
        self._start_async_load(source, auto_start=True)

    def _auto_load_video(self, source: str):
        self._auto_load_source(source)

    def load_video(self):
        initial_dir = UPLOAD_FOLDER if os.path.exists(UPLOAD_FOLDER) else "."
        path = filedialog.askopenfilename(
            title="Select Video for Fence Drawing",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
            initialdir=initial_dir,
        )
        if path:
            self._start_async_load(path, auto_start=False)

    def display_frame(self):
        if self.frame is None:
            return

        img = Image.fromarray(self.frame).resize(
            (self.display_width, self.display_height),
            Image.Resampling.LANCZOS,
        )
        self.display_photo = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor=tk.NW,
            image=self.display_photo,
        )
        self.canvas.image = self.display_photo
        self.draw_polygon_overlay()

    def draw_polygon_overlay(self):
        if not self.polygons and len(self.points) < 1:
            return

        for poly in self.polygons:
            disp_poly = [
                (
                    int(x * self.scale_ratio) + self.offset_x,
                    int(y * self.scale_ratio) + self.offset_y,
                )
                for x, y in poly
            ]
            if len(disp_poly) >= 3:
                self.canvas.create_polygon(
                    disp_poly,
                    outline="red",
                    fill="",
                    width=2,
                )
            for pt in disp_poly:
                self.canvas.create_oval(
                    pt[0] - 3,
                    pt[1] - 3,
                    pt[0] + 3,
                    pt[1] + 3,
                    fill="red",
                    outline="red",
                )

        disp_pts = []
        for x, y in self.points:
            dx = int(x * self.scale_ratio) + self.offset_x
            dy = int(y * self.scale_ratio) + self.offset_y
            disp_pts.append((dx, dy))

        for pt in disp_pts:
            self.canvas.create_oval(
                pt[0] - 3,
                pt[1] - 3,
                pt[0] + 3,
                pt[1] + 3,
                fill="red",
                outline="red",
            )

        for i in range(len(disp_pts) - 1):
            self.canvas.create_line(disp_pts[i], disp_pts[i + 1], fill="red", width=2)

        if self.drawing and disp_pts:
            self.canvas.create_line(
                disp_pts[-1],
                self.last_cursor_pos,
                fill="red",
                dash=(4, 2),
                width=1,
            )

    def on_mouse_click(self, event):
        if self.frame is None or not self.drawing:
            return

        x_in_img = (event.x - self.offset_x) / self.scale_ratio
        y_in_img = (event.y - self.offset_y) / self.scale_ratio

        x_in_img = max(0, min(x_in_img, self.orig_width - 1))
        y_in_img = max(0, min(y_in_img, self.orig_height - 1))

        self.points.append((int(x_in_img), int(y_in_img)))
        self.display_frame()

    def on_right_click(self, event):
        if self.frame is None or not self.drawing:
            return

        self.drawing = False
        point_count = len(self.points)
        if point_count >= 3:
            self.polygons.append(list(self.points))
            self.points = []
            self._set_status(f"Drawing completed. {point_count} points. Ready to save.", "green")
            messagebox.showinfo(
                "Drawing Complete",
                f"Polygon with {point_count} points completed.\nYou can now save or exit.",
            )
        elif point_count >= 1:
            self._set_status(
                f"Drawing ended with {point_count} points. At least 3 points are required.",
                "orange",
            )
            messagebox.showwarning(
                "Incomplete Polygon",
                f"You have {point_count} points.\nAt least 3 points are required.",
            )
        else:
            self._set_status("Drawing cancelled.", "blue")

        self.display_frame()

    def on_mouse_move(self, event):
        if self.frame is None or not self.drawing or len(self.points) == 0:
            return
        self.last_cursor_pos = (event.x, event.y)
        self.display_frame()

    def start_drawing(self):
        if self.frame is None:
            messagebox.showwarning("Warning", "Please load a video first.")
            return
        if not self.drawing:
            self.points = []
        self.drawing = True
        self._set_status(
            "Drawing mode: left-click to add points, right-click to finish.",
            "darkgreen",
        )

    def _zone_config_key(self) -> str:
        return self.zone_key or os.path.basename(self.video_path)

    def _load_zone_config(self):
        if not os.path.exists(CONFIG_FILE):
            return {}
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_zone_config(self, config):
        os.makedirs(os.path.dirname(CONFIG_FILE) or ".", exist_ok=True)
        tmp_path = f"{CONFIG_FILE}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, CONFIG_FILE)

    def _save_current_zone(self):
        video_name = self._zone_config_key()
        config = self._load_zone_config()
        polygons = [list(poly) for poly in self.polygons]
        if len(self.points) >= 3:
            polygons.append(list(self.points))
        config[video_name] = [
            [list(point) for point in poly]
            for poly in polygons
        ]
        self._write_zone_config(config)
        return video_name

    def save_zone(self):
        if not self.polygons and len(self.points) < 3:
            messagebox.showwarning("Warning", "At least 3 points are required to form a zone.")
            return

        try:
            polygon_count = len(self.polygons) + (1 if len(self.points) >= 3 else 0)
            video_name = self._save_current_zone()
            self._set_status(f"Saved zone for {video_name} (still editing)", "green")
            messagebox.showinfo("Success", f"Saved {polygon_count} zone(s) for {video_name}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save config: {e}")

    def finish_and_exit(self):
        if not self.polygons and len(self.points) < 3:
            messagebox.showerror(
                "Error",
                "Must draw at least 3 points to define a zone.\nPlease complete the polygon.",
            )
            return

        try:
            video_name = self._save_current_zone()
            messagebox.showinfo(
                "Success",
                f"Zone saved for {video_name}.\nClosing GUI to resume processing...",
            )
            self.root.quit()
            sys.exit(0)
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save config: {e}\nCannot exit.")

    def clear_points(self):
        self.polygons = []
        self.points = []
        self.drawing = False
        if self.frame is not None:
            self.display_frame()
        self._set_status("Points cleared.", "blue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw No-Parking Zone")
    parser.add_argument("--source", type=str, help="Local video path or RTSP URL to auto-load")
    parser.add_argument("--video", type=str, help="Backward-compatible alias of --source")
    parser.add_argument("--zone-key", type=str, help="Config key used when saving the zone")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode behavior")
    args = parser.parse_args()

    auto_source = args.source or args.video

    root = tk.Tk()
    app = FenceDrawingApp(
        root,
        auto_load_source=auto_source,
        zone_key=args.zone_key,
    )

    if args.test_mode or auto_source:
        root.attributes("-topmost", True)
        root.update()
        root.attributes("-topmost", False)

    root.mainloop()
