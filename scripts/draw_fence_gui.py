# scripts/draw_fence_gui.py

"""
Interactive GUI Tool to Draw No-Parking Zones on Video Frames.
Fixes:
  - Maintains aspect ratio when displaying video frame
  - Correctly maps mouse coordinates from canvas to original image space
  - Clips points to image bounds to prevent invalid coordinates
"""

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageTk


CONFIG_FILE = "no_parking_config.json"
UPLOAD_FOLDER = "./uploads"


class FenceDrawingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎨 Draw No-Parking Zone")

        # Canvas size (max display area)
        self.canvas_width = 800
        self.canvas_height = 600

        # Image and drawing state
        self.video_path = None
        self.frame = None  # Original resolution frame (RGB)
        self.display_photo = None
        self.points: List[Tuple[int, int]] = []  # Points in ORIGINAL image coordinates
        self.drawing = False

        # Scale factor from original image to displayed image
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # initial location of mouse
        self.last_cursor_pos = (0, 0)

        # UI Elements
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="gray")
        self.canvas.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="📁 Select Video", command=self.load_video).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="✏️ Start Drawing", command=self.start_drawing).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="✅ Save Zone", command=self.save_zone).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="🗑️ Clear", command=self.clear_points).grid(row=0, column=3, padx=5)

        self.status_label = tk.Label(root, text="Please select a video.", fg="blue")
        self.status_label.pack(pady=5)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)

    def load_video(self):
        """Open file dialog to select video from uploads folder."""
        initial_dir = UPLOAD_FOLDER if os.path.exists(UPLOAD_FOLDER) else "."
        path = filedialog.askopenfilename(
            title="Select Video for Fence Drawing",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
            initialdir=initial_dir
        )
        if not path:
            return

        self.video_path = path
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", "Cannot read video.")
            return

        # Convert BGR (OpenCV) to RGB (PIL/Tkinter)
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.orig_height, self.orig_width = self.frame.shape[:2]  # ⚠️ 注意顺序：H, W

        # Compute scale to fit canvas while preserving aspect ratio
        scale_w = self.canvas_width / self.orig_width
        scale_h = self.canvas_height / self.orig_height
        self.scale_ratio = min(scale_w, scale_h)  # Uniform scale

        self.display_width = int(self.orig_width * self.scale_ratio)
        self.display_height = int(self.orig_height * self.scale_ratio)

        # Centering offsets
        self.offset_x = (self.canvas_width - self.display_width) // 2
        self.offset_y = (self.canvas_height - self.display_height) // 2

        self.status_label.config(text=f"Loaded: {os.path.basename(path)} | Size: {self.orig_width}x{self.orig_height}")
        self.display_frame()

    def display_frame(self):
        """Display the current frame centered in canvas with correct aspect ratio."""
        if self.frame is None:
            return

        # Resize image for display
        img = Image.fromarray(self.frame).resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)

        # Create PhotoImage
        self.display_photo = ImageTk.PhotoImage(image=img)

        # Clear canvas and draw image at center
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.display_photo)

        # Redraw polygon overlay
        self.draw_polygon_overlay()

    def draw_polygon_overlay(self):
        """Draw current polygon on top of displayed image."""
        if len(self.points) < 1:
            return

        # Transform points to display coordinates
        disp_pts = []
        for x, y in self.points:
            dx = int(x * self.scale_ratio) + self.offset_x
            dy = int(y * self.scale_ratio) + self.offset_y
            disp_pts.append((dx, dy))

        # Draw lines
        for i in range(len(disp_pts) - 1):
            self.canvas.create_line(disp_pts[i], disp_pts[i+1], fill="red", width=2)

        # Draw preview line from last point to cursor (if drawing)
        if self.drawing and len(disp_pts) > 0:
            self.canvas.create_line(disp_pts[-1], self.last_cursor_pos, fill="red", dash=(4, 2), width=1)

    def on_mouse_click(self, event):
        """Add a point in original image coordinates."""
        if self.frame is None or not self.drawing:
            return

        # Convert canvas coordinate to original image coordinate
        x_in_img = (event.x - self.offset_x) / self.scale_ratio
        y_in_img = (event.y - self.offset_y) / self.scale_ratio

        # ✅ 修复：强制裁剪到图像边界内（防止保存越界坐标）
        x_in_img = max(0, min(x_in_img, self.orig_width - 1))
        y_in_img = max(0, min(y_in_img, self.orig_height - 1))

        self.points.append((int(x_in_img), int(y_in_img)))
        self.display_frame()  # Re-draw with new point

    def on_mouse_move(self, event):
        """Update preview line during drawing."""
        if self.frame is None or not self.drawing or len(self.points) == 0:
            return

        # Store cursor position in display space for preview
        self.last_cursor_pos = (event.x, event.y)
        self.display_frame()  # This will draw preview line

    def start_drawing(self):
        if self.frame is None:
            messagebox.showwarning("Warning", "Please load a video first.")
            return
        self.drawing = True
        self.status_label.config(text="✏️ Drawing mode: Click to add points. Press 'Save' when done.")

    def save_zone(self):
        if len(self.points) < 3:
            messagebox.showwarning("Warning", "At least 3 points are required to form a zone.")
            return

        video_name = os.path.basename(self.video_path)

        # Load existing config
        config = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                messagebox.showwarning("Warning", "Invalid JSON. Overwriting config.")
                config = {}

        # Save new zone (convert tuples to lists for JSON serialization)
        config[video_name] = [list(point) for point in self.points]

        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Success", f"Saved {len(self.points)}-point zone for {video_name}")
            self.status_label.config(text=f"✅ Saved zone: {video_name}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save config: {e}")

    def clear_points(self):
        self.points = []
        self.drawing = False
        if self.frame is not None:
            self.display_frame()
        self.status_label.config(text="Points cleared.")


if __name__ == "__main__":
    root = tk.Tk()
    app = FenceDrawingApp(root)
    root.mainloop()
