# scripts/draw_fence_gui.py (完整修改版)
"""
Interactive GUI Tool to Draw No-Parking Zones on Video Frames.
TEST MODE: Supports auto-load video via CLI arg and "Finish & Exit" workflow.
NEW FEATURE: Right-click to end drawing mode.
"""

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import sys
import argparse
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageTk

CONFIG_FILE = "no_parking_config.json"
UPLOAD_FOLDER = "./uploads"


class FenceDrawingApp:
    def __init__(self, root: tk.Tk, auto_load_video: str = None):
        self.root = root
        self.root.title("🎨 Draw No-Parking Zone (TEST MODE)")

        # Canvas size (max display area)
        self.canvas_width = 800
        self.canvas_height = 600

        # Image and drawing state
        self.video_path = None
        self.frame = None  # Original resolution frame (RGB)
        self.display_photo = None
        self.points: List[Tuple[int, int]] = []  # Points in ORIGINAL image coordinates
        self.drawing = False
        self.auto_loaded = False  # Track if video was auto-loaded

        # Scale factor from original image to displayed image
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_cursor_pos = (0, 0)

        # UI Elements
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="gray")
        self.canvas.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="📁 Select Video", command=self.load_video).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="✏️ Start Drawing", command=self.start_drawing).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="✅ Save Zone", command=self.save_zone).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="✅ Finish & Exit", command=self.finish_and_exit, bg="#4CAF50", fg="white").grid(row=0,
                                                                                                                  column=3,
                                                                                                                  padx=5)  # NEW
        tk.Button(btn_frame, text="🗑️ Clear", command=self.clear_points).grid(row=0, column=4, padx=5)

        self.status_label = tk.Label(root, text="Please select a video.", fg="blue")
        self.status_label.pack(pady=5)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_mouse_click)  # Left click
        self.canvas.bind("<Button-3>", self.on_right_click)  # Right click (Windows/Linux)
        self.canvas.bind("<Button-2>", self.on_right_click)  # Right click (macOS)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # Auto-load video if provided (TEST MODE)
        if auto_load_video and os.path.exists(auto_load_video):
            self._auto_load_video(auto_load_video)

    def _auto_load_video(self, path: str):
        """Auto-load video in test mode"""
        self.video_path = path
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", f"Cannot read video: {path}")
            self.status_label.config(text="❌ Failed to load auto-specified video", fg="red")
            return

        # Convert BGR (OpenCV) to RGB (PIL/Tkinter)
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.orig_height, self.orig_width = self.frame.shape[:2]

        # Compute scale to fit canvas while preserving aspect ratio
        scale_w = self.canvas_width / self.orig_width
        scale_h = self.canvas_height / self.orig_height
        self.scale_ratio = min(scale_w, scale_h)

        self.display_width = int(self.orig_width * self.scale_ratio)
        self.display_height = int(self.orig_height * self.scale_ratio)

        # Centering offsets
        self.offset_x = (self.canvas_width - self.display_width) // 2
        self.offset_y = (self.canvas_height - self.display_height) // 2

        video_name = os.path.basename(path)
        self.status_label.config(
            text=f"🎯 AUTO-LOADED: {video_name} | Draw zone → Click 'Finish & Exit'",
            fg="purple",
            font=("Arial", 10, "bold")
        )
        self.auto_loaded = True
        self.display_frame()
        self.start_drawing()  # Auto-enable drawing mode

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
        self.orig_height, self.orig_width = self.frame.shape[:2]

        # Compute scale to fit canvas while preserving aspect ratio
        scale_w = self.canvas_width / self.orig_width
        scale_h = self.canvas_height / self.orig_height
        self.scale_ratio = min(scale_w, scale_h)

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

        img = Image.fromarray(self.frame).resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.display_photo)
        self.draw_polygon_overlay()

    def draw_polygon_overlay(self):
        """Draw current polygon on top of displayed image."""
        if len(self.points) < 1:
            return

        disp_pts = []
        for x, y in self.points:
            dx = int(x * self.scale_ratio) + self.offset_x
            dy = int(y * self.scale_ratio) + self.offset_y
            disp_pts.append((dx, dy))

        # Draw all points as small circles
        for pt in disp_pts:
            self.canvas.create_oval(pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3, fill="red", outline="red")

        # Draw lines between points
        for i in range(len(disp_pts) - 1):
            self.canvas.create_line(disp_pts[i], disp_pts[i + 1], fill="red", width=2)

        # Draw preview line from last point to cursor (if drawing)
        if self.drawing and len(disp_pts) > 0:
            self.canvas.create_line(disp_pts[-1], self.last_cursor_pos, fill="red", dash=(4, 2), width=1)

    def on_mouse_click(self, event):
        """Add a point in original image coordinates (left click)."""
        if self.frame is None or not self.drawing:
            return

        # Convert canvas coordinate to original image coordinate
        x_in_img = (event.x - self.offset_x) / self.scale_ratio
        y_in_img = (event.y - self.offset_y) / self.scale_ratio

        # Clip to image bounds
        x_in_img = max(0, min(x_in_img, self.orig_width - 1))
        y_in_img = max(0, min(y_in_img, self.orig_height - 1))

        self.points.append((int(x_in_img), int(y_in_img)))
        self.display_frame()  # Re-draw with new point

    def on_right_click(self, event):
        """End drawing mode on right click."""
        if self.frame is None or not self.drawing:
            return

        # End drawing mode
        self.drawing = False

        # Update status message
        point_count = len(self.points)
        if point_count >= 3:
            self.status_label.config(
                text=f"✅ Drawing completed! {point_count} points. Ready to save or exit.",
                fg="green"
            )
            messagebox.showinfo("Drawing Complete",
                                f"Polygon with {point_count} points completed!\nYou can now save or exit.")
        elif point_count >= 1:
            self.status_label.config(
                text=f"⚠️ Drawing ended with {point_count} points (need ≥3 to save).",
                fg="orange"
            )
            messagebox.showwarning("Incomplete Polygon",
                                   f"You have {point_count} points.\nAt least 3 points are required to form a valid zone.")
        else:
            self.status_label.config(text="Drawing cancelled.", fg="blue")

        self.display_frame()  # Remove preview line

    def on_mouse_move(self, event):
        """Update preview line during drawing."""
        if self.frame is None or not self.drawing or len(self.points) == 0:
            return
        self.last_cursor_pos = (event.x, event.y)
        self.display_frame()  # This will draw preview line

    def start_drawing(self):
        if self.frame is None:
            messagebox.showwarning("Warning", "Please load a video first.")
            return
        self.drawing = True
        self.status_label.config(text="✏️ Drawing mode: Click to add points. Right-click to finish drawing.",
                                 fg="darkgreen")

    def save_zone(self):
        """Save zone but stay in app (for manual use)"""
        if len(self.points) < 3:
            messagebox.showwarning("Warning", "At least 3 points are required to form a zone.")
            return

        video_name = os.path.basename(self.video_path)
        config = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except:
                config = {}

        config[video_name] = [[list(point) for point in self.points]]

        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            self.status_label.config(text=f"✅ Saved zone for {video_name} (still editing)", fg="green")
            messagebox.showinfo("Success", f"Saved {len(self.points)}-point zone for {video_name}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save config: {e}")

    def finish_and_exit(self):
        """TEST MODE: Save if valid, then exit app with success code"""
        if len(self.points) < 3:
            messagebox.showerror("Error",
                                 "❌ Must draw at least 3 points to define a zone!\nPlease complete the polygon.")
            return

        # Save zone
        video_name = os.path.basename(self.video_path)
        config = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except:
                config = {}

        config[video_name] = [[list(point) for point in self.points]]

        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("✅ Success", f"Zone saved for {video_name}!\nClosing GUI to resume processing...")
            self.root.quit()  # Exit cleanly
            sys.exit(0)  # Success exit code
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save config: {e}\nCannot exit.")
            return

    def clear_points(self):
        self.points = []
        self.drawing = False
        if self.frame is not None:
            self.display_frame()
        self.status_label.config(text="Points cleared.", fg="blue")


if __name__ == "__main__":
    # ===== TEST MODE: Support auto-load via CLI arg =====
    parser = argparse.ArgumentParser(description="Draw No-Parking Zone (Test Mode)")
    parser.add_argument("--video", type=str, help="Path to video file to auto-load")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode behavior")
    args = parser.parse_args()

    root = tk.Tk()
    app = FenceDrawingApp(root, auto_load_video=args.video if args.video else None)

    # Set window to stay on top in test mode (better UX)
    if args.test_mode or args.video:
        root.attributes('-topmost', True)
        root.update()
        root.attributes('-topmost', False)

    root.mainloop()