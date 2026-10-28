"""分身小窗：把另一个桌面上的游戏画面实时显示在主桌面的一个窗口里。

* 画面：PrintWindow 抓窗口（非活跃桌面唯一可行的后端），缩放到窗口大小；
* 鼠标：窗口里的点击/拖拽翻译成 PostMessage 发给游戏窗口，真实鼠标不动。
"""

from __future__ import annotations

import threading
import time
import tkinter as tk

import numpy as np

from . import pngio, win32

BG = "#16171B"
CARD = "#1E1F24"
TEXT = "#E7E9EE"
MUTED = "#9AA3B2"
ACCENT = "#2F7CF6"


class ShadowPreview(tk.Toplevel):
    """Live view of a window that may live on another desktop."""

    def __init__(self, master, hwnd: int, title: str = "", fps: float = 12.0) -> None:
        super().__init__(master)
        self.hwnd = int(hwnd)
        self.title(f"分身小窗 — {title[:32] or hex(self.hwnd)}")
        self.configure(bg="#000000")
        self.geometry("960x600")
        self.minsize(360, 240)
        self._fps = max(1.0, min(30.0, fps))
        self._stop = threading.Event()
        self._frame = None
        self._photo = None
        self._lock = threading.Lock()
        self._client = (1920, 1080)

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x")
        self._status = tk.Label(bar, text="连接中…", bg=BG, fg=MUTED,
                                font=("Microsoft YaHei UI", 8))
        self._status.pack(side="left", padx=10, pady=4)
        tk.Button(bar, text="关闭", command=self.close, bg="#2A2C33", fg=TEXT,
                  bd=0, relief="flat", padx=10, font=("Microsoft YaHei UI", 8),
                  cursor="hand2").pack(side="right", padx=8, pady=4)

        self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=0, cursor="none")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: None)
        self.canvas.bind("<ButtonPress-3>", lambda e: self._send_click(e, right=True))
        self.bind("<Escape>", lambda _e: self.close())
        self.focus_force()

        win32.set_shadow_target(self.hwnd)
        self.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self._loop, daemon=True).start()
        self.after(30, self._draw_loop)

    # -- capture -----------------------------------------------------------
    def _loop(self) -> None:
        period = 1.0 / self._fps
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                frame = win32.capture_window(self.hwnd, method="window")
            except Exception:
                frame = None
            if frame is not None and getattr(frame, "size", 0):
                with self._lock:
                    self._frame = frame
            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, period - elapsed))

    def _draw_loop(self) -> None:
        if self._stop.is_set():
            return
        with self._lock:
            frame = self._frame
            self._frame = None
        if frame is not None:
            self._present(frame)
            self._status.configure(
                text=f"{frame.shape[1]}×{frame.shape[0]} · 后端 PrintWindow · 鼠标已隔离")
        self.after(max(20, int(1000 / self._fps)), self._draw_loop)

    def _present(self, frame: np.ndarray) -> None:
        width = max(2, self.canvas.winfo_width())
        height = max(2, self.canvas.winfo_height())
        scale = min(width / frame.shape[1], height / frame.shape[0])
        out_w = max(1, int(frame.shape[1] * scale))
        out_h = max(1, int(frame.shape[0] * scale))
        ys = np.clip((np.arange(out_h) * frame.shape[0] // out_h), 0, frame.shape[0] - 1)
        xs = np.clip((np.arange(out_w) * frame.shape[1] // out_w), 0, frame.shape[1] - 1)
        small = frame[ys][:, xs]
        self._photo = pngio.to_photoimage(small, master=self)
        self.canvas.delete("frame")
        self.canvas.create_image(width // 2, height // 2, image=self._photo, tags="frame")

    # -- input translation -------------------------------------------------
    def _refresh_client(self) -> None:
        try:
            _x, _y, width, height = win32.client_rect(self.hwnd)
            self._client = (max(1, width), max(1, height))
        except Exception:
            pass

    def _to_client(self, event) -> tuple[int, int]:
        width = max(2, self.canvas.winfo_width())
        height = max(2, self.canvas.winfo_height())
        cw, ch = self._client
        scale = min(width / cw, height / ch)
        draw_w, draw_h = cw * scale, ch * scale
        raw_x = int((event.x - (width - draw_w) / 2.0) / max(1e-6, scale))
        raw_y = int((event.y - (height - draw_h) / 2.0) / max(1e-6, scale))
        return max(0, min(cw - 1, raw_x)), max(0, min(ch - 1, raw_y))

    def _send_click(self, event, right: bool = False) -> None:
        self._refresh_client()
        x, y = self._to_client(event)
        win32.set_shadow_target(self.hwnd)
        (win32.click_right if right else win32.click)(x, y)

    def _on_press(self, event) -> None:
        self._send_click(event)

    def _on_drag(self, event) -> None:
        self._refresh_client()
        x, y = self._to_client(event)
        win32.set_shadow_target(self.hwnd)
        win32.move_to(x, y)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._stop.set()
        self.destroy()

    def destroy(self) -> None:
        try:
            if win32.shadow_target() == self.hwnd:
                win32.set_shadow_target(0)
        except Exception:
            pass
        super().destroy()
