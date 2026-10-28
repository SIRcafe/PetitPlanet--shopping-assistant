"""格连商店助手界面：寻路 / 进店截图 / 桌面分身 / 一键停止。"""

from __future__ import annotations

import json
import pathlib
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from . import gamepath, spec, virtual_desktop, walker, win32

SETTINGS = spec.ROOT / "settings.json"

BG = "#16171B"
CARD = "#1E1F24"
LINE = "#2A2C33"
TEXT = "#E7E9EE"
MUTED = "#9AA3B2"
ACCENT = "#2F7CF6"
INPUT = "#24262C"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("星布谷地 · 格连商店助手")
        self.configure(bg=BG)
        self.geometry("860x880")
        self.minsize(760, 720)

        self.settings = self._load_settings()
        self.manual_hwnd = int(self.settings.get("manual_hwnd") or 0)
        self.shadow: virtual_desktop.VirtualDesktop | None = None
        self.preview = None
        self.worker: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.log_queue: queue.Queue = queue.Queue()

        self._build()
        self.after(100, self._drain_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- widgets -----------------------------------------------------------
    def _build(self) -> None:
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(head, text="格连商店助手", bg=BG, fg=TEXT,
                 font=("Microsoft YaHei UI", 16, "bold")).pack(side="left")
        self.state = tk.Label(head, text="就绪", bg=BG, fg=MUTED,
                              font=("Microsoft YaHei UI", 9))
        self.state.pack(side="right")

        # shadow -----------------------------------------------------------
        card = self._card()
        row = self._row(card, "桌面分身", "游戏跑到 Windows 第二个桌面上（需要管理员）")
        self.shadow_state = tk.Label(row, text="未创建", bg=CARD, fg=MUTED,
                                     font=("Microsoft YaHei UI", 8))
        self.shadow_state.pack(side="right", padx=(0, 8))
        self._button(row, "创建分身并启动游戏", self.create_shadow, "primary").pack(side="right")

        row2 = self._row(card, "分身操作", "小窗里可以直接点游戏，鼠标不会跑到外面", divider=True)
        self._button(row2, "结束分身", self.stop_shadow).pack(side="right")
        self._button(row2, "打开分身小窗", self.open_preview, "primary").pack(side="right", padx=8)

        # pipeline ---------------------------------------------------------
        card2 = self._card()
        row3 = self._row(card2, "小窗里跑完寻路 + 商店截图",
                         "点一次：走到门口 → 进店 → 连拍 → 拼图（需要先创建分身）")
        self._button(row3, "小窗：寻路+截图", lambda: self.start("both"), "primary").pack(side="right")

        row4 = self._row(card2, "分开执行", "只走路，或只进店拍照", divider=True)
        self._button(row4, "启用截图", lambda: self.start("shop")).pack(side="right")
        self._button(row4, "启用寻路", lambda: self.start("nav")).pack(side="right", padx=8)

        row5 = self._row(card2, "一键停止", "停下寻路与截图（分身和游戏保持运行）", divider=True)
        self._button(row5, "一键停止", self.stop_all).pack(side="right")

        # settings ---------------------------------------------------------
        card3 = self._card()
        row6 = self._row(card3, "输出标签", "截图存到 reports/runs/<标签>/")
        self.label = self._entry(row6, str(self.settings.get("label", "shop")), 14)
        row7 = self._row(card3, "截图张数 / 每张转角（像素）",
                         "默认 6 张；1500 像素约 60–70 度", divider=True)
        self.turn = self._entry(row7, str(self.settings.get("turn", 1500)), 8)
        self.shots = self._entry(row7, str(self.settings.get("shots", 6)), 6)
        row8 = self._row(card3, "寻路参数", "时间预算（秒）/ 最大步数", divider=True)
        self.steps = self._entry(row8, str(self.settings.get("max_steps", 90)), 6)
        self.budget = self._entry(row8, str(self.settings.get("budget_s", 600)), 8)

        row9 = self._row(card3, "游戏窗口", "只认 PetitPlanet.exe；找不到就手动选", divider=True)
        self.window_label = tk.Label(row9, text=self._window_text(), bg=CARD, fg=MUTED,
                                     font=("Microsoft YaHei UI", 8))
        self.window_label.pack(side="right", padx=(0, 8))
        self._button(row9, "选择窗口", self.pick_window).pack(side="right")

        # actions ----------------------------------------------------------
        actions = tk.Frame(self, bg=BG)
        actions.pack(fill="x", padx=16, pady=(4, 2))
        self.start_button = self._button(actions, "开始", self.start_clicked, "primary")
        self.start_button.pack(side="left")
        self.stop_button = self._button(actions, "停止", self.stop_all)
        self.stop_button.pack(side="left", padx=8)
        self.stop_button.configure(state="disabled")
        self._button(actions, "打开输出目录",
                     lambda: self._open(spec.ROOT / "reports")).pack(side="right")

        # log --------------------------------------------------------------
        log_frame = tk.Frame(self, bg=CARD)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(6, 14))
        self.log = tk.Text(log_frame, bg="#101114", fg="#C9D3E4", bd=0, relief="flat",
                           insertbackground=TEXT, font=("Consolas", 9), wrap="none")
        self.log.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        bar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        bar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=bar.set, state="disabled")

    def _card(self) -> tk.Frame:
        card = tk.Frame(self, bg=CARD)
        card.pack(fill="x", padx=16, pady=(6, 0))
        return card

    def _row(self, parent, title: str, subtitle: str = "", divider: bool = False) -> tk.Frame:
        if divider:
            tk.Frame(parent, bg=LINE, height=1).pack(fill="x")
        holder = tk.Frame(parent, bg=CARD)
        holder.pack(fill="x")
        left = tk.Frame(holder, bg=CARD)
        left.pack(side="left", fill="x", expand=True, padx=(14, 6), pady=9)
        tk.Label(left, text=title, bg=CARD, fg=TEXT, anchor="w",
                 font=("Microsoft YaHei UI", 10)).pack(fill="x")
        if subtitle:
            tk.Label(left, text=subtitle, bg=CARD, fg=MUTED, anchor="w",
                     font=("Microsoft YaHei UI", 8)).pack(fill="x", pady=(2, 0))
        control = tk.Frame(holder, bg=CARD)
        control.pack(side="right", padx=(0, 14), pady=9)
        return control

    def _entry(self, parent, value: str, width: int) -> tk.Entry:
        entry = tk.Entry(parent, width=width, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                         insertbackground=TEXT, font=("Microsoft YaHei UI", 9),
                         justify="center")
        entry.insert(0, value)
        entry.pack(side="right", ipady=3, padx=(6, 0))
        return entry

    def _button(self, parent, text, command, kind: str = "ghost") -> tk.Button:
        return tk.Button(parent, text=text, command=command,
                         bg=ACCENT if kind == "primary" else "#2A2C33",
                         fg="#FFFFFF" if kind == "primary" else TEXT,
                         activebackground="#1F5FD0" if kind == "primary" else "#33363E",
                         activeforeground="#FFFFFF", bd=0, relief="flat",
                         padx=14, pady=4, cursor="hand2",
                         font=("Microsoft YaHei UI", 9))

    # -- window helpers ----------------------------------------------------
    def _window_text(self) -> str:
        if self.manual_hwnd:
            return f"手动：{win32.window_text(self.manual_hwnd)[:24]} (#{self.manual_hwnd})"
        found = walker.find_window()
        if found:
            return f"自动：{str(found.get('title', ''))[:24]} (#{found['hwnd']})"
        return "未找到（先启动游戏，或点「选择窗口」）"

    def pick_window(self) -> None:
        usable = [w for w in win32.list_windows()
                  if int(w.get("w", 0)) > 400 and int(w.get("h", 0)) > 300]
        if not usable:
            messagebox.showinfo("选择窗口", "没有可用的窗口")
            return
        dialog = tk.Toplevel(self)
        dialog.title("选择游戏窗口")
        dialog.configure(bg=BG)
        dialog.geometry("640x420")
        dialog.transient(self)
        listbox = tk.Listbox(dialog, bg=CARD, fg=TEXT, bd=0, relief="flat",
                             highlightthickness=0, selectbackground=ACCENT,
                             selectforeground="#FFFFFF", font=("Microsoft YaHei UI", 9))
        listbox.pack(fill="both", expand=True, padx=12, pady=12)
        for window in usable:
            mark = "●" if window.get("fg") else "○"
            listbox.insert("end", f" {mark} {str(window.get('title', ''))[:50]}  "
                                  f"{window['w']}×{window['h']}  {window.get('exe', '')}")
        listbox.selection_set(0)

        def accept() -> None:
            selection = listbox.curselection()
            if selection:
                self.set_manual_window(int(usable[int(selection[0])]["hwnd"]))
            dialog.destroy()

        self._button(dialog, "使用这个窗口", accept, "primary").pack(side="right", padx=12, pady=(0, 12))
        dialog.grab_set()

    def set_manual_window(self, hwnd: int) -> None:
        self.manual_hwnd = int(hwnd)
        self._save_settings()
        self.window_label.configure(text=self._window_text())

    # -- shadow desktop ----------------------------------------------------
    def create_shadow(self) -> None:
        if self.shadow is not None:
            self._log("分身已经创建；先点「结束分身」再重建")
            return
        if not self._is_admin():
            self._log("创建分身需要管理员：请右键「以管理员身份运行」")
            return
        executable = str(self.settings.get("game_exe", "")) or gamepath.find_game_install(timeout_s=8)
        if not executable:
            self._log("没找到 PetitPlanet.exe：先点「选择窗口」旁边的自动检测，或手动浏览")
            return
        self.settings["game_exe"] = executable
        shadow = virtual_desktop.VirtualDesktop("PNAV_格连")
        if not shadow.create():
            self._log(f"创建桌面失败：{shadow.error}")
            return
        pid = shadow.start_process(executable, cwd=str(pathlib.Path(executable).parent),
                                   name="PetitPlanet")
        if not pid:
            self._log(f"在分身上启动游戏失败：{shadow.error}")
            shadow.close()
            return
        self.shadow = shadow
        self.shadow_state.configure(text="启动中…")
        self._log(f"分身已创建（pid {pid}），等待游戏窗口…")
        threading.Thread(target=self._wait_shadow_window, daemon=True).start()

    def _wait_shadow_window(self) -> None:
        window = self.shadow.wait_for_window(timeout=180) if self.shadow else None
        if not window:
            self.after(0, lambda: self.shadow_state.configure(text="未等到窗口"))
            self._log("180 秒内没等到游戏窗口（可能要登录/更新）")
            return
        self.after(0, lambda: self._shadow_ready(window))

    def _shadow_ready(self, window: dict) -> None:
        self.set_manual_window(int(window["hwnd"]))
        self.shadow_state.configure(text="运行中")
        self._log(f"分身窗口就绪：{window['title']} {window['w']}×{window['h']}")
        self.open_preview()

    def open_preview(self) -> None:
        from . import preview
        hwnd = self.manual_hwnd or (walker.find_window() or {}).get("hwnd", 0)
        if not hwnd:
            self._log("还没有可显示的窗口：先创建分身，或选择窗口")
            return
        if self.preview is not None:
            try:
                if self.preview.winfo_exists():
                    self.preview.lift()
                    return
            except Exception:
                pass
        self.preview = preview.ShadowPreview(self, int(hwnd), title=self.window_label.cget("text"))

    def stop_shadow(self) -> None:
        if self.preview is not None:
            try:
                self.preview.close()
            except Exception:
                pass
            self.preview = None
        if self.shadow is None:
            return
        self.shadow.close()
        self.shadow = None
        self.shadow_state.configure(text="未创建")
        self._log("分身已结束（游戏进程一并关闭）")

    @staticmethod
    def _is_admin() -> bool:
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

    # -- running -----------------------------------------------------------
    def start_clicked(self) -> None:
        self.start("both")

    def start(self, phase: str) -> None:
        if self.worker is not None and self.worker.is_alive():
            self._log("已经在跑了，先「一键停止」")
            return
        hwnd = self.manual_hwnd or (walker.find_window() or {}).get("hwnd", 0)
        if not hwnd:
            messagebox.showwarning("没找到游戏窗口",
                                   "请先启动星布谷地，或点「选择窗口」手动指定。")
            return
        try:
            options = walker.ShopOptions(
                label=self.label.get() or "shop",
                walk_budget_s=float(self.budget.get()),
                max_walk_steps=int(self.steps.get()),
                shots=int(self.shots.get()),
                turn=int(self.turn.get()),
                manual_hwnd=hwnd,
                log=self._log)
        except ValueError:
            messagebox.showwarning("参数不对", "预算/步数/张数/转角必须是数字")
            return
        self._save_settings()
        self.stop_event = threading.Event()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        names = {"nav": "寻路", "shop": "进店截图", "both": "寻路 + 截图"}
        self.state.configure(text=names.get(phase, phase), fg=ACCENT)
        self._log(f"--- 开始：{names.get(phase, phase)} ---")
        self.worker = threading.Thread(target=self._run, args=(phase, options), daemon=True)
        self.worker.start()

    def _run(self, phase: str, options) -> None:
        try:
            result = walker.run_phase(phase, options, self.stop_event)
            self._log(f"结果：{'成功' if result.get('ok') else '未完成'}"
                      f"{'' if result.get('ok') else ' — ' + str(result.get('reason', ''))}")
        except Exception as exc:
            self._log(f"出错：{exc}")
        finally:
            self.after(0, self._finish)

    def _finish(self) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.state.configure(text="就绪", fg=MUTED)

    def stop_all(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self._log("已请求停止（当前动作结束后退出）")

    def _open(self, folder: pathlib.Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["explorer", str(folder)])
        except Exception:
            pass

    # -- log / settings ----------------------------------------------------
    def _log(self, message: str) -> None:
        self.log_queue.put(str(message))

    def _drain_log(self) -> None:
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            self.log.configure(state="normal")
            self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {line}\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.after(100, self._drain_log)

    def _load_settings(self) -> dict:
        try:
            if SETTINGS.exists():
                return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_settings(self) -> None:
        def number(widget, fallback: float) -> float:
            try:
                return float(widget.get())
            except Exception:
                return fallback
        data = dict(
            label=self.label.get() or "shop",
            shots=int(number(self.shots, 6)),
            turn=int(number(self.turn, 1500)),
            max_steps=int(number(self.steps, 90)),
            budget_s=int(number(self.budget, 600)),
            manual_hwnd=self.manual_hwnd,
            game_exe=self.settings.get("game_exe", ""),
        )
        self.settings.update(data)
        try:
            SETTINGS.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.shadow is not None:
            try:
                self.shadow.close()
            except Exception:
                pass
        self.destroy()


def main() -> int:
    win32.make_dpi_aware()
    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
