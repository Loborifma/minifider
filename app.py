import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from tkinterdnd2 import DND_FILES

import ffmpeg_worker as fw
import groq_client

WINDOW_TITLE = "minifider"
WINDOW_SIZE = "720x700"

COLOR_CARD = "#ffffff"
COLOR_BORDER = "#e5e7eb"
COLOR_ACCENT = "#2563eb"
COLOR_ACCENT_HOVER = "#1d4fd1"
COLOR_TEXT = "#111827"
COLOR_TEXT_MUTED = "#6b7280"
COLOR_STATUS_QUEUED = "#6b7280"
COLOR_STATUS_PROGRESS = "#2563eb"
COLOR_STATUS_DONE = "#1a7f37"
COLOR_STATUS_ERROR = "#cf222e"


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(640, 640)

        self.files: list[str] = []
        self.row_widgets: dict[str, dict] = {}
        self.ui_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.current_proc_path: str | None = None
        self.cancel_event = threading.Event()
        self.proc_holder: list = []
        self._local_model = None
        self._files_snapshot: list[str] = []
        self._done_count = 0
        self._batch_total = 0

        self._build_widgets()
        self._verify_ffmpeg()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queue)
        if not groq_client.has_any_keys():
            self.root.after(200, self.open_settings)

    # ---- layout helpers ----

    def _step_header(self, parent, number, title):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x")
        badge = ctk.CTkFrame(row, width=24, height=24, corner_radius=12, fg_color=COLOR_ACCENT)
        badge.pack(side="left")
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text=str(number), text_color="white", font=("Segoe UI", 11, "bold")).pack(expand=True)
        ctk.CTkLabel(
            row, text=title, text_color=COLOR_TEXT, font=("Segoe UI", 13, "bold"),
        ).pack(side="left", padx=(8, 0))

    def _connector(self, parent):
        wrapper = ctk.CTkFrame(parent, fg_color="transparent", height=18)
        wrapper.pack(fill="x")
        wrapper.pack_propagate(False)
        line = ctk.CTkFrame(wrapper, width=2, fg_color=COLOR_BORDER)
        line.place(x=12, y=0, relheight=1)

    def _build_widgets(self):
        content = ctk.CTkFrame(self.root, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=16, pady=16)

        # ---- Step 1: Добавьте файлы ----
        self._step_header(content, 1, "Добавьте файлы")

        card1 = ctk.CTkFrame(
            content, corner_radius=12, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER,
        )
        card1.pack(fill="x", pady=(6, 0))
        drop_row = ctk.CTkFrame(card1, fg_color="transparent")
        drop_row.pack(fill="x", padx=14, pady=14)
        ctk.CTkLabel(drop_row, text="↑", text_color=COLOR_ACCENT, font=("Segoe UI", 16)).pack(side="left")
        ctk.CTkLabel(
            drop_row, text="Перетащите видео сюда", text_color=COLOR_TEXT,
        ).pack(side="left", padx=(8, 0))
        select_btn = ctk.CTkButton(
            drop_row, text="Выбрать файлы", command=self._on_select_files,
            fg_color="#eef0f3", text_color=COLOR_TEXT, hover_color="#e2e5ea", corner_radius=8,
        )
        select_btn.pack(side="right")

        for target in (card1, drop_row):
            target.drop_target_register(DND_FILES)
            target.dnd_bind("<<Drop>>", self._on_drop)

        self._connector(content)

        # ---- Step 2: Настройте обработку ----
        self._step_header(content, 2, "Настройте обработку")
        card2 = ctk.CTkFrame(
            content, corner_radius=12, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER,
        )
        card2.pack(fill="x", pady=(6, 0))

        compress_row = ctk.CTkFrame(card2, fg_color="transparent")
        compress_row.pack(fill="x", padx=14, pady=(14, 6))
        ctk.CTkLabel(compress_row, text="Сжать видео", text_color=COLOR_TEXT).pack(side="left")
        self.compress_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(
            compress_row, text="", variable=self.compress_var, command=self._on_compress_toggle,
            progress_color=COLOR_ACCENT,
        ).pack(side="right")

        self.compress_options_frame = ctk.CTkFrame(card2, fg_color="transparent")
        opt_row1 = ctk.CTkFrame(self.compress_options_frame, fg_color="transparent")
        opt_row1.pack(fill="x", padx=14, pady=(0, 4))
        ctk.CTkLabel(opt_row1, text="Качество", text_color=COLOR_TEXT_MUTED).pack(side="left")
        self.preset_var = tk.StringVar(value=fw.DEFAULT_PRESET)
        self.preset_combo = ctk.CTkComboBox(
            opt_row1, variable=self.preset_var, values=list(fw.PRESETS.keys()), width=230, state="readonly",
        )
        self.preset_combo.pack(side="left", padx=(8, 0))

        level_row = ctk.CTkFrame(self.compress_options_frame, fg_color="transparent")
        level_row.pack(fill="x", padx=14, pady=(0, 14))
        self.level_audio_var = tk.BooleanVar(value=False)
        self.level_audio_check = ctk.CTkCheckBox(
            level_row, text="Выровнять громкость голоса", variable=self.level_audio_var,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
        )
        self.level_audio_check.pack(side="left")

        transcribe_row = ctk.CTkFrame(card2, fg_color="transparent")
        transcribe_row.pack(fill="x", padx=14, pady=(6, 6))
        ctk.CTkLabel(transcribe_row, text="Транскрибировать", text_color=COLOR_TEXT).pack(side="left")
        self.transcribe_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(
            transcribe_row, text="", variable=self.transcribe_var, command=self._on_transcribe_toggle,
            progress_color=COLOR_ACCENT,
        ).pack(side="right")
        self._transcribe_row = transcribe_row

        self.transcribe_options_frame = ctk.CTkFrame(card2, fg_color="transparent")
        fmt_row = ctk.CTkFrame(self.transcribe_options_frame, fg_color="transparent")
        fmt_row.pack(fill="x", padx=14, pady=(0, 4))
        ctk.CTkLabel(fmt_row, text="Формат", text_color=COLOR_TEXT_MUTED).pack(side="left")
        self.subtitle_format_var = tk.StringVar(value="SRT")
        ctk.CTkSegmentedButton(
            fmt_row, values=["SRT", "TXT"], variable=self.subtitle_format_var,
            selected_color=COLOR_ACCENT, selected_hover_color=COLOR_ACCENT_HOVER,
        ).pack(side="left", padx=(8, 0))

        settings_row = ctk.CTkFrame(self.transcribe_options_frame, fg_color="transparent")
        settings_row.pack(fill="x", padx=14, pady=(0, 14))
        self.settings_link = ctk.CTkLabel(
            settings_row, text="Настройки API распознавания…", text_color=COLOR_ACCENT,
            font=("Segoe UI", 12, "underline"), cursor="hand2",
        )
        self.settings_link.pack(side="left")
        self.settings_link.bind("<Button-1>", lambda e: self.open_settings())

        self._on_compress_toggle()
        self._on_transcribe_toggle()

        self._connector(content)

        # ---- Step 3: Запустите и следите ----
        self._step_header(content, 3, "Запустите и следите")

        self.status_label = ctk.CTkLabel(
            content, text="Готово к работе", text_color=COLOR_TEXT_MUTED, anchor="w",
        )
        self.status_label.pack(fill="x", pady=(6, 4))

        self.progress_bar = ctk.CTkProgressBar(content, progress_color=COLOR_ACCENT)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x")

        bottom = ctk.CTkFrame(content, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", pady=(10, 0))
        self.start_btn = ctk.CTkButton(
            bottom, text="Начать обработку", command=self._on_start, corner_radius=10,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
        )
        self.start_btn.pack(side="right")

        self.tabview = ctk.CTkTabview(content, fg_color=COLOR_CARD)
        self.tabview.pack(fill="both", expand=True, pady=(10, 0))
        files_tab = self.tabview.add("Файлы")
        log_tab = self.tabview.add("Журнал")

        self.files_list = ctk.CTkScrollableFrame(files_tab, fg_color="transparent")
        self.files_list.pack(fill="both", expand=True, padx=4, pady=4)

        self.log_widget = ctk.CTkTextbox(log_tab, fg_color="#ffffff", text_color="#1f2328", wrap="word")
        self.log_widget.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_widget.configure(state="disabled")

    def _on_compress_toggle(self):
        if self.compress_var.get():
            self.compress_options_frame.pack(fill="x", before=self._transcribe_row)
        else:
            self.compress_options_frame.pack_forget()

    def _on_transcribe_toggle(self):
        if self.transcribe_var.get():
            self.transcribe_options_frame.pack(fill="x")
        else:
            self.transcribe_options_frame.pack_forget()

    def _verify_ffmpeg(self):
        try:
            fw.get_ffmpeg_path()
        except FileNotFoundError as exc:
            messagebox.showerror("minifider", str(exc))
            self.start_btn.configure(state="disabled")

    def _set_settings_link_enabled(self, enabled):
        if enabled:
            self.settings_link.configure(text_color=COLOR_ACCENT, cursor="hand2")
            self.settings_link.bind("<Button-1>", lambda e: self.open_settings())
        else:
            self.settings_link.configure(text_color=COLOR_TEXT_MUTED, cursor="arrow")
            self.settings_link.unbind("<Button-1>")

    def open_settings(self):
        SettingsDialog(self.root, on_saved=lambda: self.log("Настройки Groq API сохранены."))

    def log(self, text):
        self.ui_queue.put(("log", text))

    def _append_log(self, text):
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _get_local_model(self):
        import transcribe_worker as tw

        if self._local_model is None:
            self.ui_queue.put(("log", "Загружаю локальную модель Whisper (fallback)..."))
            self._local_model = tw.load_model(log=self.log)
        return self._local_model

    def _on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        self._add_files(paths)

    def _on_select_files(self):
        paths = filedialog.askopenfilenames(title="Выберите видео")
        self._add_files(paths)

    def _add_files(self, paths):
        for path in paths:
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in fw.VIDEO_EXTENSIONS:
                continue
            if path in self.files:
                continue
            self.files.append(path)
            self._add_file_row(path)

    def _add_file_row(self, path):
        row = ctk.CTkFrame(self.files_list, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(
            row, text=os.path.basename(path), text_color=COLOR_TEXT, anchor="w",
        ).pack(side="left", fill="x", expand=True)
        status_label = ctk.CTkLabel(row, text="В очереди", text_color=COLOR_STATUS_QUEUED, anchor="e")
        status_label.pack(side="right")
        self.row_widgets[path] = {"row": row, "status": status_label}

    def _set_row_status(self, path, text, color):
        widgets = self.row_widgets.get(path)
        if widgets:
            widgets["status"].configure(text=text, text_color=color)

    def _clear_file_rows(self):
        for widgets in self.row_widgets.values():
            widgets["row"].destroy()
        self.row_widgets.clear()

    def _update_status_line(self):
        if self._batch_total and self._done_count < self._batch_total:
            current_path = self._files_snapshot[self._done_count]
            basename = os.path.basename(current_path)
            self.status_label.configure(text=f"Файл {self._done_count + 1} из {self._batch_total} · {basename}")
        else:
            self.status_label.configure(text="Готово к работе")

    def _on_start(self):
        if not self.files:
            return
        if self.worker_thread and self.worker_thread.is_alive():
            return
        compress = self.compress_var.get()
        transcribe = self.transcribe_var.get()
        if not compress and not transcribe:
            messagebox.showwarning("minifider", "Выбери хотя бы одно действие: сжатие или транскрибация.")
            return

        self.start_btn.configure(state="disabled")
        self._set_settings_link_enabled(False)
        self.progress_bar.set(0)
        preset = self.preset_var.get()
        level_audio = self.level_audio_var.get()
        subtitle_format = self.subtitle_format_var.get().lower()
        files_snapshot = list(self.files)
        self._files_snapshot = files_snapshot
        self._done_count = 0
        self._batch_total = len(files_snapshot)
        self._update_status_line()
        self.cancel_event.clear()
        self.worker_thread = threading.Thread(
            target=fw.run_batch,
            args=(files_snapshot, preset, self.ui_queue, self.cancel_event, self.proc_holder),
            kwargs={
                "level_audio": level_audio,
                "compress": compress,
                "transcribe": transcribe,
                "subtitle_format": subtitle_format,
                "get_local_model": self._get_local_model,
            },
            daemon=True,
        )
        self.worker_thread.start()

    def _poll_queue(self):
        try:
            while True:
                self._handle_msg(self.ui_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _handle_msg(self, msg):
        kind = msg[0]
        if kind == "progress":
            _, path, pct = msg
            self.current_proc_path = path
            self._set_row_status(path, f"{pct}%", COLOR_STATUS_PROGRESS)
            self.progress_bar.set(pct / 100)
        elif kind == "file_done":
            _, path, ok, info = msg
            if ok:
                self._set_row_status(path, "✓ Готово", COLOR_STATUS_DONE)
            else:
                self._set_row_status(path, f"⚠ Ошибка: {info}", COLOR_STATUS_ERROR)
        elif kind == "batch_progress":
            _, index, total = msg
            self._done_count = index
            self._batch_total = total
            self._update_status_line()
        elif kind == "log":
            self._append_log(msg[1])
        elif kind == "all_done":
            self.current_proc_path = None
            self.progress_bar.set(0)
            self._done_count = 0
            self._batch_total = 0
            self.status_label.configure(text="Готово к работе")
            self.start_btn.configure(state="normal")
            self._set_settings_link_enabled(True)
            self.files.clear()
            self._clear_file_rows()
            messagebox.showinfo("minifider", "Обработка завершена")

    def _on_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno("minifider", "Сжатие ещё выполняется. Прервать и закрыть?"):
                return
            self.cancel_event.set()
            for proc in list(self.proc_holder):
                try:
                    proc.terminate()
                except OSError:
                    pass
            self.worker_thread.join(timeout=3)
        self.root.destroy()


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, on_saved=None):
        super().__init__(parent)
        self.title("Настройки Groq API")
        self.transient(parent)
        self.on_saved = on_saved

        ctk.CTkLabel(
            self, justify="left", wraplength=420,
            text=(
                "Введи один или несколько Groq API-ключей (по одному на строку).\n"
                "Получить бесплатный ключ: https://console.groq.com/keys\n"
                "Приложение пробует ключи по очереди; если все исчерпали лимит,\n"
                "оно подождёт обновления лимита (если локальный Whisper ниже выключен)."
            ),
        ).pack(padx=12, pady=(12, 6), anchor="w")

        self.text = ctk.CTkTextbox(self, width=440, height=140)
        self.text.pack(padx=12, pady=4)
        existing_keys = [entry["key"] for entry in groq_client.load_config()["groq_keys"]]
        self.text.insert("1.0", "\n".join(existing_keys))

        self.local_fallback_var = tk.BooleanVar(value=groq_client.get_use_local_fallback())
        ctk.CTkCheckBox(
            self, variable=self.local_fallback_var,
            text="Использовать локальную модель Whisper как запасной вариант\n(скачивается ~3 ГБ при первом использовании)",
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
        ).pack(padx=12, pady=(6, 6), anchor="w")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(6, 12))
        ctk.CTkButton(
            btn_frame, text="Сохранить", command=self._save,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            btn_frame, text="Отмена", command=self.destroy,
            fg_color="#eef0f3", text_color=COLOR_TEXT, hover_color="#e2e5ea",
        ).pack(side="left", padx=4)

        self.grab_set()

    def _save(self):
        raw = self.text.get("1.0", "end")
        keys = [line.strip() for line in raw.splitlines() if line.strip()]
        groq_client.set_keys(keys)
        groq_client.set_use_local_fallback(self.local_fallback_var.get())
        if self.on_saved:
            self.on_saved()
        self.destroy()
