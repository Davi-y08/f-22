from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from agent.manager import AgentManager
from discovery.service import (
    DiscoveredCamera,
    describe_camera,
    discover_cameras,
    resolve_camera_source,
    save_camera_selection,
)
from utils.config import build_default_raw_config, load_config, load_raw_config, save_raw_config
from utils.logger import configure_logging, get_logger


def _runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _looks_like_model_path(reference: str) -> bool:
    lowered = reference.lower().strip()
    if any(token in lowered for token in ("/", "\\")):
        return True
    return lowered.endswith((".pt", ".onnx", ".engine", ".torchscript"))


class StealthLensDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Stealth Lens Agent")
        self.root.geometry("1080x700")
        self.root.minsize(980, 620)

        self.runtime_base_dir = _runtime_base_dir()
        default_config = (self.runtime_base_dir / "config.json").resolve()
        self.config_path_var = tk.StringVar(value=str(default_config))
        self.camera_name_var = tk.StringVar(value="")
        self.username_var = tk.StringVar(value="")
        self.password_var = tk.StringVar(value="")
        self.manual_rtsp_var = tk.StringVar(value="")
        self.skip_validation_var = tk.BooleanVar(value=False)
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.selected_camera_var = tk.StringVar(value="Nenhuma câmera selecionada")
        self.status_var = tk.StringVar(value="Pronto")

        self._event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._discovered: list[DiscoveredCamera] = []
        self._resolution_result_source: str | int | None = None
        self._busy = False

        self._manager: AgentManager | None = None
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._auto_start_after_save = False

        self._config_bootstrap_message = self._bootstrap_config_file()
        self._configure_logging()
        self.logger = get_logger("stealth_lens.desktop")

        self._build_styles()
        self._build_layout()
        if self._config_bootstrap_message:
            self._append_log(self._config_bootstrap_message)
        self._append_log(f"Config em uso: {self.config_path_var.get()}")
        self._set_status("Pronto para descobrir câmeras.")
        self._append_log("Interface desktop iniciada.")

        self.root.after(120, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_logging(self) -> None:
        raw = load_raw_config(self.config_path_var.get())
        logging_config = raw.get("logging", {}) if isinstance(raw, dict) else {}
        configure_logging(
            level=str(logging_config.get("level", "INFO")),
            json_output=bool(logging_config.get("json", True)),
        )

    def _bootstrap_config_file(self) -> str | None:
        config_path = Path(self.config_path_var.get()).expanduser().resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            try:
                normalized = load_raw_config(config_path)
                save_raw_config(config_path, normalized)
            except Exception:
                return None
            return None

        save_raw_config(config_path, build_default_raw_config())
        return "Config criado automaticamente com padrão inicial."

    def _build_styles(self) -> None:
        self.root.configure(bg="#0a1220")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background="#0a1220")
        style.configure("Panel.TFrame", background="#13233a")
        style.configure(
            "Title.TLabel",
            background="#0a1220",
            foreground="#e6f2ff",
            font=("Segoe UI Semibold", 19),
        )
        style.configure(
            "Body.TLabel",
            background="#13233a",
            foreground="#d7e9ff",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Status.TLabel",
            background="#0a1220",
            foreground="#93c8ff",
            font=("Segoe UI", 10),
        )
        style.configure("Action.TButton", font=("Segoe UI Semibold", 10), padding=(10, 8))
        style.map("Action.TButton", background=[("active", "#3ca0e2")])
        style.configure(
            "Treeview",
            background="#0f1d30",
            foreground="#d9ebff",
            fieldbackground="#0f1d30",
            rowheight=28,
        )
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10), background="#1e3555", foreground="#eaf4ff")

    def _build_layout(self) -> None:
        wrapper = ttk.Frame(self.root, style="App.TFrame", padding=16)
        wrapper.pack(fill="both", expand=True)

        title = ttk.Label(wrapper, text="Stealth Lens Agent", style="Title.TLabel")
        title.pack(anchor="w")

        subtitle = ttk.Label(
            wrapper,
            text="Fluxo simples: 1) Descobrir câmera  2) Salvar  3) Iniciar monitoramento",
            style="Status.TLabel",
        )
        subtitle.pack(anchor="w", pady=(0, 12))

        top_panel = ttk.Frame(wrapper, style="Panel.TFrame", padding=12)
        top_panel.pack(fill="x")

        ttk.Label(top_panel, text="Arquivo de Configuração", style="Body.TLabel").grid(row=0, column=0, sticky="w")
        config_entry = ttk.Entry(top_panel, textvariable=self.config_path_var, width=52)
        config_entry.grid(row=1, column=0, padx=(0, 8), sticky="we")
        top_panel.columnconfigure(0, weight=1)

        discover_button = ttk.Button(
            top_panel,
            text="1. Descobrir",
            command=self._start_discovery,
            style="Action.TButton",
        )
        discover_button.grid(row=1, column=1, padx=(4, 4), sticky="ew")
        self.discover_button = discover_button

        validate_button = ttk.Button(
            top_panel,
            text="2. Validar",
            command=self._validate_selected_camera,
            style="Action.TButton",
        )
        validate_button.grid(row=1, column=2, padx=(4, 4), sticky="ew")
        self.validate_button = validate_button

        save_button = ttk.Button(
            top_panel,
            text="3. Salvar",
            command=self._save_selected_camera,
            style="Action.TButton",
        )
        save_button.grid(row=1, column=3, padx=(4, 0), sticky="ew")
        self.save_button = save_button

        content = ttk.Frame(wrapper, style="App.TFrame")
        content.pack(fill="both", expand=True, pady=(12, 8))
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(content, style="Panel.TFrame", padding=10)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ttk.Label(left_panel, text="Câmeras Encontradas", style="Body.TLabel").pack(anchor="w")
        columns = ("tipo", "nome", "host", "rtsp")
        tree = ttk.Treeview(left_panel, columns=columns, show="headings", selectmode="browse", height=12)
        tree.heading("tipo", text="Tipo")
        tree.heading("nome", text="Nome")
        tree.heading("host", text="Host/Index")
        tree.heading("rtsp", text="RTSP")
        tree.column("tipo", width=80, anchor="center")
        tree.column("nome", width=230, anchor="w")
        tree.column("host", width=140, anchor="w")
        tree.column("rtsp", width=120, anchor="w")
        tree.pack(fill="both", expand=True, pady=(8, 6))
        tree.bind("<<TreeviewSelect>>", self._on_camera_selected)
        self.cameras_tree = tree

        right_panel = ttk.Frame(content, style="Panel.TFrame", padding=10)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right_panel.columnconfigure(0, weight=1)

        ttk.Label(right_panel, text="Configuração da Câmera", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        form = ttk.Frame(right_panel, style="Panel.TFrame")
        form.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Selecionada", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(form, textvariable=self.selected_camera_var, style="Body.TLabel").grid(
            row=0, column=1, sticky="w", pady=4
        )

        ttk.Label(form, text="Nome da Câmera", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.camera_name_var).grid(row=1, column=1, sticky="ew", pady=4)

        self.advanced_toggle_button = ttk.Button(
            form,
            text="Mostrar Opções Avançadas",
            command=self._toggle_advanced,
            style="Action.TButton",
        )
        self.advanced_toggle_button.grid(row=2, column=1, sticky="w", pady=(4, 6))

        advanced = ttk.Frame(form, style="Panel.TFrame")
        advanced.grid(row=3, column=0, columnspan=2, sticky="ew")
        advanced.columnconfigure(1, weight=1)
        self.advanced_frame = advanced

        ttk.Label(advanced, text="Usuário RTSP", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(advanced, textvariable=self.username_var).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(advanced, text="Senha RTSP", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(advanced, textvariable=self.password_var, show="*").grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(advanced, text="RTSP Manual", style="Body.TLabel").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(advanced, textvariable=self.manual_rtsp_var).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(
            advanced,
            text="Salvar mesmo sem validar stream",
            variable=self.skip_validation_var,
        ).grid(row=3, column=1, sticky="w", pady=(6, 2))

        if not self.show_advanced_var.get():
            self.advanced_frame.grid_remove()

        controls = ttk.Frame(right_panel, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 10))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self.start_button = ttk.Button(
            controls,
            text="Iniciar Monitoramento",
            command=self._start_monitoring,
            style="Action.TButton",
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_button = ttk.Button(
            controls,
            text="Parar Monitoramento",
            command=self._stop_monitoring,
            style="Action.TButton",
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.save_start_button = ttk.Button(
            controls,
            text="Salvar e Iniciar",
            command=self._save_and_start_selected_camera,
            style="Action.TButton",
        )
        self.save_start_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        log_label = ttk.Label(right_panel, text="Log Operacional", style="Body.TLabel")
        log_label.grid(row=3, column=0, sticky="w")

        log_text = tk.Text(
            right_panel,
            height=14,
            bg="#081425",
            fg="#d6ebff",
            insertbackground="#d6ebff",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        log_text.grid(row=4, column=0, sticky="nsew", pady=(6, 0))
        right_panel.rowconfigure(4, weight=1)
        self.log_text = log_text

        status = ttk.Label(wrapper, textvariable=self.status_var, style="Status.TLabel")
        status.pack(anchor="w", pady=(8, 0))

    def _on_camera_selected(self, _: Any) -> None:
        camera = self._selected_camera()
        if camera is None:
            self.selected_camera_var.set("Nenhuma câmera selecionada")
            return
        self.camera_name_var.set(camera.name)
        self.selected_camera_var.set(describe_camera(camera))
        self._append_log(f"Selecionada: {describe_camera(camera)}")

    def _selected_camera(self) -> DiscoveredCamera | None:
        selection = self.cameras_tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self._discovered):
            return self._discovered[index]
        return None

    def _toggle_advanced(self) -> None:
        showing = bool(self.show_advanced_var.get())
        if showing:
            self.advanced_frame.grid_remove()
            self.show_advanced_var.set(False)
            self.advanced_toggle_button.configure(text="Mostrar Opções Avançadas")
            return

        self.advanced_frame.grid()
        self.show_advanced_var.set(True)
        self.advanced_toggle_button.configure(text="Ocultar Opções Avançadas")

    def _save_and_start_selected_camera(self) -> None:
        if self._busy:
            return
        self._save_selected_camera(auto_start=True)

    def _start_discovery(self) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._set_status("Descobrindo câmeras na rede...")
        self._append_log("Iniciando varredura ONVIF + RTSP + webcams locais.")
        threading.Thread(target=self._discover_worker, name="discover-worker", daemon=True).start()

    def _discover_worker(self) -> None:
        try:
            cameras = discover_cameras(logger=self.logger)
            self._event_queue.put(("discovery-success", cameras))
        except Exception as exc:
            self._event_queue.put(("error", f"Falha na descoberta: {exc}"))
        finally:
            self._event_queue.put(("busy", False))

    def _validate_selected_camera(self) -> None:
        if self._busy:
            return
        camera = self._selected_camera()
        if camera is None:
            messagebox.showwarning("Stealth Lens", "Selecione uma câmera primeiro.")
            return

        self._set_busy(True)
        self._set_status("Validando câmera selecionada...")
        self._append_log(f"Validando source para: {camera.name}")
        threading.Thread(target=self._validate_worker, args=(camera,), name="validate-worker", daemon=True).start()

    def _validate_worker(self, camera: DiscoveredCamera) -> None:
        try:
            result = resolve_camera_source(
                camera=camera,
                username=self.username_var.get().strip(),
                password=self.password_var.get(),
                manual_rtsp_url=self.manual_rtsp_var.get().strip() or None,
                validate_stream=True,
            )
            self._event_queue.put(("validation-result", result))
        except Exception as exc:
            self._event_queue.put(("error", f"Erro na validação: {exc}"))
        finally:
            self._event_queue.put(("busy", False))

    def _save_selected_camera(self, auto_start: bool = False) -> None:
        if self._busy:
            return
        self._auto_start_after_save = bool(auto_start)
        camera = self._selected_camera()
        if camera is None:
            messagebox.showwarning("Stealth Lens", "Selecione uma câmera para salvar.")
            self._auto_start_after_save = False
            return

        camera_name = self.camera_name_var.get().strip() or camera.name
        validate_stream = not self.skip_validation_var.get()

        self._set_busy(True)
        self._set_status("Salvando câmera no config...")
        self._append_log(f"Resolvendo source e salvando câmera: {camera_name}")
        threading.Thread(
            target=self._save_worker,
            args=(camera, camera_name, validate_stream),
            name="save-worker",
            daemon=True,
        ).start()

    def _save_worker(self, camera: DiscoveredCamera, camera_name: str, validate_stream: bool) -> None:
        try:
            result = resolve_camera_source(
                camera=camera,
                username=self.username_var.get().strip(),
                password=self.password_var.get(),
                manual_rtsp_url=self.manual_rtsp_var.get().strip() or None,
                validate_stream=validate_stream,
            )
            if result.source is None:
                message = result.error or "Não foi possível definir source para a câmera."
                self._event_queue.put(("error", message))
                return

            self._resolution_result_source = result.source
            saved_path = save_camera_selection(
                config_path=self.config_path_var.get().strip() or "config.json",
                camera_name=camera_name,
                source=result.source,
                camera_key=camera.key,
            )
            self._event_queue.put(
                (
                    "saved",
                    {
                        "path": str(saved_path),
                        "name": camera_name,
                        "source": result.source,
                        "validated": result.validated,
                    },
                )
            )
        except Exception as exc:
            self._event_queue.put(("error", f"Erro ao salvar câmera: {exc}"))
        finally:
            self._event_queue.put(("busy", False))

    def _start_monitoring(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            messagebox.showinfo("Stealth Lens", "O monitoramento já está em execução.")
            return

        config_path = self.config_path_var.get().strip() or "config.json"
        self._configure_logging()
        self._append_log(f"Iniciando monitoramento com config: {config_path}")
        self._set_status("Inicializando monitoramento...")
        self._monitor_stop.clear()

        self._monitor_thread = threading.Thread(
            target=self._monitor_worker,
            args=(config_path,),
            name="monitor-worker",
            daemon=True,
        )
        self._monitor_thread.start()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def _monitor_worker(self, config_path: str) -> None:
        manager: AgentManager | None = None
        try:
            config = load_config(config_path)
            self._validate_monitoring_config(config)

            manager = AgentManager(config)
            self._manager = manager

            manager.start()
            self._event_queue.put(("monitor-started", f"Monitoramento iniciado com {len(config.cameras)} câmera(s)."))

            time.sleep(1.5)
            snapshot = manager.status_snapshot()
            active_cameras = [camera for camera in snapshot.get("cameras", []) if camera.get("state") != "disabled"]
            if active_cameras and all(camera.get("state") == "error" for camera in active_cameras):
                details = "; ".join(
                    f"{camera.get('name', camera.get('camera_id', 'camera'))}: "
                    f"{camera.get('last_error') or 'erro desconhecido'}"
                    for camera in active_cameras
                )
                raise RuntimeError(f"Todas as câmeras falharam ao iniciar. {details}")

            while not self._monitor_stop.is_set() and not manager.should_stop:
                time.sleep(0.25)
        except Exception as exc:
            self._event_queue.put(("error", f"Falha ao iniciar monitoramento: {exc}"))
        finally:
            if manager is not None:
                try:
                    manager.stop()
                    self._event_queue.put(("monitor-stopped", "Monitoramento finalizado."))
                except Exception as stop_exc:
                    self._event_queue.put(("error", f"Falha ao encerrar monitoramento: {stop_exc}"))
            self._manager = None
            self._event_queue.put(("monitor-ui-reset", None))

    def _validate_monitoring_config(self, config: Any) -> None:
        enabled_cameras = [camera for camera in config.cameras if camera.enabled]
        if not enabled_cameras:
            raise ValueError("Nenhuma câmera habilitada no config. Descubra/salve uma câmera antes de iniciar.")

        issues: list[str] = []
        for camera in enabled_cameras:
            if not camera.models:
                issues.append(f"{camera.name}: sem modelos configurados.")
                continue

            for model_ref in camera.models:
                model_cfg = config.model_catalog.get(model_ref)
                if model_cfg is not None:
                    if not model_cfg.path.exists():
                        issues.append(f"{camera.name}: modelo '{model_ref}' não encontrado em '{model_cfg.path}'.")
                    continue

                if not _looks_like_model_path(model_ref):
                    issues.append(
                        f"{camera.name}: alias '{model_ref}' não existe em model_catalog e não é caminho de modelo."
                    )

        if issues:
            preview = "\n".join(issues[:8])
            raise FileNotFoundError(f"Config inválido para monitoramento:\n{preview}")

    def _stop_monitoring(self) -> None:
        self._monitor_stop.set()
        self._set_status("Encerrando monitoramento...")
        self._append_log("Parada solicitada pelo usuário.")

    def _poll_events(self) -> None:
        while True:
            try:
                event_name, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event_name, payload)

        self.root.after(120, self._poll_events)

    def _handle_event(self, event_name: str, payload: Any) -> None:
        if event_name == "discovery-success":
            self._discovered = self._dedupe_discovered(payload)
            self._refresh_tree()
            total = len(self._discovered)
            self._set_status(f"Descoberta concluída: {total} câmera(s).")
            self._append_log(f"Descoberta concluída com {total} câmera(s).")
            return

        if event_name == "validation-result":
            if payload.source is not None:
                self._resolution_result_source = payload.source
                status = "validada" if payload.validated else "resolvida sem validação"
                self._set_status(f"Câmera {status}: {payload.source}")
                self._append_log(f"Source selecionado: {payload.source}")
                messagebox.showinfo("Stealth Lens", f"Source encontrado: {payload.source}")
            else:
                self._set_status("Validação falhou.")
                self._append_log(payload.error or "Falha na validação.")
                messagebox.showwarning("Stealth Lens", payload.error or "Falha ao validar câmera.")
            return

        if event_name == "saved":
            self._set_status(f"Câmera salva em {payload['path']}")
            self._append_log(
                f"Câmera '{payload['name']}' salva com source={payload['source']} "
                f"(validated={payload['validated']})."
            )
            messagebox.showinfo(
                "Stealth Lens",
                f"Câmera salva com sucesso em:\n{payload['path']}",
            )
            if self._auto_start_after_save:
                self._auto_start_after_save = False
                self._append_log("Iniciando monitoramento automaticamente após salvar.")
                self._start_monitoring()
            return

        if event_name == "monitor-started":
            self._set_status("Monitoramento em execução.")
            self._append_log(str(payload))
            return

        if event_name == "monitor-stopped":
            self._set_status("Monitoramento parado.")
            self._append_log(str(payload))
            return

        if event_name == "monitor-ui-reset":
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return

        if event_name == "busy":
            self._set_busy(bool(payload))
            return

        if event_name == "error":
            self._auto_start_after_save = False
            self._set_status("Erro operacional.")
            self._append_log(str(payload))
            messagebox.showerror("Stealth Lens", str(payload))
            return

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        discover_state = "disabled" if busy else "normal"
        validate_state = "disabled" if busy else "normal"
        save_state = "disabled" if busy else "normal"
        self.discover_button.configure(state=discover_state)
        self.validate_button.configure(state=validate_state)
        self.save_button.configure(state=save_state)
        self.save_start_button.configure(state=save_state)

    def _refresh_tree(self) -> None:
        self.cameras_tree.delete(*self.cameras_tree.get_children())
        for index, camera in enumerate(self._discovered):
            host_value = str(camera.local_index) if camera.kind == "local" else (camera.host or "")
            rtsp_ports = ",".join(str(port) for port in sorted(camera.rtsp_ports)) if camera.rtsp_ports else "-"
            self.cameras_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(camera.kind, camera.name, host_value, rtsp_ports),
            )

        if self._discovered:
            self.cameras_tree.selection_set("0")
            self.cameras_tree.focus("0")
            self._on_camera_selected(None)
            return

        self.selected_camera_var.set("Nenhuma câmera encontrada")

    def _dedupe_discovered(self, cameras: list[DiscoveredCamera]) -> list[DiscoveredCamera]:
        unique: list[DiscoveredCamera] = []
        seen_keys: set[str] = set()

        for camera in cameras:
            source_key = camera.key
            if camera.kind == "local" and camera.local_index is not None:
                source_key = f"local:{camera.local_index}"
            elif camera.host:
                source_key = f"network:{camera.host.lower()}"

            if source_key in seen_keys:
                continue

            seen_keys.add(source_key)
            unique.append(camera)

        return unique

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _on_close(self) -> None:
        self._stop_monitoring()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._append_log("Aguardando finalização do monitoramento...")
            self.root.after(250, self._on_close)
            return
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    StealthLensDesktopApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
