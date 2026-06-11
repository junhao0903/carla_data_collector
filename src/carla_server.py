import os
import glob
import signal
import subprocess
import sys
import time
from pathlib import Path


class CarlaConnectionError(RuntimeError):
    pass


def _import_carla(carla_path=None):
    """Import CARLA — try environment first, fall back to .egg from CARLA path."""
    try:
        import carla
        return carla
    except ImportError:
        pass
    if carla_path is None:
        raise CarlaConnectionError(
            "CARLA Python API not found. Install via pip/conda or set launch_command in config."
        )
    egg_dir = os.path.join(carla_path, "PythonAPI", "carla", "dist")
    eggs = glob.glob(os.path.join(egg_dir, "carla-*py3.7*.egg"))
    if not eggs:
        eggs = glob.glob(os.path.join(egg_dir, "carla-*.egg"))
    if not eggs:
        raise CarlaConnectionError(f"No CARLA .egg found in {egg_dir}")
    if eggs[0] not in sys.path:
        sys.path.insert(0, eggs[0])
    try:
        import carla
        return carla
    except ImportError:
        raise CarlaConnectionError(
            "CARLA Python API not found. Install via pip/conda or set launch_command in config."
        )


class CarlaServer:
    def __init__(self, config):
        cfg = config["carla"]
        self._host = cfg.get("host", "localhost")
        self._port = cfg.get("port", 2000)
        self._timeout = cfg.get("timeout_seconds", 10.0)
        self._startup_wait = cfg.get("startup_wait_seconds", 60)
        self._launch_command = cfg.get("launch_command", [])
        self._auto_launch = cfg.get("auto_launch", True)
        self._process = None
        self._pgid = None
        self._owns_process = False
        carla_path = None
        if self._launch_command:
            script = Path(self._launch_command[0]).expanduser()
            if script.exists():
                carla_path = str(script.resolve().parent)
        self._carla = _import_carla(carla_path)
        self._client = None
        self._world = None

    def connect(self):
        try:
            return self._connect_once()
        except CarlaConnectionError:
            if not self._should_auto_launch():
                raise

        self._launch()
        deadline = time.time() + self._startup_wait
        last_error = None
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise CarlaConnectionError(
                    "CARLA server process exited before accepting connections."
                )
            try:
                return self._connect_once()
            except CarlaConnectionError as exc:
                last_error = exc
                time.sleep(1.0)
        self.shutdown()
        if last_error is not None:
            raise last_error
        raise CarlaConnectionError(
            f"Timed out waiting for CARLA after {self._startup_wait}s"
        )

    def _connect_once(self):
        self._client = self._carla.Client(self._host, self._port)
        self._client.set_timeout(self._timeout)
        try:
            self._world = self._client.get_world()
            self._world.get_map()
        except RuntimeError as exc:
            raise CarlaConnectionError(
                f"Failed to connect to CARLA at {self._host}:{self._port}"
            ) from exc
        return self._world

    def _should_auto_launch(self):
        return (
            self._auto_launch
            and self._host in ("127.0.0.1", "localhost")
            and len(self._launch_command) > 0
        )

    def _launch(self):
        if self._process is not None:
            return
        command = list(self._launch_command)
        binary_path = Path(command[0]).expanduser()
        if not binary_path.exists():
            raise CarlaConnectionError(f"CARLA binary not found: {binary_path}")
        command[0] = str(binary_path)
        try:
            self._process = subprocess.Popen(command, start_new_session=True)
            self._pgid = os.getpgid(self._process.pid)
        except OSError as exc:
            raise CarlaConnectionError(f"Failed to launch CARLA: {command}") from exc
        self._owns_process = True

    def shutdown(self):
        if not self._owns_process or self._pgid is None:
            return
        try:
            os.killpg(self._pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(10):
            try:
                os.killpg(self._pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        else:
            try:
                os.killpg(self._pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._process = None
        self._pgid = None
        self._owns_process = False
        print("CARLA server stopped")

    @property
    def client(self):
        return self._client

    @property
    def world(self):
        return self._world

    @property
    def carla(self):
        return self._carla
