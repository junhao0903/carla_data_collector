#!/usr/bin/env python3
import os, sys, subprocess, time, yaml
from src.carla_server import CarlaServer, CarlaConnectionError
from src.collector import DataCollector

CARLA_START_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "scripts", "start_carla.sh")


def _start_carla():
    proc = subprocess.Popen(
        ["bash", CARLA_START_SCRIPT],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Waiting for CARLA to be ready...")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"CARLA start script failed with exit code {ret}")


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    layout_path = config.get("sensor_layout")
    if layout_path:
        if not os.path.isabs(layout_path):
            if "/" not in layout_path:
                layout_path = os.path.join("config", "sensor", layout_path)
            else:
                layout_path = os.path.join(os.path.dirname(config_path), layout_path)
        with open(layout_path, "r") as f:
            config["_sensor_layout"] = yaml.safe_load(f)

    filter_path = config.get("ground_truth", {}).get("annotation_filter")
    if filter_path:
        if not os.path.isabs(filter_path):
            if "/" not in filter_path:
                filter_path = os.path.join("config", "filter", filter_path)
            else:
                filter_path = os.path.join(os.path.dirname(config_path), filter_path)
        with open(filter_path, "r") as f:
            config["_filter_config"] = yaml.safe_load(f)

    return config


def main():
    config_name = "default.yaml"
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    if "/" not in config_name:
        config_path = os.path.join("config", "main", config_name)
    else:
        config_path = config_name

    config = load_config(config_path)

    # Always start from a clean state: kill any leftover CARLA, then launch.
    started_by_us = True
    stop_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "scripts", "stop_carla.sh")
    subprocess.run(["bash", stop_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _start_carla()

    # CARLA port may be open but server not yet ready for RPC — poll until alive
    import carla
    client = None
    for _ in range(30):
        try:
            client = carla.Client(config["carla"]["host"], config["carla"]["port"])
            client.set_timeout(3)
            client.get_server_version()
            break
        except RuntimeError:
            time.sleep(1)
    else:
        raise RuntimeError("CARLA did not become responsive within 30s")

    server = CarlaServer(config)

    try:
        collector = DataCollector(config, server)
        collector.run()
    except CarlaConnectionError:
        server.shutdown()
        raise
    finally:
        if started_by_us:
            stop_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "scripts", "stop_carla.sh")
            subprocess.run(["bash", stop_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("CARLA stopped.")


if __name__ == "__main__":
    main()
