#!/usr/bin/env python3
import os
import sys
import yaml
from src.carla_server import CarlaServer, CarlaConnectionError
from src.collector import DataCollector


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
    server = CarlaServer(config)

    try:
        collector = DataCollector(config, server)
        collector.run()
    except CarlaConnectionError:
        server.shutdown()
        raise


if __name__ == "__main__":
    main()
