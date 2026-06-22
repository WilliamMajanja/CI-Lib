"""File I/O utilities for Tank & Dozer."""

import csv
import json
from pathlib import Path
from typing import Any


def read_csv(filepath: str) -> list[dict[str, str]]:
    """Read a CSV file and return a list of row dicts."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_json(filepath: str) -> Any:
    """Read a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with path.open() as f:
        return json.load(f)


def write_json(filepath: str, data: Any, indent=2):
    """Write data to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=indent)


def write_report(filepath: str, content: str):
    """Write a text report to file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(content)


def generate_sample_logs(filepath: str, n_entries=200):
    """Generate sample security log data for demonstration."""
    import random
    import numpy as np

    rng = np.random.default_rng(42)
    ips = [f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(8)]
    ports = [22, 80, 443, 3306, 8080, 8443, 3389, 445]
    actions = ["ALLOW", "DENY", "ALERT", "AUTH_FAIL"]

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "src_ip", "dst_port", "action", "bytes_sent", "bytes_recv", "duration"])
        base_ts = 1700000000
        for i in range(n_entries):
            ts = base_ts + i * random.randint(1, 30)
            ip = rng.choice(ips)
            port = rng.choice(ports)
            action = rng.choice(actions, p=[0.7, 0.2, 0.05, 0.05])
            b_sent = int(abs(rng.normal(1024, 512)))
            b_recv = int(abs(rng.normal(4096, 2048)))
            dur = round(abs(rng.normal(2.0, 1.5)), 2)
            writer.writerow([ts, ip, port, action, b_sent, b_recv, dur])

    print(f"[+] Sample log data written to {filepath} ({n_entries} entries)")
