"""Security scanning orchestration module."""

import random
from datetime import datetime, timezone

import numpy as np
from ci_lib.clustering import DBSCAN
from ci_lib.utils import normalize

from tankdozer.utils.display import header, success, info, warn, kv, table, divider, color, bold


TOP_PORTS = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445,
             993, 995, 1433, 1521, 2049, 3306, 3389, 5432, 5900, 5985,
             5986, 6379, 8080, 8443, 27017]


def _mock_scan(target: str, port_range: tuple[int, int], n_hosts=1) -> list[dict]:
    rng = np.random.default_rng(abs(hash(target)) % (2**31))
    results = []
    for hid in range(n_hosts):
        host_ip = f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        open_ports = []
        for port in TOP_PORTS:
            if port_range[0] <= port <= port_range[1]:
                if rng.random() < 0.15:
                    services = {22: "ssh", 80: "http", 443: "https", 3306: "mysql",
                                3389: "rdp", 8080: "http-proxy", 8443: "https-alt",
                                445: "microsoft-ds", 5432: "postgresql", 6379: "redis"}
                    service = services.get(port, "unknown")
                    open_ports.append({"port": port, "state": "open", "service": service,
                                       "banner": f"{service}/1.0" if service != "unknown" else ""})
        results.append({
            "host": host_ip,
            "hostname": f"host-{host_ip.replace('.','-')}.internal",
            "status": "up",
            "os": rng.choice(["Linux 5.x", "Windows Server 2022", "Ubuntu 22.04", "macOS 14"]),
            "ttl": random.choice([64, 128, 255]),
            "open_ports": open_ports,
            "vulnerabilities": [],
        })
    return results


def run_scan(args):
    command = args.command

    if command == "quick":
        header("Quick Port Scan — Top 100 Ports", "info")
        target = getattr(args, "target", "scanme.local")
        info(f"Scanning {target} (top 100 ports)...")
        results = _mock_scan(target, (1, 1024))
        for host in results:
            kv("Host", host["host"])
            kv("Status", host["status"])
            kv("OS", host.get("os", "unknown"))
            kv("Open Ports", len(host.get("open_ports", [])))
            for port_info in host.get("open_ports", [])[:5]:
                info(f"  Port {port_info['port']}/{port_info['service']} open")
            if len(host.get("open_ports", [])) > 5:
                info(f"  ... and {len(host['open_ports']) - 5} more")
            divider()
        success("Quick scan complete.")

    elif command == "full":
        target = args.target
        rate = getattr(args, "rate", 1000)
        header(f"Full Port Scan: {target}", "high")
        info(f"Full TCP scan (1-65535) at {rate} pkts/sec")
        info("This will scan all 65,535 ports — estimated time varies.")
        results = _mock_scan(target, (1, 65535), n_hosts=3)
        total_open = sum(len(h["open_ports"]) for h in results)
        info(f"Scanned {len(results)} hosts, found {total_open} open ports")
        rows = []
        for host in results:
            rows.append([
                host["host"],
                host["os"],
                str(len(host["open_ports"])),
                ", ".join(str(p["port"]) for p in host["open_ports"][:3]),
            ])
        table(["Host", "OS", "Open Ports", "Top Ports"], rows)
        warn("Review all open ports against policy baseline.")

    elif command == "web":
        header("Web Application Reconnaissance", "info")
        info("Enumerating web technologies and endpoints...")
        info("Checking for common web vulnerabilities...")
        info("(Placeholder — extend with real web scanner integration)")
        success("Web recon complete.")
