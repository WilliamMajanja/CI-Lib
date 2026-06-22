"""Indicator of Compromise enrichment and analysis module."""

import re
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from ci_lib.clustering import KMeans
from ci_lib.utils import normalize, label_encode

from tankdozer.utils.display import header, success, info, warn, kv, table, divider, bold, color
from tankdozer.utils.io import read_csv


IOC_TYPES = {
    "ip":     r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    "domain": r"^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$",
    "hash":   r"^[a-fA-F0-9]{32,64}$",
    "url":    r"^https?://",
}


def _detect_type(indicator: str) -> str:
    for ioc_type, pattern in IOC_TYPES.items():
        if re.match(pattern, indicator):
            return ioc_type
    return "unknown"


def _mock_enrichment(indicator: str, ioc_type: str) -> dict:
    rng = np.random.default_rng(hash(indicator) % (2**31))
    return {
        "indicator": indicator,
        "type": ioc_type,
        "first_seen": "2024-09-15T12:00:00Z",
        "last_seen": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confidence": round(rng.uniform(30, 95), 1),
        "severity": rng.choice(["low", "medium", "high", "critical"], p=[0.1, 0.3, 0.4, 0.2]),
        "tags": list(rng.choice(["malware", "phishing", "c2", "scan", "bruteforce", "exploit"], size=2, replace=False)),
        "related_ips": [f"{rng.integers(1,255)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,255)}" for _ in range(rng.integers(1, 4))],
        "reported_by": rng.choice(["AlienVault OTX", "VirusTotal", "AbuseIPDB", "ThreatFox"]),
        "description": f"Indicator flagged by threat intelligence feeds. Associated with {rng.choice(['TA505', 'Lazarus', 'APT29', 'UNC1878'])} activity patterns.",
    }


def _enrich_display(enrichment: dict):
    kv("Indicator", enrichment.get("indicator", "?"))
    kv("Type", enrichment.get("type", "?"))
    kv("Confidence", f"{enrichment.get('confidence', 0):.1f}%")
    sev = enrichment.get("severity", "low")
    kv("Severity", sev.upper())
    kv("Tags", ", ".join(enrichment.get("tags", [])))
    kv("First Seen", enrichment.get("first_seen", ""))
    kv("Last Seen", enrichment.get("last_seen", ""))
    kv("Reported By", enrichment.get("reported_by", ""))
    kv("Description", enrichment.get("description", ""))
    related = enrichment.get("related_ips", [])
    if related:
        kv("Related IPs", ", ".join(related))


def run_ioc(args):
    command = args.command

    if command == "enrich":
        indicator = args.indicator
        ioc_type = args.type or _detect_type(indicator)

        header(f"IOC Enrichment: {indicator}", "info")
        kv("Type", ioc_type)
        divider()

        enrichment = _mock_enrichment(indicator, ioc_type)
        _enrich_display(enrichment)
        success("Enrichment complete.")

    elif command == "bulk":
        header("Bulk IOC Enrichment", "info")
        info("Reading IOC list (generating sample data)...")
        sample_iocs = [
            "185.130.5.190", "evil.example.com", "d41d8cd98f00b204e9800998ecf8427e",
            "192.168.1.50", "malware.net", "10.0.0.15",
            "c2-server.org", "45.33.32.156", "scanme.local",
        ]
        results = []
        for ioc in sample_iocs:
            ioc_type = _detect_type(ioc)
            enrichment = _mock_enrichment(ioc, ioc_type)
            results.append(enrichment)

        rows = []
        for r in results:
            rows.append([r["indicator"], r["type"], r["severity"].upper(), f"{r['confidence']:.0f}%", ", ".join(r["tags"])])
        table(["Indicator", "Type", "Severity", "Confidence", "Tags"], rows)
        summary = {"total": len(results), "critical": sum(1 for r in results if r["severity"] == "critical"),
                   "high": sum(1 for r in results if r["severity"] == "high"),
                   "medium": sum(1 for r in results if r["severity"] == "medium"),
                   "low": sum(1 for r in results if r["severity"] == "low")}
        divider()
        for k, v in summary.items():
            kv(k.capitalize(), v)
        success(f"Enriched {len(results)} indicators.")

    elif command == "cluster":
        filepath = args.file
        header(f"IOC Clustering: {filepath}", "info")
        info("Loading and featurizing IOCs...")
        generate_sample_iocs(filepath, n=50)
        rows = read_csv(filepath)
        info(f"Loaded {len(rows)} IOCs for clustering")
        ip_features = np.array([
            [int(parts[0]), int(parts[1]),
             int(parts[2]), int(parts[3])]
            for row in rows
            if (indicator := row.get("indicator", ""))
            and (parts := indicator.split("."))
            and len(parts) == 4
        ], dtype=np.float64)

        if len(ip_features) < 4:
            warn("Not enough IP-based IOCs to cluster meaningfully.")
            return

        X_norm = normalize(ip_features, method="zscore")
        info("Clustering IOCs by similarity (K-Means)...")
        km = KMeans(n_clusters=min(3, len(ip_features) // 5), seed=42)
        km.fit(X_norm)

        cluster_info = []
        for i in range(km.n_clusters):
            members = np.where(km.labels_ == i)[0]
            ips_in_cluster = [rows[idx].get("indicator", "?")
                              for idx in members[:5]]
            cluster_info.append([f"Group {i}", len(members), ", ".join(ips_in_cluster)])

        table(["Cluster", "Size", "Sample IPs"], cluster_info)
        success("IOC clustering complete — investigate each group.")


def generate_sample_iocs(filepath, n=50):
    import csv
    from pathlib import Path
    rng = np.random.default_rng(42)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["indicator", "type", "first_seen"])
        for _ in range(n):
            ip = f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
            writer.writerow([ip, "ip", "2024-10-01"])
