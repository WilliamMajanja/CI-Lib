"""Security telemetry analysis module using CI-Lib algorithms."""

import numpy as np

from ci_lib.clustering import KMeans, DBSCAN
from ci_lib.utils import normalize

from tankdozer.utils.display import header, success, info, warn, kv, table, divider, color, bold
from tankdozer.utils.io import read_csv, generate_sample_logs


def run_analysis(args):
    command = args.command

    if command == "logs":
        filepath = args.file
        header(f"Log Analysis: {filepath}", "info")
        generate_sample_logs(filepath, n_entries=200)
        info("Loading and preprocessing log data...")
        rows = read_csv(filepath)
        info(f"Loaded {len(rows)} log entries")

        X = np.array([
            [float(r.get("dst_port", 0)), float(r.get("bytes_sent", 0)),
             float(r.get("bytes_recv", 0)), float(r.get("duration", 0))]
            for r in rows
        ], dtype=np.float64)

        X_norm = normalize(X, method="zscore")
        n_clusters = args.clusters

        info(f"Running K-Means anomaly detection (k={n_clusters})...")
        km = KMeans(n_clusters=n_clusters, seed=42)
        km.fit(X_norm)
        labels = km.labels_
        inertias = [np.sum((X_norm[labels == i] - km.cluster_centers_[i]) ** 2)
                    for i in range(n_clusters)]

        info(f"Running DBSCAN for noise/outlier detection...")
        db = DBSCAN(eps=0.8, min_samples=5)
        db.fit(X_norm)
        noise_mask = db.labels_ == -1

        cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]
        rows_t = []
        for i in range(n_clusters):
            size = cluster_sizes[i]
            pct = size / len(rows) * 100
            outliers_in_cluster = int(np.sum((labels == i) & noise_mask))
            label = "ANOMALOUS" if outliers_in_cluster > size * 0.3 else "NORMAL"
            rows_t.append([f"Cluster {i}", str(size), f"{pct:.1f}%", str(outliers_in_cluster), label])

        header("Clustering Results", "info")
        table(["Cluster", "Size", "Percentage", "Outliers", "Classification"], rows_t)

        n_outliers = int(np.sum(noise_mask))
        header("Anomaly Detection Summary", "high" if n_outliers > 0 else "low")
        kv("Total Entries", len(rows))
        kv("Anomalies Detected", n_outliers)
        kv("Anomaly Rate", f"{n_outliers / len(rows) * 100:.1f}%")
        kv("Behavior Clusters", n_clusters)

        if n_outliers > 0:
            warn(f"Found {n_outliers} anomalous entries requiring investigation!")
            anomaly_rows = [rows[i] for i in np.where(noise_mask)[0][:5]]
            for a in anomaly_rows:
                kv("Suspicious", f"{a.get('src_ip','?')}:{a.get('dst_port','?')} → {a.get('action','?')}", 1)
        else:
            success("No anomalies detected — traffic patterns appear normal.")

        sil = km.silhouette_score(X_norm)
        kv("Silhouette Score", f"{sil:.3f}")
        divider()

    elif command == "network":
        header("Network Flow Analysis", "info")
        generate_sample_logs("/tmp/tankdozer_netflow.csv", n_entries=300)
        rows = read_csv("/tmp/tankdozer_netflow.csv")
        info(f"Analyzing {len(rows)} network flows")

        X = np.array([
            [float(r.get("bytes_sent", 0)), float(r.get("bytes_recv", 0)),
             float(r.get("duration", 0)), float(r.get("dst_port", 0))]
            for r in rows
        ], dtype=np.float64)
        X_norm = normalize(X, method="zscore")

        db = DBSCAN(eps=0.5, min_samples=10)
        db.fit(X_norm)
        n_clusters = db.n_clusters_
        n_noise = int(np.sum(db.labels_ == -1))

        kv("Traffic Profiles Identified", n_clusters)
        kv("Anomalous Flows", n_noise)
        kv("Noise Ratio", f"{n_noise / len(rows) * 100:.1f}%")

        if n_noise > 0:
            warn("Suspicious traffic patterns detected — investigate immediately!")

    elif command == "traffic":
        header("Traffic Pattern Analysis", "info")
        info("Analyzing traffic patterns using time-series clustering...")
        info("(Placeholder — extend with real packet capture analysis)")
