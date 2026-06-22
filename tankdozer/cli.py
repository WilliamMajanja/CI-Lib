"""Tank & Dozer CLI entry point."""

from __future__ import annotations

import argparse
import sys

from tankdozer import __version__, __app_name__
from tankdozer.modules.ir import run_ir_workflow
from tankdozer.modules.analyze import run_analysis
from tankdozer.modules.report import run_report
from tankdozer.modules.ioc import run_ioc
from tankdozer.modules.scan import run_scan
from tankdozer.utils.display import color, bold, dim


BANNER_ART = r"""
▄▄▄█████▓ ▄▄▄       ███▄    █  ██ ▄█▀ ▓█████▄  ▒█████   ▒█████  ▒███████▒
▓  ██▒ ▓▒▒████▄     ██ ▀█   █  ██▄█▒ ▒██▀ ██▌▒██▒  ██▒▒██▒  ██▒▒ ▒ ▒ ▄▀░
▒ ▓██░ ▒░▒██  ▀█▄  ▓██  ▀█ ██▒▓███▄░ ░██   █▌▒██░  ██▒▒██░  ██▒░ ▒ ▄▀▒░
░ ▓██▓ ░ ░██▄▄▄▄██ ▓██▒  ▐▌██▒▓██ █▄ ░▓█▄   ▌▒██   ██░▒██   ██░  ▄▀▒   ░
  ▒██▒ ░  ▓█   ▓██▒▒██░   ▓██░▒██▒ █▄░▒████▓ ░ ████▓▒░░ ████▓▒░▒███████▒
  ▒ ░░    ▒▒   ▓▒█░░ ▒░   ▒ ▒ ▒ ▒▒ ▓▒ ▒▒▓  ▒ ░ ▒░▒░▒░ ░ ▒░▒░▒░ ░▒▒ ▓░▒░▒
    ░      ▒   ▒▒ ░░ ░░   ░ ▒░░ ░▒ ▒░ ░ ▒  ▒   ░ ▒ ▒░   ░ ▒ ▒░ ░░▒ ▒ ░ ▒
  ░        ░   ▒      ░   ░ ░ ░ ░░ ░  ░ ░  ░ ░ ░ ░ ▒  ░ ░ ░ ▒  ░ ░ ░ ░ ░
               ░  ░         ░ ░  ░      ░        ░ ░      ░ ░      ░ ░
"""


def print_banner() -> None:
    width = 60
    term_width = 80
    try:
        import shutil
        term_width = min(shutil.get_terminal_size().columns, 100)
    except Exception:
        pass

    print()
    for line in BANNER_ART.splitlines():
        if line.strip():
            print(color(line.center(term_width), "critical"))
    print(color(f"{'═' * width}".center(term_width), "dim"))
    title = f"  {__app_name__}  v{__version__}  "
    print(bold(color(title.center(term_width), "high")))
    print(color(f"  Cybersecurity Incident Response Framework  ".center(term_width), "info"))
    print(color(f"  Computational Intelligence Security Operations  ".center(term_width), "medium"))
    print(color(f"{'─' * width}".center(term_width), "dim"))
    print()


def cli():
    parser = argparse.ArgumentParser(
        prog="tankdozer",
        description=f"{__app_name__} — Cybersecurity Incident Response Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tankdozer ir init              Start a new incident response case
  tankdozer analyze logs log.csv  Analyze log data for anomalies
  tankdozer scan network 10.0.0.0/24   Orchestrate network scan
  tankdozer ioc enrich 185.130.5.190  Enrich indicator of compromise
  tankdozer report generate      Generate incident report
        """,
    )

    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="module", required=True)

    # ── ir ────────────────────────────────────────────────
    ir_p = sub.add_parser("ir", help="Incident response workflow management")
    ir_sub = ir_p.add_subparsers(dest="command", required=True)
    ir_init = ir_sub.add_parser("init", help="Initialize a new IR case")
    ir_init.add_argument("case_name", nargs="?", default="untitled", help="Case name")
    ir_init.add_argument("--severity", choices=["low", "medium", "high", "critical"], default="medium")
    ir_sub.add_parser("status", help="Show current IR case status")
    ir_sub.add_parser("close", help="Close the current IR case")

    # ── analyze ────────────────────────────────────────────
    an_p = sub.add_parser("analyze", help="Analyze security telemetry using CI algorithms")
    an_sub = an_p.add_subparsers(dest="command", required=True)
    an_logs = an_sub.add_parser("logs", help="Cluster & detect anomalies in log data")
    an_logs.add_argument("file", help="Path to CSV log file")
    an_logs.add_argument("--clusters", type=int, default=3, help="Number of behavior clusters")
    an_sub.add_parser("network", help="Analyze network flow data (traffic patterns)")
    an_net = an_sub.add_parser("traffic", help="Traffic pattern analysis")

    # ── scan ──────────────────────────────────────────────
    sc_p = sub.add_parser("scan", help="Orchestrate security scanning operations")
    sc_sub = sc_p.add_subparsers(dest="command", required=True)
    sc_sub.add_parser("quick", help="Quick port scan (top 100 ports)")
    sc_full = sc_sub.add_parser("full", help="Full port scan (1-65535)")
    sc_full.add_argument("target", help="Target IP or CIDR")
    sc_full.add_argument("--rate", type=int, default=1000, help="Packet rate")
    sc_web = sc_sub.add_parser("web", help="Web application reconnaissance")

    # ── ioc ────────────────────────────────────────────────
    ioc_p = sub.add_parser("ioc", help="Indicator of Compromise enrichment & analysis")
    ioc_sub = ioc_p.add_subparsers(dest="command", required=True)
    ioc_enrich = ioc_sub.add_parser("enrich", help="Enrich a single IOC")
    ioc_enrich.add_argument("indicator", help="IP, domain, hash, or URL")
    ioc_enrich.add_argument("--type", choices=["ip", "domain", "hash", "url"], help="IOC type (auto-detect if omitted)")
    ioc_sub.add_parser("bulk", help="Bulk IOC enrichment from file")
    ioc_bulk = ioc_sub.add_parser("cluster", help="Cluster IOCs by similarity")
    ioc_bulk.add_argument("file", help="CSV of IOCs to cluster")

    # ── report ──────────────────────────────────────────────
    rp_p = sub.add_parser("report", help="Generate security incident reports")
    rp_sub = rp_p.add_subparsers(dest="command", required=True)
    rp_sub.add_parser("generate", help="Generate comprehensive incident report")
    rp_gen = rp_sub.add_parser("executive", help="Generate executive summary report")
    rp_sub.add_parser("timeline", help="Generate incident timeline report")

    args = parser.parse_args()
    print_banner()

    try:
        dispatch(args)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        sys.exit(1)


def dispatch(args):
    module = args.module

    if module == "ir":
        run_ir_workflow(args)
    elif module == "analyze":
        run_analysis(args)
    elif module == "scan":
        run_scan(args)
    elif module == "ioc":
        run_ioc(args)
    elif module == "report":
        run_report(args)
    else:
        print(f"[!] Unknown module: {module}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
