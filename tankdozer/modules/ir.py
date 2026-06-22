"""Incident response workflow management module."""

import json
from datetime import datetime, timezone
from pathlib import Path

from tankdozer.utils.display import header, success, info, warn, kv, divider, color, bold, table


CASE_DIR = Path.home() / ".tankdozer" / "cases"
CASE_DIR.mkdir(parents=True, exist_ok=True)

IR_PHASES = [
    "Preparation",
    "Identification",
    "Containment",
    "Eradication",
    "Recovery",
    "Lessons Learned",
]

DEFAULT_PLAYBOOK = {
    "Preparation": [
        "Deploy monitoring tools",
        "Establish baselines",
        "Create IR playbooks",
        "Train incident response team",
    ],
    "Identification": [
        "Collect initial alert data",
        "Correlate across log sources",
        "Determine scope of compromise",
        "Classify incident severity",
    ],
    "Containment": [
        "Isolate affected systems",
        "Block malicious IPs/domains",
        "Preserve forensic evidence",
        "Apply temporary mitigations",
    ],
    "Eradication": [
        "Remove malware/backdoors",
        "Patch vulnerabilities",
        "Reset compromised credentials",
        "Validate eradication complete",
    ],
    "Recovery": [
        "Restore systems from backup",
        "Monitor for re-infection",
        "Return to normal operations",
        "Verify system integrity",
    ],
    "Lessons Learned": [
        "Conduct post-mortem analysis",
        "Update detection rules",
        "Improve security controls",
        "Document findings",
    ],
}


def _load_case(name: str | None = None) -> dict:
    if name is None:
        files = sorted(CASE_DIR.glob("*.json"))
        if not files:
            return {}
        return json.loads(files[-1].read_text())
    path = CASE_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_case(case: dict):
    name = case.get("name", "untitled")
    path = CASE_DIR / f"{name}.json"
    path.write_text(json.dumps(case, indent=2, default=str))
    return path


def run_ir_workflow(args):
    command = args.command

    if command == "init":
        case = {
            "name": args.case_name,
            "severity": args.severity,
            "status": "open",
            "created": datetime.now(timezone.utc).isoformat(),
            "phases": {p: {"status": "pending", "tasks": DEFAULT_PLAYBOOK[p]} for p in IR_PHASES},
            "findings": [],
            "artifacts": [],
        }
        path = _save_case(case)
        header(f"IR Case: {args.case_name}", args.severity)
        kv("Case Name", args.case_name)
        kv("Severity", args.severity.upper())
        kv("Status", "OPEN")
        kv("Created", case["created"])
        kv("Phase", "Preparation")
        kv("Case File", str(path))
        success("Incident response case initialized.")
        print()
        info("Next steps:")
        for task in DEFAULT_PLAYBOOK["Preparation"][:3]:
            info(f"  → {task}")

    elif command == "status":
        cases = sorted(CASE_DIR.glob("*.json"))
        if not cases:
            warn("No active IR cases found.")
            warn("Start one with: tankdozer ir init <case_name>")
            return
        case = _load_case()
        header(f"IR Case Status: {case.get('name', 'unknown')}", case.get("severity", "info"))
        kv("Name", case.get("name"))
        kv("Severity", case.get("severity", "unknown").upper())
        kv("Status", case.get("status", "unknown").upper())
        kv("Created", case.get("created", ""))

        rows = []
        for phase in IR_PHASES:
            pdata = case.get("phases", {}).get(phase, {})
            status = pdata.get("status", "pending")
            done = sum(1 for t in pdata.get("tasks", []) if t.startswith("[x]"))
            total = len(pdata.get("tasks", []))
            rows.append([phase, status.upper(), f"{done}/{total}"])
        table(["Phase", "Status", "Tasks Complete"], rows)

    elif command == "close":
        case = _load_case()
        if not case:
            warn("No active IR case to close.")
            return
        case["status"] = "closed"
        case["closed"] = datetime.now(timezone.utc).isoformat()
        _save_case(case)
        header(f"Case Closed: {case['name']}", "info")
        success("Incident response case closed.")
        kv("Case", case["name"])
        kv("Closed", case["closed"])
        info("Don't forget the Lessons Learned phase!")
