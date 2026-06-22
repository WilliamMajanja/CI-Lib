"""Terminal display utilities for Tank & Dozer."""

import shutil

SEP = "─"
HEADER_COLORS = {
    "critical": "\033[91m",  # red
    "high":     "\033[93m",  # yellow
    "medium":   "\033[94m",  # blue
    "low":      "\033[92m",  # green
    "info":     "\033[96m",  # cyan
    "reset":    "\033[0m",
    "bold":     "\033[1m",
    "dim":      "\033[2m",
}


def color(text, style="info"):
    c = HEADER_COLORS.get(style, HEADER_COLORS["info"])
    return f"{c}{text}{HEADER_COLORS['reset']}"


def bold(text):
    return f"{HEADER_COLORS['bold']}{text}{HEADER_COLORS['reset']}"


def dim(text):
    return f"{HEADER_COLORS['dim']}{text}{HEADER_COLORS['reset']}"


def header(title, level="info"):
    width = min(shutil.get_terminal_size().columns, 80)
    print()
    print(color(f"{'─' * width}", level))
    print(color(bold(f"  {title}"), level))
    print(color(f"{'─' * width}", level))


def table(headers, rows):
    """Print a simple formatted table."""
    if not rows:
        print(dim("  (no data)"))
        return
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    print()
    print(bold(fmt.format(*headers)))
    print(dim(fmt.format(*["─" * w for w in col_widths])))
    for row in rows:
        print(fmt.format(*row))
    print()


def kv(key, value, indent=0):
    pad = "  " * indent
    print(f"{pad}{bold(key)}: {value}")


def success(msg):
    print(f"  {color('[+]', 'low')} {msg}")


def info(msg):
    print(f"  {color('[*]', 'info')} {msg}")


def warn(msg):
    print(f"  {color('[!]', 'high')} {msg}")


def error(msg):
    print(f"  {color('[-]', 'critical')} {msg}")


def divider():
    width = min(shutil.get_terminal_size().columns, 80)
    print(dim(f"{'─' * width}"))
