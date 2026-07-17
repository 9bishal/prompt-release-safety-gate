"""
CLI entrypoint for the Prompt Release Safety Gate.

Usage:
    python scripts/run_gate_cli.py --old prompts/old_prompt.txt --new prompts/new_prompt.txt

Exits with code 1 if the verdict is FAIL, so CI can block the merge on it.
Writes reports/gate_report.md and reports/gate_report.json for inspection
and for posting as a PR comment.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from src import config  # noqa: E402
from src.gate import run_gate  # noqa: E402
from src.report import render_markdown  # noqa: E402

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run the Prompt Release Safety Gate")
    parser.add_argument("--old", default="prompts/old_prompt.txt", help="Path to the current production prompt")
    parser.add_argument("--new", default="prompts/new_prompt.txt", help="Path to the candidate prompt")
    parser.add_argument("--report-dir", default="reports", help="Where to write gate_report.md/.json")
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        console.print("[bold red]GROQ_API_KEY not set.[/bold red] Copy .env.example to .env and add your key.")
        sys.exit(1)

    console.print(Panel.fit(
        f"Comparing:\n  old = {args.old}\n  new = {args.new}",
        title="Prompt Release Safety Gate",
    ))

    report = run_gate(args.old, args.new)

    os.makedirs(args.report_dir, exist_ok=True)
    md = render_markdown(report)
    with open(os.path.join(args.report_dir, "gate_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(args.report_dir, "gate_report.json"), "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2)

    console.print(md)

    if report.verdict == "FAIL":
        console.print("\n[bold red]GATE FAILED — blocking merge.[/bold red]")
        sys.exit(1)
    elif report.verdict == "WARN":
        console.print("\n[bold yellow]GATE PASSED WITH WARNINGS.[/bold yellow]")
        sys.exit(0)
    else:
        console.print("\n[bold green]GATE PASSED.[/bold green]")
        sys.exit(0)


if __name__ == "__main__":
    main()
