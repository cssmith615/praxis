"""
Shaun CLI

Commands:
  praxis run      "<program>"   — parse, validate, execute; print results
  praxis validate "<program>"   — parse + validate only; print errors or OK
  praxis parse    "<program>"   — parse only; pretty-print AST
  praxis run      --file <path> — run a .shaun file
  praxis goal     "<goal>"      — plan + execute + store; uses Claude API

Examples:
  praxis run "ING.sales.db -> CLN.null -> SUMM.text"
  praxis validate "ING.flights -> BADVERB.something"
  praxis parse "PAR(ING.sales, ING.marketing) -> MERGE"
  praxis run --file monitor.shaun
  praxis goal "check denver flight prices and alert me if under $200"
  praxis goal "summarize my sales data and send to telegram"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich import box

from praxis.grammar import parse
from praxis.validator import validate, ShaunValidationError
from praxis.executor import Executor
from praxis.handlers import HANDLERS
from praxis.memory import ProgramMemory
from praxis.constitution import Constitution
from praxis.planner import Planner, PlanningFailure
from praxis.improver import Improver

console = Console()


@click.group()
@click.version_option("0.1.0", prog_name="shaun")
def main():
    """Shaun — AI-native intermediate language for agentic workflows."""


# ──────────────────────────────────────────────────────────────────────────────
# praxis run
# ──────────────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("program", required=False)
@click.option("--file", "-f", "filepath", type=click.Path(exists=True),
              help="Path to a .shaun file")
@click.option("--mode", "-m", default="dev", type=click.Choice(["dev", "prod"]),
              show_default=True, help="Execution mode")
@click.option("--json-out", is_flag=True, help="Output results as JSON")
def run(program: str | None, filepath: str | None, mode: str, json_out: bool):
    """Parse, validate, and execute a Praxis program."""
    source = _load_source(program, filepath)

    try:
        ast = parse(source)
    except Exception as exc:
        console.print(f"[bold red]Parse error:[/] {exc}")
        sys.exit(1)

    errors = validate(ast, mode=mode)
    if errors:
        console.print("[bold red]Validation errors:[/]")
        for e in errors:
            console.print(f"  [red]•[/] {e}")
        sys.exit(1)

    executor = Executor(HANDLERS, mode=mode)
    results = executor.execute(ast)

    if json_out:
        print(json.dumps(results, indent=2, default=str))
    else:
        _print_results(results)


# ──────────────────────────────────────────────────────────────────────────────
# praxis validate
# ──────────────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("program", required=False)
@click.option("--file", "-f", "filepath", type=click.Path(exists=True))
@click.option("--mode", "-m", default="dev", type=click.Choice(["dev", "prod"]))
def validate_cmd(program: str | None, filepath: str | None, mode: str):
    """Validate a Praxis program without executing it."""
    source = _load_source(program, filepath)

    try:
        ast = parse(source)
    except Exception as exc:
        console.print(f"[bold red]Parse error:[/] {exc}")
        sys.exit(1)

    errors = validate(ast, mode=mode)
    if errors:
        console.print("[bold red]Validation failed:[/]")
        for e in errors:
            console.print(f"  [red]•[/] {e}")
        sys.exit(1)
    else:
        console.print("[bold green]✓[/] Program is valid.")


main.add_command(validate_cmd, name="validate")


# ──────────────────────────────────────────────────────────────────────────────
# praxis parse
# ──────────────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("program", required=False)
@click.option("--file", "-f", "filepath", type=click.Path(exists=True))
def parse_cmd(program: str | None, filepath: str | None):
    """Parse a Praxis program and print the AST."""
    source = _load_source(program, filepath)

    try:
        ast = parse(source)
    except Exception as exc:
        console.print(f"[bold red]Parse error:[/] {exc}")
        sys.exit(1)

    import dataclasses
    def _asdict(obj):
        if dataclasses.is_dataclass(obj):
            return {
                "__type__": type(obj).__name__,
                **{k: _asdict(v) for k, v in dataclasses.asdict(obj).items()},
            }
        if isinstance(obj, list):
            return [_asdict(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _asdict(v) for k, v in obj.items()}
        return obj

    output = json.dumps(_asdict(ast), indent=2, default=str)
    syntax = Syntax(output, "json", theme="monokai", line_numbers=False)
    console.print(syntax)


main.add_command(parse_cmd, name="parse")


# ──────────────────────────────────────────────────────────────────────────────
# praxis goal
# ──────────────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("goal")
@click.option("--mode", "-m", default="dev", type=click.Choice(["dev", "prod"]))
@click.option("--db", default=None, help="Path to program memory SQLite file")
@click.option("--no-execute", is_flag=True, help="Plan only; don't execute")
@click.option("--no-store", is_flag=True, help="Don't store the result in memory")
@click.option("--show-program", is_flag=True, default=True, help="Print the generated program")
def goal(
    goal: str,
    mode: str,
    db: str | None,
    no_execute: bool,
    no_store: bool,
    show_program: bool,
):
    """
    Translate a natural language GOAL into a Praxis program, execute it,
    and store the result in program memory.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    memory = ProgramMemory(db_path=db)
    constitution = Constitution()
    planner = Planner(memory=memory, constitution=constitution, mode=mode)

    # ── Plan ──────────────────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Planning:[/] {goal}")
    if memory.count() > 0:
        console.print(f"[dim]  Program memory: {memory.count()} stored programs[/]")

    try:
        result = planner.plan(goal)
    except PlanningFailure as exc:
        console.print(f"[bold red]Planning failed after {exc.attempts} attempts:[/]")
        console.print(f"  [red]{exc.last_error}[/]")
        sys.exit(1)
    except EnvironmentError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        sys.exit(1)

    # ── Show plan ─────────────────────────────────────────────────────────────
    adapt_label = "[green]adapted[/]" if result.adapted else "[blue]generated[/]"
    console.print(
        f"[bold green]✓[/] Program {adapt_label} "
        f"in {result.attempts} attempt(s)"
    )

    if result.similar:
        top = result.similar[0]
        console.print(
            f"[dim]  Top match: {top.similarity:.2f} similarity "
            f"— \"{top.goal_text[:60]}\"[/]"
        )

    if show_program:
        console.print()
        syntax = Syntax(result.program, "text", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print()

    if no_execute:
        if not no_store:
            pid = memory.store(goal, result.program, "planned", [])
            console.print(f"[dim]Stored as {pid[:8]}… (not executed)[/]")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    console.print("[bold cyan]Executing...[/]")
    try:
        ast = parse(result.program)
        executor = Executor(HANDLERS, mode=mode)
        exec_results = executor.execute(ast)
        outcome = "success" if all(r["status"] != "error" for r in exec_results) else "partial"
    except Exception as exc:
        console.print(f"[bold red]Execution error:[/] {exc}")
        outcome = "failure"
        exec_results = []

    _print_results(exec_results)

    # ── Store ─────────────────────────────────────────────────────────────────
    if not no_store:
        pid = memory.store(goal, result.program, outcome, exec_results)
        console.print(f"[dim]Stored in memory as {pid[:8]}… (outcome: {outcome})[/]")


# ──────────────────────────────────────────────────────────────────────────────
# praxis memory
# ──────────────────────────────────────────────────────────────────────────────

@main.command("memory")
@click.option("--db", default=None, help="Path to program memory SQLite file")
@click.option("-n", default=10, help="Number of recent programs to show")
def memory_cmd(db: str | None, n: int):
    """Show recent programs in the program library."""
    memory = ProgramMemory(db_path=db)
    programs = memory.recent(n)
    if not programs:
        console.print("[dim]Program memory is empty. Run `praxis goal` to populate it.[/]")
        return

    console.print(f"\n[bold]Program Memory[/] — {memory.count()} stored programs\n")
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Goal")
    table.add_column("Outcome")
    table.add_column("Stored")

    for p in programs:
        color = {"success": "green", "failure": "red", "partial": "yellow"}.get(
            p.outcome, "white"
        )
        table.add_row(
            p.id[:8] + "…",
            p.goal_text[:60] + ("…" if len(p.goal_text) > 60 else ""),
            f"[{color}]{p.outcome}[/]",
            p.created_at[:19],
        )
    console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
# praxis improve
# ──────────────────────────────────────────────────────────────────────────────

@main.command("improve")
@click.option("--log", default=None, help="Path to execution log (default: ~/.praxis/execution.log)")
@click.option("--constitution", "const_path", default=None, help="Path to constitution file")
@click.option("--llm", is_flag=True, help="Use Claude API to write rule text (requires ANTHROPIC_API_KEY)")
@click.option("--yes", "-y", is_flag=True, help="Accept all proposals without prompting")
@click.option("--dry-run", is_flag=True, help="Show proposals but don't write anything")
def improve_cmd(
    log: str | None,
    const_path: str | None,
    llm: bool,
    yes: bool,
    dry_run: bool,
):
    """
    Analyze execution history and propose constitutional rules.

    Reads ~/.praxis/execution.log, finds recurring failure patterns,
    generates rule candidates, and appends accepted rules to
    praxis-constitution.md.
    """
    from pathlib import Path

    constitution = Constitution(const_path) if const_path else Constitution()
    improver = Improver(constitution=constitution, log_path=log, use_llm=llm)

    console.print("\n[bold cyan]Analyzing execution history...[/]")

    patterns = improver.analyze()
    if not patterns:
        console.print("[dim]No significant failure patterns found. "
                      "Run more programs to build up history.[/]")
        return

    console.print(f"[green]Found {len(patterns)} failure pattern(s):[/]\n")
    for p in patterns:
        console.print(
            f"  [bold]{p.verb}[/]  {p.count} failures  "
            f"[dim]{p.error_summary[:60]}[/]"
        )

    console.print()
    proposals = improver.propose(patterns)
    if not proposals:
        console.print("[dim]No new rules to propose (all patterns already covered).[/]")
        return

    console.print(f"[bold cyan]Proposing {len(proposals)} rule(s):[/]\n")

    accepted_count = 0
    skipped_count  = 0

    for i, proposal in enumerate(proposals, 1):
        # Display proposal
        console.print(f"[bold]Rule {i}/{len(proposals)}[/]  "
                      f"[dim]source: {proposal.source}[/]")
        console.print(f"  Pattern: [yellow]{proposal.pattern.verb}[/] "
                      f"failed {proposal.pattern.count}×")
        console.print(f"  Verbs:   {', '.join(proposal.verbs)}")
        console.print(f"  Impact:  ~{proposal.affected_programs} programs affected, "
                      f"~{proposal.estimated_prevented} failures prevented")
        console.print()
        console.print(f"  [bold green]Proposed rule:[/]")
        console.print(f"  [italic]{proposal.rule_text}[/]")
        console.print()

        if dry_run:
            console.print("  [dim](dry-run — not written)[/]\n")
            continue

        if yes:
            accept = True
        else:
            accept = click.confirm("  Accept this rule?", default=True)

        if accept:
            added = improver.accept(proposal)
            if added:
                console.print("  [green]✓ Rule added to constitution.[/]\n")
                accepted_count += 1
            else:
                console.print("  [dim]✗ Duplicate — already in constitution.[/]\n")
                skipped_count += 1
        else:
            console.print("  [dim]Skipped.[/]\n")
            skipped_count += 1

    if not dry_run:
        console.print(
            f"[bold]Done.[/] {accepted_count} rule(s) added, {skipped_count} skipped."
        )
        if accepted_count:
            console.print(
                f"[dim]Constitution updated: {constitution.path}[/]"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_source(program: str | None, filepath: str | None) -> str:
    if filepath:
        return Path(filepath).read_text(encoding="utf-8")
    if program:
        return program
    # Read from stdin if neither provided
    if not sys.stdin.isatty():
        return sys.stdin.read()
    console.print("[yellow]No program provided. Pass a program string or --file.[/]")
    sys.exit(1)


def _print_results(results: list[dict]) -> None:
    if not results:
        console.print("[dim]No results.[/]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Step", style="dim", width=4)
    table.add_column("Verb", style="bold")
    table.add_column("Target")
    table.add_column("Status")
    table.add_column("Output (preview)")
    table.add_column("ms", justify="right", style="dim")

    for i, r in enumerate(results, 1):
        status_color = {"ok": "green", "error": "red", "skipped": "yellow"}.get(
            r["status"], "white"
        )
        target_str = ".".join(r["target"]) if r["target"] else ""
        output_preview = str(r["output"])[:60] + ("…" if len(str(r["output"])) > 60 else "")
        table.add_row(
            str(i),
            r["verb"],
            target_str,
            f"[{status_color}]{r['status']}[/]",
            output_preview,
            str(r["duration_ms"]),
        )

    console.print(table)
