"""
Praxis Interactive REPL — `praxis chat`

A conversational interface for building and running Praxis programs.

Modes
-----
  Program mode  — input that looks like a .px program is parsed, validated,
                  and optionally run inline.
  Goal mode     — natural-language input is sent to the planner (requires an
                  LLM provider).  Falls back to program mode if no provider.

Session commands (prefix with :)
---------------------------------
  :run          — execute the current program buffer
  :validate     — validate without executing
  :show         — print the current buffer
  :clear        — clear the buffer
  :save <file>  — save the current buffer to a .px file
  :history      — show recent programs from memory
  :mode         — toggle between goal/program modes
  :help         — show this help
  :quit / :exit — exit the REPL

Multi-line
----------
  End a line with \\ to continue on the next line.
  Blank line flushes the buffer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.syntax import Syntax
from rich.rule import Rule
from rich import box
from rich.table import Table

from praxis.grammar import parse
from praxis.validator import validate as validate_program
from praxis.executor import Executor
from praxis.handlers import HANDLERS
from praxis.planner import Planner, PlanningFailure
from praxis.constitution import Constitution

console = Console()

# ── heuristic: is this input a .px program or a natural-language goal? ────────

_PROGRAM_HINTS = re.compile(
    r"^[A-Z]{2,8}\.[a-zA-Z]"   # VERB.target at start
    r"|->|PAR\s*\(|LOOP\s*\("  # chain arrow, PAR block, LOOP block
    r"|IF\s+\S+\s+=="           # condition
    r"|^\s*#",                  # comment line
    re.MULTILINE,
)


def _looks_like_program(text: str) -> bool:
    return bool(_PROGRAM_HINTS.search(text.strip()))


# ── REPL ──────────────────────────────────────────────────────────────────────

class PraxisREPL:
    """
    Interactive REPL session.

    Parameters
    ----------
    memory : ProgramMemory | None
        Program library for storing/retrieving results.
    provider : Provider | None
        LLM provider for goal mode.  If None, goal mode is unavailable.
    mode : str
        Execution mode — "dev" or "prod".
    """

    PROMPT        = "[bold cyan]praxis>[/] "
    CONTINUE_PROMPT = "[dim]      …[/] "

    def __init__(self, memory=None, provider=None, mode: str = "dev") -> None:
        self.memory   = memory
        self.provider = provider
        self.mode     = mode
        self._goal_mode = provider is not None  # auto-enable if provider available
        self._buffer: list[str] = []
        self._history: list[str] = []  # session history of executed programs

    # ── Entry ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the REPL loop."""
        self._print_banner()

        # Enable readline history if available
        try:
            import readline  # noqa: F401
        except ImportError:
            pass

        while True:
            try:
                prompt = self.CONTINUE_PROMPT if self._buffer else self.PROMPT
                raw = console.input(prompt)
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/]")
                break

            line = raw.rstrip()

            # ── Session commands ──────────────────────────────────────────────
            if line.startswith(":"):
                self._handle_command(line)
                continue

            # ── Multi-line continuation ───────────────────────────────────────
            if line.endswith("\\"):
                self._buffer.append(line[:-1])
                continue

            # ── Flush on blank line with pending buffer ───────────────────────
            if not line and self._buffer:
                self._dispatch("\n".join(self._buffer))
                self._buffer.clear()
                continue

            if not line:
                continue

            # ── Single-line input (or last line of multi-line) ────────────────
            if self._buffer:
                self._buffer.append(line)
                full = "\n".join(self._buffer)
                self._buffer.clear()
                self._dispatch(full)
            else:
                self._dispatch(line)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if _looks_like_program(text):
            self._run_program(text, auto=True)
        elif self._goal_mode and self.provider is not None:
            self._run_goal(text)
        else:
            # Try as program anyway; if it fails, hint about goal mode
            if not self._run_program(text, auto=True, silent_hint=True):
                if self.provider is None:
                    console.print(
                        "[yellow]Looks like a goal but no LLM provider is configured.[/]\n"
                        "[dim]Set ANTHROPIC_API_KEY (or OPENAI_API_KEY, etc.) and restart chat.[/]\n"
                        "[dim]Or write a .px program: VERB.target -> VERB2.target[/]"
                    )
                else:
                    console.print("[dim]Could not parse as program. Try :mode to switch to goal mode.[/]")

    # ── Program execution ─────────────────────────────────────────────────────

    def _run_program(
        self, source: str, auto: bool = False, silent_hint: bool = False
    ) -> bool:
        """
        Parse, validate, and (if auto=True) run a program.
        Returns True if parse succeeded, False on parse/validation error.
        """
        # Parse
        try:
            ast = parse(source)
        except Exception as exc:
            if not silent_hint:
                console.print(f"[bold red]Parse error:[/] {exc}")
            return False

        # Validate
        errors = validate_program(ast, mode=self.mode)
        if errors:
            console.print("[bold red]Validation errors:[/]")
            for e in errors:
                console.print(f"  [red]•[/] {e}")
            return True  # parse succeeded even though validation failed

        console.print("[bold green]✓[/] Valid")

        if not auto:
            return True

        # Ask to run (unless it's already committed)
        console.print()
        _show_program(source)
        console.print()

        try:
            run = console.input("[dim]Run? [Y/n]:[/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return True

        if run in ("", "y", "yes"):
            self._execute_and_show(source, ast)

        return True

    def _execute_and_show(self, source: str, ast: Any = None) -> None:
        if ast is None:
            try:
                ast = parse(source)
            except Exception as exc:
                console.print(f"[bold red]Parse error:[/] {exc}")
                return

        try:
            executor = Executor(HANDLERS, mode=self.mode)
            results = executor.execute(ast)
        except Exception as exc:
            console.print(f"[bold red]Execution error:[/] {exc}")
            return

        _print_results(results)
        self._history.append(source)

        # Store in memory if available
        if self.memory is not None:
            outcome = "success" if all(r["status"] != "error" for r in results) else "partial"
            pid = self.memory.store(source, source, outcome, results)
            console.print(f"[dim]Stored in memory as {pid[:8]}…[/]")

    # ── Goal planning ─────────────────────────────────────────────────────────

    def _run_goal(self, goal: str) -> None:
        constitution = Constitution()
        planner = Planner(
            memory=self.memory,
            constitution=constitution,
            provider=self.provider,
            mode=self.mode,
        )

        console.print(f"\n[bold cyan]Planning:[/] {goal}")
        if self.memory and self.memory.count() > 0:
            console.print(f"[dim]  Memory: {self.memory.count()} stored programs[/]")

        try:
            result = planner.plan(goal)
        except PlanningFailure as exc:
            console.print(f"[bold red]Planning failed after {exc.attempts} attempts:[/] {exc.last_error}")
            return
        except Exception as exc:
            console.print(f"[bold red]Planning error:[/] {exc}")
            return

        adapt_label = "[green]adapted[/]" if result.adapted else "[blue]generated[/]"
        console.print(f"[bold green]✓[/] Program {adapt_label} in {result.attempts} attempt(s)")

        if result.similar:
            top = result.similar[0]
            console.print(
                f"[dim]  Top match: {top.similarity:.2f} — \"{top.goal_text[:55]}\"[/]"
            )

        console.print()
        _show_program(result.program)
        console.print()

        try:
            run = console.input("[dim]Run? [Y/n]:[/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if run in ("", "y", "yes"):
            try:
                ast = parse(result.program)
            except Exception as exc:
                console.print(f"[bold red]Parse error:[/] {exc}")
                return
            self._execute_and_show(result.program, ast)

            if self.memory is not None:
                # Update stored entry with final outcome (already stored above
                # in _execute_and_show; store goal mapping here as well)
                pass
        else:
            if self.memory is not None:
                pid = self.memory.store(goal, result.program, "planned", [])
                console.print(f"[dim]Stored as {pid[:8]}… (not executed)[/]")

    # ── Commands ──────────────────────────────────────────────────────────────

    def _handle_command(self, line: str) -> None:
        parts = line[1:].split(None, 1)
        cmd   = parts[0].lower() if parts else ""
        arg   = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Bye.[/]")
            sys.exit(0)

        elif cmd == "help":
            self._print_help()

        elif cmd == "show":
            buf = "\n".join(self._buffer)
            if buf.strip():
                _show_program(buf)
            else:
                console.print("[dim]Buffer is empty.[/]")

        elif cmd == "clear":
            self._buffer.clear()
            console.print("[dim]Buffer cleared.[/]")

        elif cmd == "run":
            buf = "\n".join(self._buffer)
            if not buf.strip():
                console.print("[yellow]Buffer is empty. Type a program first.[/]")
            else:
                self._execute_and_show(buf)
                self._buffer.clear()

        elif cmd == "validate":
            buf = "\n".join(self._buffer)
            if not buf.strip():
                console.print("[yellow]Buffer is empty.[/]")
            else:
                self._run_program(buf, auto=False)

        elif cmd == "save":
            path = arg.strip() or "program.px"
            buf = "\n".join(self._buffer)
            if not buf.strip():
                console.print("[yellow]Buffer is empty.[/]")
            else:
                Path(path).write_text(buf, encoding="utf-8")
                console.print(f"[green]✓[/] Saved to {path}")

        elif cmd == "history":
            self._print_history()

        elif cmd == "mode":
            if self.provider is None:
                console.print("[yellow]No provider configured — goal mode unavailable.[/]")
            else:
                self._goal_mode = not self._goal_mode
                label = "goal" if self._goal_mode else "program"
                console.print(f"[dim]Switched to [bold]{label}[/] mode.[/]")

        else:
            console.print(f"[yellow]Unknown command: :{cmd}  — type :help[/]")

    # ── History ───────────────────────────────────────────────────────────────

    def _print_history(self) -> None:
        if self.memory is not None:
            programs = self.memory.recent(10)
        else:
            programs = []

        if not programs and not self._history:
            console.print("[dim]No history yet.[/]")
            return

        if programs:
            console.print(f"\n[bold]Program Memory[/] — last {len(programs)} entries\n")
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            table.add_column("ID", style="dim", width=10)
            table.add_column("Goal / Source")
            table.add_column("Outcome")
            table.add_column("Stored")
            for p in programs:
                color = {"success": "green", "failure": "red", "partial": "yellow"}.get(
                    p.outcome, "white"
                )
                preview = p.goal_text[:55] + ("…" if len(p.goal_text) > 55 else "")
                table.add_row(
                    p.id[:8] + "…",
                    preview,
                    f"[{color}]{p.outcome}[/]",
                    p.created_at[:19],
                )
            console.print(table)
        else:
            console.print(f"\n[dim]Session history ({len(self._history)} programs):[/]")
            for i, prog in enumerate(self._history[-10:], 1):
                preview = prog.replace("\n", " ")[:70]
                console.print(f"  [dim]{i}.[/] {preview}")
            console.print()

    # ── Banner / help ─────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        console.print()
        console.print(Rule("[bold cyan]Praxis REPL[/]"))
        mode_label   = f"[bold]{self.mode}[/]"
        goal_label   = "[green]enabled[/]" if self._goal_mode else "[dim]disabled (no provider)[/]"
        memory_label = f"[dim]{self.memory.count()} stored programs[/]" if self.memory else "[dim]no memory[/]"
        console.print(f"  mode={mode_label}  goal-mode={goal_label}  memory={memory_label}")
        console.print("  Type a [bold].px[/] program or a natural-language [bold]goal[/].")
        console.print("  [dim]:help for commands  :quit to exit[/]")
        console.print(Rule())
        console.print()

    def _print_help(self) -> None:
        console.print()
        console.print("[bold cyan]Praxis REPL — Commands[/]\n")
        rows = [
            (":run",          "Execute the current buffer"),
            (":validate",     "Validate without executing"),
            (":show",         "Print the current buffer"),
            (":clear",        "Clear the buffer"),
            (":save <file>",  "Save buffer to a .px file (default: program.px)"),
            (":history",      "Show recent programs from memory"),
            (":mode",         "Toggle goal/program input mode"),
            (":help",         "Show this help"),
            (":quit / :exit", "Exit the REPL"),
        ]
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Command", style="bold cyan", width=22)
        table.add_column("Description")
        for cmd, desc in rows:
            table.add_row(cmd, desc)
        console.print(table)
        console.print("[dim]Multi-line: end a line with \\ to continue. Blank line flushes.[/]\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _show_program(source: str) -> None:
    syntax = Syntax(source, "text", theme="monokai", line_numbers=True)
    console.print(syntax)


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
        color = {"ok": "green", "error": "red", "skipped": "yellow"}.get(r["status"], "white")
        target_str = ".".join(r["target"]) if r["target"] else ""
        preview = str(r["output"])[:55] + ("…" if len(str(r["output"])) > 55 else "")
        table.add_row(
            str(i), r["verb"], target_str,
            f"[{color}]{r['status']}[/]",
            preview, str(r["duration_ms"]),
        )
    console.print(table)
