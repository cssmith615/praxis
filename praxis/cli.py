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
from praxis.providers import resolve_provider
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7822

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
@click.option("--provider", "-p", default=None,
              type=click.Choice(["anthropic", "openai", "ollama", "grok", "gemini"],
                                case_sensitive=False),
              help="LLM provider (default: auto-detect from env)")
@click.option("--model", default=None, help="Model override for the chosen provider")
def goal(
    goal: str,
    mode: str,
    db: str | None,
    no_execute: bool,
    no_store: bool,
    show_program: bool,
    provider: str | None,
    model: str | None,
):
    """
    Translate a natural language GOAL into a Praxis program, execute it,
    and store the result in program memory.

    Provider is auto-detected from environment variables:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GROK_API_KEY, GEMINI_API_KEY.
    Use --provider to override. Defaults to Ollama if no key is set.
    """
    memory = ProgramMemory(db_path=db)
    constitution = Constitution()
    try:
        llm_provider = resolve_provider(provider=provider, model=model)
    except Exception as exc:
        console.print(f"[bold red]Provider error:[/] {exc}")
        sys.exit(1)

    console.print(f"[dim]Provider: {llm_provider}[/]")
    planner = Planner(memory=memory, constitution=constitution,
                      provider=llm_provider, mode=mode)

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
@click.option("--llm", is_flag=True, help="Use an LLM to write rule text (see --provider)")
@click.option("--provider", "-p", default=None,
              type=click.Choice(["anthropic", "openai", "ollama", "grok", "gemini"],
                                case_sensitive=False),
              help="LLM provider for --llm mode (default: auto-detect)")
@click.option("--model", default=None, help="Model override for --llm mode")
@click.option("--yes", "-y", is_flag=True, help="Accept all proposals without prompting")
@click.option("--dry-run", is_flag=True, help="Show proposals but don't write anything")
def improve_cmd(
    log: str | None,
    const_path: str | None,
    llm: bool,
    provider: str | None,
    model: str | None,
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

    if llm:
        try:
            llm_provider = resolve_provider(provider=provider, model=model)
            console.print(f"[dim]LLM provider: {llm_provider}[/]")
        except Exception as exc:
            console.print(f"[bold red]Provider error:[/] {exc}")
            sys.exit(1)
    else:
        llm_provider = None

    improver = Improver(constitution=constitution, log_path=log,
                        use_llm=llm, provider=llm_provider)

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
# praxis serve
# ──────────────────────────────────────────────────────────────────────────────

@main.command("serve")
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Bind address")
@click.option("--port", default=DEFAULT_PORT, show_default=True, type=int, help="Port")
@click.option("--open", "open_browser", is_flag=True, help="Open browser on start")
def serve_cmd(host: str, port: int, open_browser: bool):
    """Start the Praxis web dashboard at http://localhost:7822"""
    url = f"http://{host}:{port}"
    console.print(f"\n[bold cyan]Praxis Dashboard[/]  {url}")
    console.print("[dim]  Tabs: Dashboard · Programs · Logs · Constitution · Schedules · Webhooks · Activity · Audit · Editor[/]")
    console.print("[dim]  Press Ctrl+C to stop.\n[/]")

    if open_browser:
        import threading, webbrowser, time
        def _open():
            time.sleep(0.8)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        from praxis.server import serve as _serve
        _serve(host=host, port=port)
    except ImportError as exc:
        console.print(f"[bold red]Missing dependency:[/] {exc}")
        console.print("[dim]Install with: pip install praxis-lang[bridge][/]")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# praxis search / install / publish  (Sprint 30 — program registry)
# ──────────────────────────────────────────────────────────────────────────────

@main.command("search")
@click.argument("query", required=False, default="")
@click.option("--registry", default=None, help="Registry index URL (overrides PRAXIS_REGISTRY_URL)")
def search_cmd(query: str, registry: str | None):
    """Search the Praxis program registry."""
    from praxis.registry import search_registry, RegistryError, REGISTRY_URL
    url = registry or REGISTRY_URL
    try:
        programs = search_registry(query, registry_url=url)
    except RegistryError as e:
        console.print(f"[red]Registry error:[/] {e}")
        sys.exit(1)

    if not programs:
        console.print("[yellow]No programs found.[/]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Tags", style="dim")
    table.add_column("Author", style="dim")
    for p in programs:
        table.add_row(p.name, p.description, ", ".join(p.tags), p.author)
    console.print(table)
    console.print(f"\n[dim]{len(programs)} program(s) found. Install with: praxis install <name>[/]")


@main.command("install")
@click.argument("name")
@click.option("--db", default=None, help="Path to program memory SQLite file")
@click.option("--registry", default=None, help="Registry index URL")
def install_cmd(name: str, db: str | None, registry: str | None):
    """Install a program from the Praxis registry into your program memory."""
    from praxis.registry import install_program, RegistryError, REGISTRY_URL
    from praxis.memory import ProgramMemory

    url = registry or REGISTRY_URL
    memory = ProgramMemory(db_path=db) if db else ProgramMemory()

    console.print(f"[cyan]Fetching '{name}' from registry…[/]")
    try:
        prog = install_program(name, memory=memory, registry_url=url)
    except RegistryError as e:
        console.print(f"[red]Install failed:[/] {e}")
        sys.exit(1)

    console.print(f"[green]✓[/] Installed [bold]{prog.name}[/] v{prog.version}")
    console.print(f"  [dim]{prog.description}[/]")
    console.print(f"  [dim]Now available in memory — run with: praxis goal \"{prog.description}\"[/]")


@main.command("publish")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--name", required=True, help="Registry name for this program (e.g. news-brief)")
@click.option("--description", required=True, help="One-line description")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--author", default="", help="Your GitHub username")
@click.option("--out", "outdir", default=".", help="Output directory for packaged files")
def publish_cmd(
    filepath: str,
    name: str,
    description: str,
    tags: str,
    author: str,
    outdir: str,
):
    """Package a program for submission to the Praxis registry."""
    from praxis.registry import publish_program
    from praxis.grammar import parse

    program_text = Path(filepath).read_text(encoding="utf-8")

    # Validate it parses
    try:
        parse(program_text)
    except Exception as exc:
        console.print(f"[red]Parse error:[/] {exc}")
        sys.exit(1)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    out_path = Path(outdir) / f"{name}.px"
    metadata = publish_program(
        program_text=program_text,
        name=name,
        description=description,
        tags=tag_list,
        author=author,
        output_path=out_path,
    )

    console.print(f"[green]✓[/] Packaged as [bold]{out_path}[/]")
    console.print(f"  Metadata: {out_path.with_suffix('.json')}")
    console.print(f"\n[dim]To publish, open a PR to github.com/cssmith615/praxis adding your entry to registry/index.json[/]")


# ──────────────────────────────────────────────────────────────────────────────
# praxis compile
# ──────────────────────────────────────────────────────────────────────────────

@main.command("compile")
@click.argument("program", required=False)
@click.option("--file", "-f", "filepath", type=click.Path(exists=True),
              help="Path to a .px file")
@click.option("--target", "-t", default="typescript",
              type=click.Choice(["typescript", "wasm"]), show_default=True,
              help="Compile target language")
@click.option("--out", "-o", "outfile", type=click.Path(),
              help="Output file path (default: stdout)")
@click.option("--embed-runtime", is_flag=True,
              help="Embed runtime stub in the output file")
@click.option("--runtime-import", default="./praxis-runtime", show_default=True,
              help="Import path for the Praxis runtime module")
def compile_cmd(
    program: str | None,
    filepath: str | None,
    target: str,
    outfile: str | None,
    embed_runtime: bool,
    runtime_import: str,
):
    """Compile a Praxis program to a target language."""
    source = _load_source(program, filepath)
    try:
        prog = parse(source)
    except Exception as exc:
        console.print(f"[bold red]Parse error:[/] {exc}")
        sys.exit(1)

    if target == "wasm":
        from praxis.codegen.wasm import WasmGenerator
        code = WasmGenerator().generate(prog, source_text=source)
        ext = ".wat"
    else:
        from praxis.codegen.typescript import TypeScriptGenerator, RUNTIME_STUB
        gen = TypeScriptGenerator(runtime_import=runtime_import, embed_runtime=embed_runtime)
        code = gen.generate(prog, source_text=source)
        ext = ".ts"

    if outfile:
        out_path = Path(outfile)
        out_path.write_text(code, encoding="utf-8")
        if target == "typescript" and not embed_runtime:
            runtime_out = out_path.parent / "praxis-runtime.ts"
            if not runtime_out.exists():
                from praxis.codegen.typescript import RUNTIME_STUB
                runtime_out.write_text(RUNTIME_STUB, encoding="utf-8")
                console.print(f"[dim]Runtime stub: {runtime_out}[/]")
        console.print(f"[bold green]Compiled → {out_path}[/]")
    else:
        click.echo(code)


# ──────────────────────────────────────────────────────────────────────────────
# praxis worker
# ──────────────────────────────────────────────────────────────────────────────

@main.command("worker")
@click.option("--port", "-p", default=7821, show_default=True, type=int,
              help="Port to serve this worker's bridge on")
@click.option("--host", default="0.0.0.0", show_default=True,
              help="Bind address for this worker's bridge")
@click.option("--id", "worker_id", default=None,
              help="Worker agent_id (default: hostname-<port>)")
@click.option("--role", default="worker", show_default=True,
              help="Worker role label")
@click.option("--verbs", default=None,
              help="Comma-separated list of verbs this worker handles (default: all)")
@click.option("--hub", "hub_url", default=None,
              help="Hub URL to register with (e.g. http://hub:7821). Optional.")
@click.option("--advertise", default=None,
              help="URL that the hub should use to reach this worker. "
                   "Default: http://<host>:<port>")
@click.option("--heartbeat", default=30, show_default=True, type=int,
              help="Seconds between heartbeat pings to the hub (0 = disable)")
def worker_cmd(
    port: int,
    host: str,
    worker_id: str | None,
    role: str,
    verbs: str | None,
    hub_url: str | None,
    advertise: str | None,
    heartbeat: int,
):
    """
    Start a Praxis worker process.

    The worker serves a local Praxis bridge (POST /execute) and optionally
    registers itself with a hub so coordinators can discover it.

    Examples:
        praxis worker --port 7823 --hub http://hub:7821 --role data --verbs ING,CLN,XFRM
        praxis worker --port 7824 --id analysis --role analysis --verbs SUMM,EVAL,GEN
    """
    import socket
    import threading

    from praxis.distributed import WorkerClient

    hostname = socket.gethostname()
    wid      = worker_id or f"{hostname}-{port}"
    verb_list = [v.strip().upper() for v in verbs.split(",")] if verbs else []
    worker_url = advertise or f"http://{hostname}:{port}"

    console.print(f"\n[bold cyan]Praxis Worker[/]  id=[bold]{wid}[/]  port={port}")
    console.print(f"  role=[bold]{role}[/]  verbs={verb_list or 'all'}")
    if hub_url:
        console.print(f"  hub=[dim]{hub_url}[/]  advertise=[dim]{worker_url}[/]")
    console.print("[dim]  Press Ctrl+C to stop.\n[/]")

    # ── Register with hub ────────────────────────────────────────────────────
    client: WorkerClient | None = None
    if hub_url:
        client = WorkerClient(hub_url)
        ok = client.register(wid, role, verb_list, worker_url)
        if ok:
            console.print(f"[green]✓[/] Registered with hub at {hub_url}")
        else:
            console.print(f"[yellow]⚠[/] Could not register with hub — continuing anyway")

    # ── Background heartbeat ─────────────────────────────────────────────────
    _stop_event = threading.Event()

    def _heartbeat_loop() -> None:
        while not _stop_event.wait(heartbeat):
            if client:
                client.heartbeat(wid)

    if client and heartbeat > 0:
        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb_thread.start()

    # ── Serve bridge ─────────────────────────────────────────────────────────
    try:
        import uvicorn
        from praxis.bridge import app
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError as exc:
        console.print(f"[bold red]Missing dependency:[/] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        if client:
            client.deregister(wid)
            console.print(f"[dim]Deregistered {wid} from hub.[/]")


# ──────────────────────────────────────────────────────────────────────────────
# praxis chat
# ──────────────────────────────────────────────────────────────────────────────

@main.command("chat")
@click.option("--mode", "-m", default="dev", type=click.Choice(["dev", "prod"]),
              show_default=True, help="Execution mode")
@click.option("--db", default=None, help="Path to program memory SQLite file")
@click.option("--provider", "-p", default=None,
              type=click.Choice(["anthropic", "openai", "ollama", "grok", "gemini"],
                                case_sensitive=False),
              help="LLM provider for goal mode (default: auto-detect from env)")
@click.option("--model", default=None, help="Model override for the chosen provider")
def chat_cmd(mode: str, db: str | None, provider: str | None, model: str | None):
    """
    Start the interactive Praxis REPL.

    Type .px programs directly or natural-language goals (requires an LLM
    provider).  Session commands start with : — type :help for the full list.
    """
    from praxis.chat import PraxisREPL
    from praxis.memory import ProgramMemory

    memory = ProgramMemory(db_path=db)

    llm_provider = None
    try:
        llm_provider = resolve_provider(provider=provider, model=model)
    except Exception:
        pass   # goal mode simply disabled; not a fatal error in chat

    repl = PraxisREPL(memory=memory, provider=llm_provider, mode=mode)
    repl.run()


# ──────────────────────────────────────────────────────────────────────────────
# praxis agent
# ──────────────────────────────────────────────────────────────────────────────

@main.command("agent")
@click.option("--token", envvar="TELEGRAM_BOT_TOKEN",
              help="Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)")
@click.option("--chat-id", "chat_ids", multiple=True, envvar="TELEGRAM_CHAT_IDS",
              help="Allowed Telegram chat id(s). Can be repeated. "
                   "Env: TELEGRAM_CHAT_IDS (comma-separated). "
                   "Leave empty to allow any chat (not recommended).")
@click.option("--trigger", default=None, envvar="AGENT_TRIGGER",
              help="Trigger word prefix for group chats (e.g. '@praxis'). "
                   "Env: AGENT_TRIGGER.")
@click.option("--mode", "-m", default="dev", type=click.Choice(["dev", "prod"]),
              show_default=True, help="Praxis execution mode")
@click.option("--provider", "-p", default=None,
              type=click.Choice(["anthropic", "openai", "ollama", "grok", "gemini"],
                                case_sensitive=False),
              help="LLM provider for goal planning (default: auto-detect from env)")
@click.option("--model", default=None, help="Model override for the chosen provider")
@click.option("--agent-model", default="claude-sonnet-4-6", show_default=True,
              help="Claude model for the agent conversation loop")
@click.option("--fast-model", default=None, envvar="PRAXIS_FAST_MODEL",
              help="Model for simple requests (multi-tier routing). "
                   "Default: claude-haiku-4-5-20251001. "
                   "Set to 'off' to disable routing. Env: PRAXIS_FAST_MODEL")
@click.option("--schedule/--no-schedule", default=False,
              help="Enable the Scheduler background thread")
@click.option("--memory/--no-memory", "use_memory", default=False,
              help="Enable ProgramMemory (requires sentence-transformers)")
@click.option("--db", default=None, help="Custom database directory path")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY",
              help="Anthropic API key. Env: ANTHROPIC_API_KEY")
def agent_cmd(
    token: str | None,
    chat_ids: tuple[str, ...],
    trigger: str | None,
    mode: str,
    provider: str | None,
    model: str | None,
    agent_model: str,
    fast_model: str | None,
    schedule: bool,
    use_memory: bool,
    db: str | None,
    api_key: str | None,
):
    """
    Start the Praxis Agent — a native AI agent with direct .px execution.

    The agent listens on Telegram (or another channel) and can run programs,
    validate syntax, plan goals, schedule tasks, and search program memory —
    all without any translation layer.

    Quick start:

    \b
        export ANTHROPIC_API_KEY=sk-ant-...
        export TELEGRAM_BOT_TOKEN=your-token
        export TELEGRAM_CHAT_IDS=123456789
        praxis agent

    Run with Docker:

    \b
        docker compose -f praxis/agent/docker-compose.yml up -d
    """
    if not token:
        console.print(
            "[bold red]No Telegram token.[/] "
            "Pass --token or set TELEGRAM_BOT_TOKEN."
        )
        sys.exit(1)

    from praxis.agent.core import PraxisAgent
    from praxis.agent.runner import AgentRunner
    from praxis.agent.channels.telegram import TelegramChannel

    # Handle comma-separated chat ids from env var
    allowed: set[str] = set()
    for cid in chat_ids:
        # env var may be comma-separated, click may pass as single string
        for part in cid.split(","):
            part = part.strip()
            if part:
                allowed.add(part)

    channel = TelegramChannel(
        token=token,
        allowed_chat_ids=allowed or None,
        trigger_word=trigger,
    )

    router_enabled = fast_model != "off"
    agent = PraxisAgent(
        model=agent_model,
        api_key=api_key,
        fast_model=fast_model if router_enabled else None,
        router_enabled=router_enabled,
    )

    runner = AgentRunner(
        agent=agent,
        channel=channel,
        mode=mode,
        provider=provider,
        model=model,
        enable_scheduler=schedule,
        enable_memory=use_memory,
        db_path=db,
    )

    fast_display = fast_model or "claude-haiku-4-5-20251001"
    routing_display = f"off" if not router_enabled else f"fast={fast_display}"
    console.print(
        f"\n[bold cyan]Praxis Agent[/]  model=[bold]{agent_model}[/]  "
        f"routing=[bold]{routing_display}[/]  mode={mode}"
    )
    if allowed:
        console.print(f"  chat whitelist: {sorted(allowed)}")
    else:
        console.print("  [yellow]⚠  No chat whitelist — all chats accepted[/]")
    if trigger:
        console.print(f"  trigger word: [bold]{trigger}[/]")
    console.print("[dim]  Press Ctrl+C to stop.\n[/]")

    runner.run()


@main.command("mcp")
def mcp_cmd():
    """
    Start the Praxis MCP server (stdio transport).

    Exposes Praxis tools to Claude Code, Cursor, Zed, and any MCP-aware host.

    Add to Claude Code settings.json:

    \b
      {
        "mcpServers": {
          "praxis": {
            "command": "praxis",
            "args": ["mcp"]
          }
        }
      }

    Tools: run_program, validate_program, plan_goal, recall_similar,
           search_registry, install_program, get_constitution

    Resources: praxis://constitution, praxis://programs
    """
    from praxis.mcp_server import run_server
    run_server()


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
