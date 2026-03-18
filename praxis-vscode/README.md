# Praxis Language — VS Code Extension

Syntax highlighting, inline validation, and run commands for [Praxis](https://github.com/cssmith615/praxis) `.px` workflow files.

## Features

- **Syntax highlighting** — verbs, targets, parameters, `$variables`, keywords, and comments
- **Inline validation** — red squiggles on save for invalid programs (runs `praxis validate` in the background)
- **Run command** — execute the current `.px` file from the editor title bar or right-click menu
- **Auto-close** — brackets, parentheses, and string delimiters

## Requirements

Install the Praxis CLI:

```bash
pip install praxis-lang
```

Verify it's on your PATH:

```bash
praxis validate "LOG.hello"
# → ✓ Program is valid.
```

## Usage

Open any `.px` file. Syntax highlighting activates automatically.

**Validate** — right-click → `Praxis: Validate File`, or save the file (auto-validates)

**Run** — click the ▶ button in the editor title bar, or right-click → `Praxis: Run File`. Opens an integrated terminal and runs `praxis run <file>`.

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `praxis.executablePath` | `praxis` | Path to the praxis CLI (if not on PATH) |
| `praxis.validateOnSave` | `true` | Validate on every save |
| `praxis.mode` | `dev` | Execution mode: `dev` or `prod` |

## Example `.px` file

```
GOAL:hn_briefing

FETCH.data(src="https://hacker-news.firebaseio.com/v0/topstories.json") ->
XFRM.slice(limit=5) ->
FETCH.data(src="https://hacker-news.firebaseio.com/v0/item/$item.json") ->
XFRM.pluck(field=title) ->
XFRM.join(sep="\n") ->
OUT.telegram
```

## More information

- [Praxis on GitHub](https://github.com/cssmith615/praxis)
- [Full documentation](https://github.com/cssmith615/praxis#readme)
- [Report issues](https://github.com/cssmith615/praxis/issues)
