# LocalAgent

A local AI agent framework powered by [MLX](https://github.com/ml-explore/mlx) for running autonomous skills on Apple Silicon. Everything runs on-device -- no cloud APIs, no data leaving your machine.

## Overview

LocalAgent provides a plugin-based skill system where each skill is an autonomous agent that can be triggered manually or on a schedule. Skills interact with the local filesystem through a sandboxed API with strict permission enforcement.

The first built-in skill is the **File Organizer** -- an agent that scans your Desktop and Downloads folders, uses a local LLM to understand file contents and purpose, and organizes them into semantically meaningful categories.

See [GUARDRAILS.md](GUARDRAILS.md) for the full safety architecture.

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.12+
- MLX and mlx-lm (pulled in as dependencies via `pip install`)

## Installation

```bash
git clone <repo-url> && cd localagent
pip install -r requirements.txt
pip install -e ".[dev]"
```

On first run, the default model (`mlx-community/Llama-3.2-3B-Instruct-4bit`) will be downloaded from HuggingFace if not already cached.

## Quick Start

![LocalAgent CLI](docs/images/cli-overview.png)

```bash
# List available skills
localagent list

# Run the file organizer interactively (dry-run, then confirm)
localagent run file-organizer

# Run in auto mode (for scheduled use)
localagent run file-organizer --auto

# Undo the last run
localagent undo file-organizer

# Selectively undo specific moves
localagent undo file-organizer --interactive

# Schedule to run daily at noon
localagent schedule file-organizer

# Remove the schedule
localagent unschedule file-organizer
```

## How the File Organizer Works

1. **Scan** -- walks `~/Desktop` and `~/Downloads`, collecting file metadata and reading content previews of text-based files.
2. **Categorize** -- sends the file inventory to the local LLM, which proposes a taxonomy of categories tailored to your specific files (not a fixed list of presets).
3. **Review** -- in interactive mode, displays a table of proposed moves for you to approve or reject.
4. **Execute** -- moves files into category subfolders within their source directory. Every move is journaled for undo.

### Adaptive Taxonomy

The organizer does not use pre-defined categories. On first run, the LLM analyzes your actual files -- names, types, and content -- and generates categories that fit your scenario (e.g. "Receipts & Invoices", "Machine Learning Projects", "Meeting Notes"). On subsequent runs, it loads the existing taxonomy and extends it only when genuinely new file types appear.

The taxonomy is saved to `~/.config/localagent/file-organizer/taxonomy.yaml` and is fully editable:

```bash
# View the learned taxonomy
localagent taxonomy show

# Edit in your $EDITOR
localagent taxonomy edit

# Start fresh
localagent taxonomy reset
```

You can lock any category by adding `user_locked: true` in the YAML -- the LLM will never rename or remove locked categories.

## Configuration

Configuration lives at `~/.config/localagent/config.yaml`. A default is created on first use. Key settings:

```yaml
model:
  model_path: "mlx-community/Llama-3.2-3B-Instruct-4bit"
  max_tokens: 2048
  temperature: 0.3

skills:
  file-organizer:
    watch_directories:
      - "~/Desktop"
      - "~/Downloads"
    exclude_patterns:
      - ".DS_Store"
      - "*.crdownload"
      - "*.tmp"
    skip_hidden: true
    content_preview_bytes: 512
    move_warning_threshold: 50

triggers:
  file-organizer:
    type: cron
    schedule: "0 12 * * *"
```

To swap models, change `model_path` to any MLX-compatible model (e.g. a quantized Gemma, Phi, or Mistral).

## Safety

LocalAgent is designed with multiple layers of guardrails to prevent autonomous agents from causing harm. Skills operate in a sandbox with no ability to delete files, access directories outside their config, or bypass permission checks.

See [GUARDRAILS.md](GUARDRAILS.md) for the complete safety architecture.

## Project Structure

```
src/localagent/
├── cli.py                          # CLI entry point
├── config.py                       # Config loading and merging
├── core/
│   ├── engine.py                   # MLX model wrapper
│   ├── safefs.py                   # Sandboxed filesystem API
│   ├── skill.py                    # Skill ABC and manifest
│   ├── registry.py                 # Skill discovery and instantiation
│   └── triggers.py                 # Cron scheduling
└── skills/
    └── file_organizer/
        ├── scanner.py              # File inventory with content preview
        ├── categorizer.py          # LLM-powered adaptive taxonomy
        ├── mover.py                # Safe moves with undo journal
        └── skill.py                # FileOrganizerSkill
```

## Tests

```bash
pytest tests/ -v
```

53 tests covering SafeFS boundary enforcement, path traversal attacks, config merging, JSON extraction, scanner behavior, categorizer validation, and move/undo operations.

## License

MIT License. See [LICENSE](LICENSE).
