# CLAUDE.md

Development guide for working on LocalAgent with Claude Code.

## Project Summary

LocalAgent is a local AI agent framework powered by MLX. Skills are autonomous agents that run on-device using local LLMs. The first skill is a file organizer that categorizes files in watched directories using LLM-generated taxonomy.

## Build and Test

```bash
pip install -e ".[dev]"    # Install in dev mode
pytest tests/ -v           # Run tests
localagent list            # Verify CLI works
localagent eval file-organizer  # Run skill evals against default model
```

## Architecture

```
src/localagent/
├── cli.py              # argparse CLI, dispatches to command handlers
├── config.py           # YAML config loading with deep merge
├── core/
│   ├── engine.py       # MLX model wrapper (lazy load, chat template, JSON extraction)
│   ├── safefs.py       # Sandboxed filesystem -- THE critical safety layer
│   ├── skill.py        # Skill ABC + SkillManifest
│   ├── registry.py     # Skill registration and instantiation
│   └── triggers.py     # Cron trigger management
└── skills/
    └── file_organizer/ # First built-in skill
```

### Key Interfaces

**Adding a new skill** requires:
1. Subclass `Skill` from `localagent.core.skill`
2. Define a `SkillManifest` with name, description, and required permissions
3. Implement `plan(engine)`, `execute(actions)`, and `undo(log_path)`
4. Register in `localagent.core.registry.register_builtin_skills()`

**Engine** (`core/engine.py`): Call `generate_text()` for raw output or `generate_json()` for parsed JSON with retry logic. The engine handles chat template formatting and code fence extraction automatically.

**SafeFS** (`core/safefs.py`): Skills receive a `SafeFS` instance scoped to their allowed directories. Use `list_dir()`, `read_file()`, `stat()`, `move_file()`, `make_dir()`. There is no delete operation.

## Guardrails -- Read Before Modifying

**Read [GUARDRAILS.md](GUARDRAILS.md) before making changes to the safety layer.**

Critical invariants that must never be broken:

1. **SafeFS has no delete method.** Do not add one. File deletion is structurally impossible by design.
2. **SafeFS resolves all paths before boundary checks.** Symlinks and `..` are resolved via `Path.resolve()` before checking against `allowed_paths`. Do not skip this step.
3. **SafeFS never overwrites.** `move_file()` appends a timestamp on collision. Do not add an overwrite flag.
4. **Skills never import `os`, `shutil`, or `pathlib` for file operations.** All filesystem access goes through the `SafeFS` instance provided by the framework. If a new skill needs filesystem access, it must declare it in its manifest and use the provided `SafeFS`.
5. **LLM output is data, not code.** JSON responses are parsed with `json.loads()`. No `eval()`, no `exec()`, no shell execution of LLM output. Ever.
6. **Auto mode aborts on any error.** When `--auto` is set (cron runs), any failure must abort the entire run. No partial execution.

## Adding a New Skill

Example skeleton for a new skill:

```python
# src/localagent/skills/my_skill/skill.py

from localagent.core.skill import Action, Report, Skill, SkillManifest
from localagent.core.engine import Engine

class MySkill(Skill):
    manifest = SkillManifest(
        name="my-skill",
        description="What this skill does",
        permissions={"read"},  # only request what you need
    )

    def plan(self, engine: Engine) -> list[Action]:
        # Use self.safefs for filesystem access
        # Use engine.generate_json() for LLM queries
        # Use self.state_dir for persistent state
        # Use self._config for skill-specific config
        return [Action(action_type="...", detail="...")]

    def execute(self, actions: list[Action]) -> Report:
        # Carry out the planned actions
        return Report(skill_name=self.manifest.name, actions_performed=len(actions))

    def undo(self, log_path, *, interactive=False) -> Report:
        # Reverse previous execution
        return Report(skill_name=self.manifest.name, actions_performed=0)
```

Then register it in `src/localagent/core/registry.py`:

```python
def register_builtin_skills() -> None:
    from localagent.skills.file_organizer.skill import FileOrganizerSkill
    from localagent.skills.my_skill.skill import MySkill

    _registry.register(FileOrganizerSkill)
    _registry.register(MySkill)
```

Add a config section in `config/default.yaml` under `skills:` with the skill's `config_key`.

## Adding a New Trigger Type

The trigger system is in `core/triggers.py`. Currently only `CronTrigger` exists. To add a new trigger type (e.g. filesystem watcher, webhook):

1. Create a new class alongside `CronTrigger`
2. Add an `install()` / `uninstall()` interface
3. Wire it into the CLI in `cli.py`
4. Add a `type` discriminator in the config's `triggers:` section

## Config and State Paths

| Path | Purpose |
|---|---|
| `~/.config/localagent/config.yaml` | User configuration |
| `~/.config/localagent/<skill>/` | Per-skill learned state (e.g. taxonomy) |
| `~/.local/share/localagent/logs/` | Undo journals |
| `config/default.yaml` | Shipped defaults (merged under user config) |

## Design Principles

These are lessons learned during development. Follow them for future changes.

- **README stays generic.** Do not embed specific counts, scenario names, source file paths, or other details that change frequently (e.g. "ships with 5 scenarios testing X, Y, Z" or "See the [eval scenarios source](path/to/file)"). Let the code be the source of truth. Benchmark results with concrete numbers are fine since they represent a point-in-time snapshot.
- **Evals are not tests.** Evals run against a real LLM via `localagent eval`. Do not write pytest tests for eval scenarios — no test file should import from or assert against eval scenario classes. When evals change (new scenarios, updated scoring), tests should not need updating.
- **Skill-specific concepts stay out of the top-level CLI.** The CLI should only expose framework-level commands (`run`, `undo`, `schedule`, `config`, `eval`, `list`). If a concept belongs to a single skill (e.g. taxonomy is file-organizer's learned state), it should not be a top-level subcommand. Users can manage skill state files directly.
- **Clean up dead imports.** When removing a feature, also remove any imports that become unused (`os`, `subprocess`, etc.).

## Code Style

- Type hints on all function signatures
- `from __future__ import annotations` in every module
- Logging via `logging.getLogger(__name__)`
- Rich for terminal output (tables, prompts, formatted text)
- Tests use `pytest` with `tmp_path` fixtures and mocked LLM calls

## Model Compatibility

The engine is model-agnostic. Any model supported by `mlx-lm` works -- set `model.model_path` in config. The engine uses `tokenizer.apply_chat_template()` for prompt formatting, so instruction-tuned models with chat templates work best.

Tested with:
- `mlx-community/Llama-3.2-3B-Instruct-4bit` (default)
