# Guardrails

This document describes the safety architecture that constrains what LocalAgent skills can do. The goal is to make harmful outcomes structurally impossible rather than relying on the LLM to behave correctly.

## Design Principle

**Don't trust the LLM. Trust the sandbox.**

The LLM proposes actions. The framework validates and constrains them. Every layer assumes the LLM output could be wrong, hallucinated, or adversarial.

## Safety Layers

### 1. SafeFS -- Sandboxed Filesystem

Skills never get raw `os` or `shutil` access. They receive a `SafeFS` instance that enforces boundaries at the API level.

**What SafeFS enforces:**

| Rule | How |
|---|---|
| Path boundaries | Every path is resolved (symlinks, `..`) and checked against `allowed_paths` before any operation. `SafeFSError` is raised on violation. |
| No deletion | `SafeFS` has no `delete()`, `remove()`, `unlink()`, `rmdir()`, or `rmtree()` method. Deletion is structurally impossible. |
| No overwrite | `move_file()` appends a timestamp suffix if the destination already exists. Files are never silently replaced. |
| Permission gating | Each operation checks the granted permission set (`read`, `move`). A read-only SafeFS cannot move files. |
| Symlink traversal | Symlinks are resolved to their real path before boundary checks. A symlink pointing outside the sandbox is blocked. |
| Path traversal | `..` components are resolved before checking. `allowed_dir/../forbidden_dir/secret.txt` is caught. |

**Source:** `src/localagent/core/safefs.py`

### 2. Skill Manifest -- Declared Permissions

Every skill must declare a `SkillManifest` specifying:

- `name` -- unique identifier
- `permissions` -- the set of filesystem capabilities it needs (`read`, `move`)
- `config_key` -- which config section it reads

The framework validates the manifest at registration time and constructs a `SafeFS` scoped to exactly the directories and permissions declared. A skill cannot request capabilities at runtime that were not in its manifest.

```python
manifest = SkillManifest(
    name="file-organizer",
    description="Organizes files into smart, LLM-generated categories",
    permissions={"read", "move"},
)
```

**Source:** `src/localagent/core/skill.py`

### 3. LLM Output Validation

The categorizer validates every piece of the LLM's JSON response:

- **Hallucinated filenames** -- if the LLM assigns a file that doesn't exist in the scanned inventory, the assignment is silently dropped with a warning log.
- **Unknown categories** -- if a file is assigned to a category not present in the taxonomy, it is dropped.
- **JSON parse failures** -- if the LLM returns invalid JSON, the engine retries once with a correction prompt. If it fails again, the entire run aborts.
- **Batch isolation** -- large file sets are batched. A bad response in one batch doesn't corrupt others.

**Source:** `src/localagent/skills/file_organizer/categorizer.py` (`_validate_response`)

### 4. Failure Mode -- Abort, Don't Guess

In `--auto` mode (cron-triggered runs):

- If the LLM returns unparseable output: **abort entirely**. No partial execution.
- If SafeFS raises a permission error: **abort entirely**.
- If any unexpected exception occurs: **abort entirely**, log the error, exit non-zero.

The system never partially executes a plan when running unattended. Either all proposed moves succeed validation, or none are attempted.

**Source:** `src/localagent/cli.py` (`cmd_run`, auto mode path)

### 5. Move Volume Warning

A configurable threshold (`move_warning_threshold`, default 50) logs a warning when a single run proposes an unusually high number of moves. This catches cases where the LLM produces an unexpectedly broad plan.

The warning is logged but does not block execution (soft warning), as agreed during design.

**Source:** `src/localagent/skills/file_organizer/mover.py` (`execute_moves`)

### 6. Undo Journal

Every file move is journaled to a JSONL file **before** the move is executed. This means:

- A crash mid-run leaves a complete record of what was attempted.
- `localagent undo file-organizer` reverses all moves from the last journal.
- `localagent undo file-organizer --interactive` lets you pick which moves to reverse.

Undo operations also go through SafeFS, so they respect the same path boundaries.

Journal location: `~/.local/share/localagent/logs/file-organizer-YYYY-MM-DD.jsonl`

**Source:** `src/localagent/skills/file_organizer/mover.py` (`MoveRecord`, `undo_moves`)

### 7. User Taxonomy Control

The LLM generates categories, but the user has final authority:

- The taxonomy is saved as a plain YAML file at `~/.config/localagent/file-organizer/taxonomy.yaml`.
- Users can edit it freely (`localagent taxonomy edit`).
- Setting `user_locked: true` on any category prevents the LLM from modifying or removing it.
- `localagent taxonomy reset` wipes the learned state for a fresh start.

### 8. Interactive Confirmation

In interactive mode (the default), the skill:

1. Scans files and runs the LLM categorization.
2. Displays a table of proposed categories and moves.
3. Prompts `Proceed with these moves? [y/N]`.
4. Only executes if the user explicitly confirms.

No files are touched until the user says yes.

## What the Framework Cannot Do

These are structural constraints, not policy:

- **Delete files** -- no API exists for deletion in SafeFS.
- **Overwrite files** -- collision handling is mandatory and built into `move_file()`.
- **Access arbitrary directories** -- paths are checked against the config's `watch_directories` list.
- **Bypass permission checks** -- SafeFS methods call `_require()` and `_check_inside()` before every operation.
- **Execute code from LLM output** -- the LLM's response is parsed as JSON data. No `eval()`, no shell execution, no code interpretation.

## Test Coverage

Safety-critical behavior is covered by dedicated tests in `tests/test_safefs.py`:

- `test_read_forbidden_file_raises` -- accessing files outside the sandbox
- `test_path_traversal_blocked` -- `..` escape attempts
- `test_symlink_escape_blocked` -- symlink-based escapes
- `test_read_only_cannot_move` -- permission enforcement
- `test_move_outside_sandbox_raises` -- cross-boundary moves
- `test_move_collision_appends_timestamp` -- overwrite prevention
- `test_no_delete_method` -- verifies no delete API exists on SafeFS

All 53 tests pass, including all safety tests.
