# File Organizer Skill

An autonomous agent that scans your Desktop and Downloads folders, uses a local LLM to understand file contents and purpose, and organizes them into semantically meaningful categories.

## How It Works

1. **Scan** -- walks watched directories, collecting file metadata (name, extension, MIME type, size, dates) and reading content previews of text-based files (first ~512 bytes of `.py`, `.txt`, `.md`, `.csv`, `.json`, etc.).
2. **Categorize** -- sends the file inventory to the local LLM, which proposes a taxonomy of categories tailored to your specific files (not a fixed list of presets).
3. **Review** -- in interactive mode, displays a table of proposed categories and moves for you to approve or reject.
4. **Execute** -- moves files into category subfolders within their source directory. Every move is journaled for undo.

## Usage

```bash
# Run interactively (dry-run, then confirm)
localagent run file-organizer

# Run in auto mode (for cron / scheduled use)
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

## Adaptive Taxonomy

The organizer does not use pre-defined categories. On first run, the LLM analyzes your actual files -- names, types, and content -- and generates categories that fit your scenario (e.g. "Receipts & Invoices", "Machine Learning Projects", "Meeting Notes"). On subsequent runs, it loads the existing taxonomy and extends it only when genuinely new file types appear.

The taxonomy is saved to `~/.config/localagent/file-organizer/taxonomy.yaml` and is fully editable. Open it in any text editor to rename, merge, or remove categories. Delete the file to start fresh on the next run.

You can lock any category by adding `user_locked: true` in the YAML -- the LLM will never rename or remove locked categories.

## Configuration

The file organizer's config section lives under `skills.file-organizer` in `~/.config/localagent/config.yaml`:

```yaml
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
```

| Setting | Description |
|---|---|
| `watch_directories` | Directories to scan for unorganized files |
| `exclude_patterns` | Glob patterns for files to ignore |
| `skip_hidden` | Skip files starting with `.` |
| `content_preview_bytes` | Max bytes to read from text files for LLM context |
| `move_warning_threshold` | Log a warning if a single run proposes more moves than this |

## Safety

All filesystem operations go through the framework's [SafeFS sandbox](../../../../GUARDRAILS.md):

- Only the directories listed in `watch_directories` are accessible
- Files are never deleted -- only moved
- Files are never overwritten -- collisions get a timestamp suffix
- Every move is journaled before execution for crash recovery and undo
- Auto mode aborts entirely on any error -- no partial execution

## Architecture

```
skills/file_organizer/
├── scanner.py      # Builds FileProfile inventory (metadata + content preview)
├── categorizer.py  # LLM-powered adaptive taxonomy (cold start + warm evolution)
├── mover.py        # Safe moves via SafeFS, undo journal, interactive undo
└── skill.py        # FileOrganizerSkill tying it all together
```

### Manifest

```python
SkillManifest(
    name="file-organizer",
    description="Organizes files into smart, LLM-generated categories",
    permissions={"read", "move"},
)
```
