"""Embeddings-based file categorization.

Three-stage pipeline:
1. **Embed** — encode each file's name + content preview via a lightweight
   embedding model (BGE-small).
2. **Cluster** — group similar embeddings with agglomerative clustering.
3. **Name** — send a sample of each cluster to the LLM to generate a
   concise 2-to-3 word folder name.

On warm runs (existing taxonomy), new files are assigned to the nearest
existing cluster centroid when close enough, and only truly novel groups
trigger a new LLM naming call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering

from localagent.core.embedder import Embedder
from localagent.core.engine import Engine
from localagent.skills.file_organizer.scanner import FileProfile

logger = logging.getLogger(__name__)

TAXONOMY_FILE = "taxonomy.yaml"

# ── Taxonomy I/O ────────────────────────────────────────────────────────────


def load_taxonomy(state_dir: Path) -> dict[str, Any] | None:
    """Load the learned taxonomy from disk, or return None on first run."""
    path = state_dir / TAXONOMY_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or None


def save_taxonomy(state_dir: Path, taxonomy: dict[str, Any]) -> Path:
    """Persist the taxonomy to disk."""
    path = state_dir / TAXONOMY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(taxonomy, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved taxonomy to %s", path)
    return path


# ── Centroid persistence (warm runs) ────────────────────────────────────────

_CENTROIDS_FILE = "centroids.npz"


def _load_centroids(state_dir: Path) -> tuple[np.ndarray, list[str]] | None:
    """Load saved cluster centroids and their category names.

    Returns ``(centroids_matrix, category_names)`` or ``None``.
    """
    path = state_dir / _CENTROIDS_FILE
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        centroids = data["centroids"]
        # Category names stored as a separate text array
        meta_path = state_dir / "centroid_names.yaml"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            names = yaml.safe_load(f)
        if not names or len(names) != centroids.shape[0]:
            return None
        return centroids, names
    except Exception as exc:
        logger.warning("Cannot load centroids: %s", exc)
        return None


def _save_centroids(
    state_dir: Path,
    centroids: np.ndarray,
    category_names: list[str],
) -> None:
    """Persist cluster centroids and their category names."""
    state_dir.mkdir(parents=True, exist_ok=True)
    np.savez(state_dir / _CENTROIDS_FILE, centroids=centroids)
    with open(state_dir / "centroid_names.yaml", "w") as f:
        yaml.dump(category_names, f, default_flow_style=False)
    logger.info("Saved %d centroids", centroids.shape[0])


# ── Clustering ──────────────────────────────────────────────────────────────

_MIN_CLUSTERS = 3
_MAX_CLUSTERS = 25


def _cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    """Cluster embeddings using agglomerative clustering with cosine distance.

    Returns an array of cluster labels (one per embedding).
    Adjusts the threshold if the number of clusters falls outside [3, 15].
    """
    n = embeddings.shape[0]
    if n <= _MIN_CLUSTERS:
        # Too few files to cluster — each gets its own cluster
        return np.arange(n)

    labels = _run_clustering(embeddings, distance_threshold)
    n_clusters = len(set(labels))

    # Adjust threshold to keep cluster count in bounds
    threshold = distance_threshold
    attempts = 0
    while n_clusters > _MAX_CLUSTERS and attempts < 5:
        threshold += 0.05
        labels = _run_clustering(embeddings, threshold)
        n_clusters = len(set(labels))
        attempts += 1
        logger.debug(
            "Too many clusters (%d), raised threshold to %.2f",
            n_clusters, threshold,
        )

    while n_clusters < _MIN_CLUSTERS and threshold > 0.1 and attempts < 10:
        threshold -= 0.05
        labels = _run_clustering(embeddings, threshold)
        n_clusters = len(set(labels))
        attempts += 1
        logger.debug(
            "Too few clusters (%d), lowered threshold to %.2f",
            n_clusters, threshold,
        )

    logger.info("Clustered %d files into %d groups (threshold=%.2f)", n, n_clusters, threshold)
    return labels


def _run_clustering(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Run agglomerative clustering and return labels."""
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(embeddings)


def _compute_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute the centroid (mean embedding) for each cluster.

    Returns an ``(n_clusters, D)`` array, L2-normalised.
    """
    unique_labels = sorted(set(labels))
    centroids = np.zeros((len(unique_labels), embeddings.shape[1]), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroid = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        centroids[i] = centroid
    return centroids


# ── Warm-run: assign new files to existing clusters ─────────────────────────

_WARM_DISTANCE_CUTOFF = 0.5  # max cosine distance to assign to existing cluster


def _assign_to_existing(
    new_embeddings: np.ndarray,
    existing_centroids: np.ndarray,
    existing_names: list[str],
) -> tuple[list[int | None], list[str | None]]:
    """Try to assign each new embedding to the nearest existing centroid.

    Returns two parallel lists (one per new embedding):
    - ``matched_indices``: index into *existing_names*, or ``None`` if no
      good match.
    - ``matched_names``: the category name, or ``None``.
    """
    # Cosine similarity: dot product of L2-normalised vectors
    similarities = new_embeddings @ existing_centroids.T  # (N_new, N_existing)
    best_idx = similarities.argmax(axis=1)
    best_sim = similarities[np.arange(len(best_idx)), best_idx]

    matched_indices: list[int | None] = []
    matched_names: list[str | None] = []
    for i, (idx, sim) in enumerate(zip(best_idx, best_sim)):
        distance = 1.0 - sim
        if distance <= _WARM_DISTANCE_CUTOFF:
            matched_indices.append(int(idx))
            matched_names.append(existing_names[idx])
        else:
            matched_indices.append(None)
            matched_names.append(None)

    n_matched = sum(1 for m in matched_names if m is not None)
    logger.info(
        "Warm assignment: %d/%d new files matched existing clusters",
        n_matched, len(new_embeddings),
    )
    return matched_indices, matched_names


# ── LLM naming prompt ──────────────────────────────────────────────────────

_NAMING_SYSTEM = """\
You are a file organization assistant. You will be given groups of files \
that have been clustered by similarity. For each group, generate a concise \
2-to-3 word folder name that describes the files in that group.

Rules:
- Each name must be 2 to 3 words, clear and descriptive.
- NEVER use generic names like "Miscellaneous", "Documents", "Other", \
"General", "Various Files", or "Mixed Files".
- NEVER use a specific filename as a category name.
- Names should describe the PURPOSE or TYPE of the files \
(e.g. "Tax Documents", "Code Projects", "Travel Photos", "Pay Statements").
- If you are given existing folder names that are already in use, reuse \
them for any group that fits well. Do NOT create a new synonym \
(e.g. if "Photos" exists, don't create "Images").

Respond with ONLY valid JSON in this exact format:
{
  "names": {
    "0": {"name": "Folder Name", "description": "One-sentence description of what belongs here"},
    "1": {"name": "Folder Name", "description": "One-sentence description of what belongs here"},
    ...
  }
}
"""


def _build_naming_prompt(
    cluster_samples: dict[int, list[dict[str, Any]]],
    existing_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build the LLM prompt for naming clusters.

    *cluster_samples* maps cluster label → list of file summaries.
    *existing_names* are folder names from previous runs to prefer reusing.
    """
    groups_text_parts: list[str] = []
    for label in sorted(cluster_samples):
        files = cluster_samples[label]
        file_list = "\n".join(
            f"  - {f['name']} ({f.get('extension', '')})"
            + (f" — preview: {f['content_preview'][:80]}" if f.get("content_preview") else "")
            for f in files
        )
        groups_text_parts.append(f"Group {label}:\n{file_list}")

    groups_text = "\n\n".join(groups_text_parts)

    existing_section = ""
    if existing_names:
        names_list = ", ".join(f'"{n}"' for n in existing_names)
        existing_section = (
            f"\n\nExisting folder names already in use: [{names_list}]. "
            "Reuse these names for any group that fits well instead of "
            "creating new ones."
        )

    user_content = (
        f"Here are {len(cluster_samples)} groups of files to name:"
        f"{existing_section}\n\n{groups_text}"
    )

    return [
        {"role": "system", "content": _NAMING_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _validate_names(
    raw_names: dict[str, Any],
    n_clusters: int,
) -> dict[int, tuple[str, str]]:
    """Validate and normalise LLM-returned cluster names.

    Handles two response formats:
    - ``{"0": {"name": "...", "description": "..."}, ...}``
    - ``{"0": "Folder Name", ...}``  (fallback / legacy)

    Returns ``{cluster_label: (name, description)}``.
    """
    validated: dict[int, tuple[str, str]] = {}
    bad_names = {
        "miscellaneous", "documents", "other", "general",
        "various files", "mixed files", "uncategorized", "misc",
    }

    for key, value in raw_names.items():
        try:
            label = int(key)
        except (ValueError, TypeError):
            logger.warning("Non-integer cluster key from LLM: '%s'", key)
            continue

        # Parse name and description from either format
        if isinstance(value, dict):
            name = str(value.get("name", "")).strip()
            description = str(value.get("description", "")).strip()
        else:
            name = str(value).strip()
            description = ""

        if not name or len(name) < 2 or name.lower() in bad_names:
            logger.warning("Bad cluster name from LLM: '%s' — using fallback", name)
            name = f"Group {label}"
            description = ""

        if not description:
            description = name

        validated[label] = (name, description)

    return validated


# ── Main entry point ────────────────────────────────────────────────────────


def categorize(
    engine: Engine,
    embedder: Embedder,
    profiles: list[FileProfile],
    state_dir: Path,
    *,
    distance_threshold: float = 0.4,
    max_cluster_samples: int = 8,
) -> dict[str, Any]:
    """Categorize files using embeddings + clustering + LLM naming.

    **Cold run** (no saved centroids): embed all files, cluster them,
    ask the LLM to name each cluster.

    **Warm run** (saved centroids exist): embed new files, assign each
    to the nearest existing centroid when close enough.  Files that
    don't match any existing cluster are clustered among themselves
    and sent to the LLM for naming (with existing names passed as
    context so it reuses them when appropriate).

    Returns ``{"taxonomy": {...}, "assignments": {"file": "category", ...}}``.
    """
    if not profiles:
        return {"taxonomy": {}, "assignments": {}}

    # ── Stage 1: Embed ──────────────────────────────────────────────
    texts = [p.embedding_text() for p in profiles]
    logger.info("Embedding %d files", len(texts))
    embeddings = embedder.embed(texts)

    # ── Check for warm-run state ────────────────────────────────────
    saved = _load_centroids(state_dir)
    existing_taxonomy = load_taxonomy(state_dir)

    assignments: dict[str, str] = {}
    taxonomy: dict[str, str] = {}

    if saved is not None:
        # ── Warm run ────────────────────────────────────────────────
        existing_centroids, existing_names = saved
        logger.info(
            "Warm run: %d existing categories, %d files to classify",
            len(existing_names), len(profiles),
        )

        # Carry forward existing taxonomy
        if existing_taxonomy:
            taxonomy.update(existing_taxonomy.get("taxonomy", {}))

        # Try to assign each file to nearest existing centroid
        _, matched_names = _assign_to_existing(
            embeddings, existing_centroids, existing_names,
        )

        # Separate matched vs. unmatched files
        unmatched_indices: list[int] = []
        for i, name in enumerate(matched_names):
            if name is not None:
                assignments[profiles[i].name] = name
            else:
                unmatched_indices.append(i)

        if not unmatched_indices:
            logger.info("All files matched existing clusters — no LLM call needed")
            save_taxonomy(state_dir, {"taxonomy": taxonomy})
            return {"taxonomy": taxonomy, "assignments": assignments}

        # Cluster unmatched files among themselves
        logger.info(
            "%d files unmatched — clustering for new categories",
            len(unmatched_indices),
        )
        unmatched_embeddings = embeddings[unmatched_indices]
        unmatched_profiles = [profiles[i] for i in unmatched_indices]

        if len(unmatched_profiles) == 1:
            # Single unmatched file — give it its own cluster
            new_labels = np.array([0])
        else:
            new_labels = _cluster_embeddings(
                unmatched_embeddings, distance_threshold,
            )

        # Sample files per new cluster for LLM naming
        new_cluster_samples = _sample_clusters(
            new_labels, unmatched_profiles, max_cluster_samples,
        )

        # Ask LLM to name new clusters, passing existing names as context
        messages = _build_naming_prompt(
            new_cluster_samples, existing_names=existing_names,
        )
        try:
            result = engine.generate_json(messages, max_tokens=2048)
        except ValueError as exc:
            logger.error("LLM naming failed: %s — using fallback names", exc)
            result = {"names": {}}

        new_names = _validate_names(result.get("names", {}), len(new_cluster_samples))

        # Assign unmatched files to their new cluster names
        for i, label in enumerate(new_labels):
            name, desc = new_names.get(int(label), (f"Group {label}", f"Group {label}"))
            assignments[unmatched_profiles[i].name] = name
            if name not in taxonomy:
                taxonomy[name] = desc

        # Update centroids: merge old + new
        new_centroids = _compute_centroids(unmatched_embeddings, new_labels)
        new_centroid_names = [
            new_names.get(int(label), (f"Group {label}", ""))[0]
            for label in sorted(set(new_labels))
        ]

        merged_centroids = np.vstack([existing_centroids, new_centroids])
        merged_names = existing_names + new_centroid_names
        _save_centroids(state_dir, merged_centroids, merged_names)

    else:
        # ── Cold run ────────────────────────────────────────────────
        logger.info("Cold run: clustering %d files", len(profiles))

        labels = _cluster_embeddings(embeddings, distance_threshold)
        cluster_samples = _sample_clusters(labels, profiles, max_cluster_samples)

        # Ask LLM to name clusters
        messages = _build_naming_prompt(cluster_samples)
        try:
            result = engine.generate_json(messages, max_tokens=2048)
        except ValueError as exc:
            logger.error("LLM naming failed: %s — using fallback names", exc)
            result = {"names": {}}

        names = _validate_names(result.get("names", {}), len(cluster_samples))

        # Assign every file to its cluster's name
        for i, label in enumerate(labels):
            name, desc = names.get(int(label), (f"Group {label}", f"Group {label}"))
            assignments[profiles[i].name] = name
            if name not in taxonomy:
                taxonomy[name] = desc

        # Save centroids for future warm runs
        centroids = _compute_centroids(embeddings, labels)
        centroid_names = [
            names.get(int(label), (f"Group {label}", ""))[0]
            for label in sorted(set(labels))
        ]
        _save_centroids(state_dir, centroids, centroid_names)

    # Save taxonomy
    save_taxonomy(state_dir, {"taxonomy": taxonomy})

    return {
        "taxonomy": taxonomy,
        "assignments": assignments,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _sample_clusters(
    labels: np.ndarray,
    profiles: list[FileProfile],
    max_samples: int,
) -> dict[int, list[dict[str, Any]]]:
    """Sample up to *max_samples* file summaries per cluster for LLM naming."""
    cluster_samples: dict[int, list[dict[str, Any]]] = {}
    for label_val in sorted(set(labels)):
        indices = [i for i, l in enumerate(labels) if l == label_val]
        # Take up to max_samples, preferring files with content previews
        with_preview = [i for i in indices if profiles[i].content_preview]
        without_preview = [i for i in indices if not profiles[i].content_preview]
        selected = (with_preview + without_preview)[:max_samples]

        cluster_samples[int(label_val)] = [
            profiles[i].to_summary() for i in selected
        ]
    return cluster_samples
