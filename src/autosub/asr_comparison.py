"""Controlled ASR evidence and post-recognition character comparison, without text repair."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .alignment_lab import file_hash
from .language_validation import normalized_text


def model_identity(path):
    path = Path(path).resolve()
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in
                   {".bin", ".safetensors", ".json", ".txt", ".model"})
    if not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise ValueError("No local model weights")
    records = [{"path": p.relative_to(path).as_posix(), "sha256": file_hash(p)} for p in files]
    signature = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"sha256": signature, "files": records}


def validate_input(manifest):
    if manifest.get("text_provided") is not False or not manifest.get("windows"):
        raise ValueError("Expected reference-free nonempty input")
    allowed = {"id", "language", "window_ms", "owner_ms", "source_audio_sha256", "padding_samples"}
    seen = set()
    for row in manifest["windows"]:
        start, end = row["window_ms"]
        if (set(row) - allowed or row["id"] in seen or not 0 <= start < end <= manifest["duration_ms"]
                or (row.get("language") or manifest.get("language")) not in {"en", "zh", "ja", "ko"}):
            raise ValueError("Invalid or text-bearing ASR window")
        seen.add(row["id"])


def validate_resume(report, signature, backend, identity, config, manifest):
    if (report.get("manifest_sha256") != signature or report.get("backend") != backend
            or report.get("model_identity") != identity or report.get("configuration") != config
            or report.get("text_provided") is not False):
        raise ValueError("Resume mismatch: model, configuration or input changed")
    windows = {r["id"]: r for r in manifest["windows"]}
    seen = set()
    for row in report["results"]:
        if row["id"] in seen or row["id"] not in windows or any(
                row.get(k) != v for k, v in windows[row["id"]].items()):
            raise ValueError("Resume contains duplicate or changed windows")
        seen.add(row["id"])


def character_alignment(reference, hypothesis):
    """Minimum character edits; deletion/insertion evidence is not a semantic diagnosis.

    Deterministic ties prefer match/substitution, then deletion, then insertion.
    Different equally optimal alignments can distribute the counts differently.
    """
    a, b = normalized_text(reference), normalized_text(hypothesis)
    if not a:
        raise ValueError("Empty normalized reference")
    grid = [list(range(len(b) + 1))]
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(grid[i - 1][j - 1] + (x != y), grid[i - 1][j] + 1, row[j - 1] + 1))
        grid.append(row)
    i, j = len(a), len(b)
    operations = []
    while i or j:
        if i and j and grid[i][j] == grid[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            operations.append({"kind": "equal" if a[i - 1] == b[j - 1] else "substitution",
                "reference": a[i - 1], "hypothesis": b[j - 1], "reference_offset": i - 1,
                "hypothesis_offset": j - 1})
            i, j = i - 1, j - 1
        elif i and grid[i][j] == grid[i - 1][j] + 1:
            operations.append({"kind": "deletion", "reference": a[i - 1], "hypothesis": "",
                "reference_offset": i - 1, "hypothesis_offset": j})
            i -= 1
        else:
            operations.append({"kind": "insertion", "reference": "", "hypothesis": b[j - 1],
                "reference_offset": i, "hypothesis_offset": j - 1})
            j -= 1
    operations.reverse()
    counts = {kind: sum(op["kind"] == kind for op in operations)
              for kind in ("substitution", "deletion", "insertion")}
    return {"reference_characters": len(a), "character_edits": grid[-1][-1],
        "mechanical_cer": grid[-1][-1] / len(a), **counts,
        "differences": [op for op in operations if op["kind"] != "equal"]}


def compare_runs(dataset_path, manifest_path, result_paths):
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_input(manifest)
    if (manifest["source_manifest_sha256"] != file_hash(dataset_path)
            or manifest["audio_sha256"] != file_hash(manifest["audio_path"])):
        raise ValueError("Dataset/input identity changed")
    samples = {r["id"]: r for r in dataset["samples"]}
    if len(samples) != len(dataset["samples"]) or set(samples) != {r["id"] for r in manifest["windows"]}:
        raise ValueError("Dataset IDs differ")
    for window in manifest["windows"]:
        sample = samples[window["id"]]
        if (window["language"] != sample["language"] or window["source_audio_sha256"] != sample["audio_sha256"]
                or file_hash(sample["audio_path"]) != sample["audio_sha256"]):
            raise ValueError("Dataset source changed")
    report = {"scope": "Controlled ASR-only read-speech comparison; not livestream or acoustic timing accuracy",
        "dataset": dataset["dataset"], "license": dataset["license"],
        "manifest_sha256": file_hash(manifest_path), "normalization":
        "NFKC/casefold/remove whitespace and Unicode punctuation; no script or numeral equivalence",
        "semantic_omissions_hallucinations_proper_names": "unrated; edit operations are only review evidence",
        "reference_quality": "dataset raw text; parenthetical glosses/numerals may differ from speech; not independently acoustically audited",
        "timing_and_segmentation": "not evaluated by CER", "runs": []}
    identities = set()
    shared_contract = None
    for path in result_paths:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        if not run.get("model_identity") or not run.get("configuration"):
            raise ValueError("Missing reproducibility evidence")
        validate_resume(run, report["manifest_sha256"], run["backend"], run["model_identity"],
                        run["configuration"], manifest)
        if run["status"] != "complete" or {r["id"] for r in run["results"]} != set(samples):
            raise ValueError("Incomplete ASR run")
        contract = {k: run["configuration"].get(k) for k in
                    ("device", "dtype", "batch_size", "language", "prompt", "vad_filter")}
        if shared_contract is not None and contract != shared_contract:
            raise ValueError("Incompatible common ASR conditions")
        shared_contract = contract
        identity = run["model_identity"]["sha256"]
        if identity in identities:
            raise ValueError("Duplicate model run")
        identities.add(identity)
        result = {"model": run["model"], "model_identity": identity, "backend": run["backend"],
            "result_path": str(Path(path).resolve()), "result_sha256": file_hash(path),
            "configuration": run["configuration"], "environment": run.get("environment"),
            "load_seconds": run["load_seconds"], "inference_seconds": sum(r["seconds"] for r in run["results"]),
            "load_attempts_seconds": run.get("load_attempts_seconds", []),
            "memory": {"gpu_before_load": run.get("gpu_before_load"), "gpu_after_inference": run.get("gpu_after_inference"),
                "peak_tensor_bytes": max((r.get("peak_tensor_bytes", 0) for r in run["results"]), default=0) or None,
                "note": "GPU snapshots include desktop/other apps; torch tensor peak excludes driver/other apps; CT2 tensor peak unavailable"},
            "samples": [], "by_language": {}}
        for row in run["results"]:
            sample = samples[row["id"]]
            result["samples"].append({"id": row["id"], "language": sample["language"],
                "reference_text": sample["reference_text"], "recognized_text": row["text"],
                "reference_parentheticals": re.findall(r"[（(][^()（）]*[)）]", sample["reference_text"]),
                **character_alignment(sample["reference_text"], row["text"])})
        for lang in sorted({r["language"] for r in result["samples"]}):
            rows = [r for r in result["samples"] if r["language"] == lang]
            counts = {key: sum(r[key] for r in rows) for key in
                      ("reference_characters", "character_edits", "substitution", "deletion", "insertion")}
            result["by_language"][lang] = {"samples": len(rows), **counts,
                "distinct_normalized_references": len({normalized_text(r["reference_text"]) for r in rows}),
                "mechanical_cer": counts["character_edits"] / counts["reference_characters"]}
        report["runs"].append(result)
    return report
