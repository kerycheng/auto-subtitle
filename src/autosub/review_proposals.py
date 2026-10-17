"""Explicit, reversible native-sentence review; never rewrite or delete ASR text."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

from .domain import Cue, Project, word_key
from .editing import merge_cues, review_cues
from .segmentation_audit import uncovered_speech
from .timing import END, joined, visible_length

ALGORITHM = "native-review-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def signature(project):
    data = asdict(project)
    # Keep append-only review history when reverting a cue operation.
    data.pop("review_events")
    return digest(data)


def _vad_coverage(project, start, end):
    intervals = sorted((max(start, a), min(end, b)) for a, b in project.speech
                       if a < end and b > start)
    covered, cursor = 0, start
    for a, b in intervals:
        covered += max(0, b-max(cursor, a))
        cursor = max(cursor, b)
    return covered / max(1, end-start)


def _short(cue):
    return 0 < visible_length(cue.text) <= 6 and cue.end_ms-cue.start_ms <= 1500


def proposals(project: Project):
    project.validate()
    if project.asr_strategy != "native-v1" or not project.native_complete:
        raise ValueError("複核候選僅適用已完成的日文原生句界專案")
    stamp = signature(project)
    result = []
    def add(kind, ids, region, evidence, preview=None):
        item = {"algorithm": ALGORITHM, "signature": stamp, "kind": kind,
                "cue_ids": ids, "range_ms": region, "evidence": evidence,
                "adoptable": kind == "merge", "decision": "unreviewed"}
        item["id"] = digest(item)[:24]
        if preview:
            preview.id = "review-" + item["id"]
            item["preview"] = asdict(preview)
        result.append(item)

    locations = {}
    owners = Counter(token for cue in project.cues for token in cue.source_tokens or [])
    for i, word in enumerate(project.words):
        locations.setdefault(word_key(word), []).append(i)
    def indexes(cue):
        tokens = cue.source_tokens or []
        if not tokens or any(len(locations.get(t, [])) != 1 or owners[t] != 1 for t in tokens):
            return []
        positions = [locations[t][0] for t in tokens]
        if positions != list(range(positions[0], positions[-1]+1)):
            return []
        if cue.text != joined([project.words[i] for i in positions]):
            return []
        return positions

    for i, cue in enumerate(project.cues):
        positions = indexes(cue)
        coverage = _vad_coverage(project, cue.speech_start_ms, cue.speech_end_ms)
        probabilities = [project.words[j].probability for j in positions]
        evidence = {"vad_coverage": coverage,
                    "min_probability": min(probabilities) if probabilities else None,
                    "short": _short(cue), "ownership_unambiguous": bool(positions)}
        if _short(cue) or coverage < .2:
            add("listen", [cue.id], [cue.start_ms, cue.end_ms], evidence)
        if i+1 == len(project.cues):
            continue
        other = project.cues[i+1]
        right = indexes(other)
        # No speaker evidence is available. These constraints only limit which
        # previews to offer; they do not establish that two cues are one sentence.
        if (not (_short(cue) or _short(other)) or not positions or not right or
                any(c.locked or c.reviewed or c.edit_group or c.candidate_original for c in (cue, other)) or
                positions[-1]+1 != right[0] or END.search(cue.text.strip())):
            continue
        gap = other.speech_start_ms-cue.speech_end_ms
        if not 0 <= gap <= 300 or other.start_ms < cue.start_ms:
            continue
        if coverage < .2 or _vad_coverage(project, other.speech_start_ms, other.speech_end_ms) < .2:
            continue
        if any(c.id not in {cue.id, other.id} and c.speech_start_ms < other.speech_end_ms
               and c.speech_end_ms > cue.speech_start_ms for c in project.cues):
            continue
        merged = merge_cues(project, i)[i]
        if (merged.text != cue.text+other.text or
                merged.end_ms-merged.start_ms > project.settings.max_duration_ms or
                visible_length(merged.text) > project.settings.max_chars or
                Counter(merged.source_tokens) != Counter(cue.source_tokens+other.source_tokens)):
            continue
        merged.flags = list(dict.fromkeys(merged.flags + ["人工採用短句合併，仍需試聽確認語意／說話者"]))
        merged.timing_trace["review_algorithm"] = ALGORITHM
        add("merge", [cue.id, other.id], [merged.start_ms, merged.end_ms],
            {"gap_ms": gap, "short_cue_ids": [c.id for c in (cue, other) if _short(c)],
             "speaker_verified": False, "semantic_verified": False}, merged)
    for start, end in uncovered_speech(project):
        add("listen-gap", [], [start, end], {"reason": "VAD 區間未被詞時間覆蓋；不等於已確認漏詞"})
    return result


def _checked_merge(project, proposal):
    if proposal.get("kind") != "merge" or proposal.get("signature") != signature(project):
        raise ValueError("候選已過期或僅供試聽；請重新產生候選")
    expected = next((p for p in proposals(project) if p["id"] == proposal.get("id")), None)
    if expected is None or proposal != expected:
        raise ValueError("候選資料不完整或已改變，拒絕採用")
    index = next(i for i, c in enumerate(project.cues) if c.id == proposal["cue_ids"][0])
    updated = copy.deepcopy(project)
    updated.cues[index:index+2] = [Cue(**copy.deepcopy(expected["preview"]))]
    # Normalize with the existing UI pass before signing the receipt. Merely
    # displaying an adopted project must not invalidate its persistent rollback.
    review_cues(updated)
    updated.validate()
    if ("".join(c.text for c in updated.cues) != "".join(c.text for c in project.cues) or
            Counter(t for c in updated.cues for t in c.source_tokens or []) !=
            Counter(t for c in project.cues for t in c.source_tokens or [])):
        raise ValueError("合併未完整保留文字或詞所有權，拒絕採用")
    return updated


def apply_proposal(project, proposal, reviewer):
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("請填入操作紀錄名稱")
    updated = _checked_merge(project, proposal)
    receipt = {"schema_version": 1, "algorithm": ALGORITHM, "proposal": copy.deepcopy(proposal),
               "before_signature": signature(project), "after_signature": signature(updated),
               "before_cues": [asdict(c) for c in project.cues],
               "reviewer": reviewer.strip(), "at_utc": datetime.now(timezone.utc).isoformat()}
    receipt["sha256"] = digest(receipt)
    updated.review_events.append({"action": "apply_merge_proposal", "proposal": proposal["id"],
                                  "receipt_sha256": receipt["sha256"], "reviewer": reviewer.strip(),
                                  "at_utc": receipt["at_utc"]})
    return updated, receipt


def restore_proposal(project, receipt):
    payload = {k: v for k, v in receipt.items() if k != "sha256"}
    if (receipt.get("schema_version") != 1 or receipt.get("algorithm") != ALGORITHM or
            digest(payload) != receipt.get("sha256") or
            not isinstance(receipt.get("before_cues"), list) or
            not isinstance(receipt.get("proposal"), dict)):
        raise ValueError("撤回紀錄無效或已損壞")
    if signature(project) != receipt.get("after_signature"):
        raise ValueError("採用後專案已變更，拒絕覆寫後續修正；請先還原後續操作")
    restored = copy.deepcopy(project)
    restored.cues = [Cue(**copy.deepcopy(c)) for c in receipt["before_cues"]]
    restored.validate()
    if signature(restored) != receipt["before_signature"]:
        raise ValueError("原始字幕備份不符，拒絕撤回")
    expected = _checked_merge(restored, receipt["proposal"])
    if signature(expected) != receipt["after_signature"]:
        raise ValueError("撤回紀錄與原操作不符")
    restored.review_events.append({"action": "restore_merge_proposal",
                                   "receipt_sha256": receipt["sha256"],
                                   "at_utc": datetime.now(timezone.utc).isoformat()})
    return restored
