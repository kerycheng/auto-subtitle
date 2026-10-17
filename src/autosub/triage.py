"""Explainable review ordering; scores indicate review urgency, never accuracy."""
from __future__ import annotations


def review_priority(row):
    reasons = []
    score = 0
    if not row["reference_ids"] or not row["generated_ids"]:
        score += 100
        reasons.append("未配對，需聽辨是否缺句或對應不同")
    if row["ambiguous"]:
        score += 80
        reasons.append("重疊或配對不確定")
    deltas = [abs(row[key]) for key in ("start_delta_ms", "end_delta_ms") if row[key] is not None]
    error = max(deltas, default=0)
    if error > 500:
        score += 60
        reasons.append("外側頭尾差超過 500ms")
    elif error > 100:
        score += 30
        reasons.append("外側頭尾差超過 100ms")
    if row["reference_ids"] and row["generated_ids"] and row["kind"] != "1:1":
        score += 20
        reasons.append("拆合句，需確認內部切點")
    return {"score": score, "max_delta_ms": error, "reasons": reasons or ["仍需確認文字與發聲邊界"]}


def review_order(session, unreviewed_only=True):
    rows = session["rows"]
    indexes = [i for i, row in enumerate(rows)
               if not unreviewed_only or row["decision"] == "unreviewed"]
    return sorted(indexes, key=lambda i: (-review_priority(rows[i])["score"],
        -review_priority(rows[i])["max_delta_ms"],
        (rows[i]["reference_span"] or rows[i]["generated_span"])[0], i))


def quality_diagnostics(project):
    """Instrument measurable risks without pretending VAD/reference is acoustic gold."""
    risks = []
    for cue in project.cues:
        reasons = []
        raw = cue.timing_trace.get("raw")
        if raw and cue.speech_end_ms < raw[1]:
            reasons.append("基礎句尾早於原辨識詞尾；需聽辨是否截尾")
        if cue.end_ms - cue.start_ms > project.settings.max_duration_ms:
            reasons.append("顯示時長超過設定上限")
        if len("".join(cue.text.split())) > project.settings.max_chars:
            reasons.append("字數超過設定上限")
        if any("重疊" in flag for flag in cue.flags):
            reasons.append("時間重疊，需檢查多人發話")
        if reasons:
            risks.append({"cue_id": cue.id, "span": [cue.start_ms, cue.end_ms], "text": cue.text,
                          "reasons": reasons})
    return risks
