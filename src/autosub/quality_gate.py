"""Development timing regression gate. A pass does not authorize a new default."""
import math


def _cases(report, policy):
    cases = {}
    for item in report["results"]:
        if item["policy"] != policy:
            continue
        if item["clip"] in cases:
            raise ValueError("重複的片段／策略")
        cases[item["clip"]] = item
    if not cases:
        raise ValueError("找不到所選策略的開發片段")
    return cases


def _rates(item):
    s = item["summary"]
    rates = {"strict_100ms": s["strict_time_estimate_rate"],
             "group_outer_100ms": s["group_outer_boundary_estimate_rates"]["100ms"],
             "group_outer_250ms": s["group_outer_boundary_estimate_rates"]["250ms"]}
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in rates.values()):
        raise ValueError("時間估計指標必須是 0–1 的有限數值")
    return rates


def compare_quality(baseline, candidate, baseline_policy="legacy", candidate_policy="conservative"):
    before, after = _cases(baseline, baseline_policy), _cases(candidate, candidate_policy)
    if before.keys() != after.keys():
        raise ValueError("兩版開發片段集合不同，不可比較")
    rows = []
    for name in sorted(before):
        a, b = before[name], after[name]
        if type(a.get("tolerance_ms")) is not int or type(b.get("tolerance_ms")) is not int or a["tolerance_ms"] != 100 or b["tolerance_ms"] != 100:
            raise ValueError("此回歸規則要求比較容差固定為 100ms")
        if type(a.get("reference_offset_ms")) is not int or type(b.get("reference_offset_ms")) is not int or a["reference_offset_ms"] != b["reference_offset_ms"]:
            raise ValueError("參考時間偏移不同或未記錄")
        for key in ("project_sha256", "reference_sha256"):
            if not a.get(key) or a[key] != b.get(key):
                raise ValueError(f"{name} 的來源專案或人工成品不同，請在同一輸入上重做策略比較")
        sa, sb = a["summary"], b["summary"]
        if sa["reference_events"] <= 0 or sa["reference_events"] != sb["reference_events"]:
            raise ValueError("人工句數分母不同或為空")
        ra, rb = _rates(a), _rates(b)
        deltas = {key: rb[key] - ra[key] for key in ra}
        regressed = [key for key, value in deltas.items() if value < -1e-9]
        rows.append({"clip": name, "reference_events": sa["reference_events"], "before": ra, "after": rb,
                     "deltas": deltas, "regressed_metrics": regressed, "passed": not regressed})
    return {"gate": "development-temporal-regression-v1", "baseline_policy": baseline_policy,
            "candidate_policy": candidate_policy, "passed": all(row["passed"] for row in rows),
            "rows": rows, "default_promotion_authorized": False,
            "note": "逐片段要求時間估計不退步；時間配對不是語意或實際發聲標準，通過後仍需人工聲學驗收。"}
