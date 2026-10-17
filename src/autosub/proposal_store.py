"""Immutable merge receipts beside a project; save evidence before changing cues."""
import json
import os
import re
import uuid
from pathlib import Path

from .review_proposals import apply_proposal, digest, restore_proposal, signature


def receipt_path(project_path, sha):
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("合併收據識別碼無效")
    project_path = Path(project_path)
    parent = project_path.parent.resolve()
    review = project_path.parent / (project_path.stem + ".review")
    root = review / "merge-receipts"
    expected = parent / review.name / "merge-receipts"
    if review.is_symlink() or root.is_symlink() or root.resolve() != expected:
        raise ValueError("合併收據目錄不可重新導向其他位置")
    result = root / (sha + ".json")
    if result.is_symlink():
        raise ValueError("合併收據不可為符號連結")
    return result


def read_receipt(project_path, sha):
    raw = json.loads(receipt_path(project_path, sha).read_text(encoding="utf-8"))
    if raw.get("sha256") != sha or digest({k:v for k,v in raw.items() if k != "sha256"}) != sha:
        raise ValueError("合併收據已損壞")
    return raw


def save_receipt(project_path, receipt):
    sha = receipt["sha256"]
    if digest({k:v for k,v in receipt.items() if k != "sha256"}) != sha:
        raise ValueError("合併收據不完整")
    target = receipt_path(project_path, sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if read_receipt(project_path, sha) != receipt:
            raise ValueError("既有收據內容不符，拒絕覆寫")
        return
    temporary = target.with_name(uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        os.replace(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)


def adopt_and_save(project, proposal, project_path):
    updated, receipt = apply_proposal(project, proposal, "GUI 明確採用合併")
    save_receipt(project_path, receipt)
    updated.save(Path(project_path))
    return updated


def restore_and_save(project, project_path):
    stamp = signature(project)
    for event in reversed(project.review_events):
        if event.get("action") != "apply_merge_proposal":
            continue
        try:
            receipt = read_receipt(project_path,event.get("receipt_sha256"))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if receipt.get("after_signature") == stamp:
            restored = restore_proposal(project,receipt)
            restored.save(Path(project_path))
            return restored
    raise ValueError("沒有符合目前專案的合併收據；請保留 .review 資料夾，並先復原後續修改。")
