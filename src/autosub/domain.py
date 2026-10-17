from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .file_io import replace_checkpoint


@dataclass
class TimingSettings:
    lead_ms: int = 20
    tail_ms: int = 200
    max_tail_ms: int = 500
    min_duration_ms: int = 500
    close_gap_ms: int = 300
    soft_duration_ms: int = 5000
    max_duration_ms: int = 7000
    max_chars: int = 42
    silence_split_ms: int = 450
    boundary_search_ms: int = 400
    start_search_ms: int = 400
    end_search_ms: int = 400

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必須為非負整數")
        if self.lead_ms > 50:
            raise ValueError("句首先行量須介於 0–50ms")
        if self.tail_ms > self.max_tail_ms:
            raise ValueError("一般尾留不能超過尾留上限")
        if not 0 < self.soft_duration_ms <= self.max_duration_ms:
            raise ValueError("分段目標須大於零且不超過分段上限")
        if self.max_chars < 2 or self.min_duration_ms < 10:
            raise ValueError("字數上限至少 2，最短時長至少 10ms")
        if max(self.boundary_search_ms, self.start_search_ms, self.end_search_ms) > 1000:
            raise ValueError("邊界搜尋範圍不得超過 1000ms")


@dataclass
class Word:
    start_ms: int
    end_ms: int
    text: str
    probability: float = 1.0


@dataclass
class Cue:
    id: str
    start_ms: int
    end_ms: int
    text: str
    speech_start_ms: int
    speech_end_ms: int
    flags: list[str] = field(default_factory=list)
    locked: bool = False
    elongated: bool = False
    source_tokens: list[str] | None = None
    reviewed: bool = False
    edit_group: str | None = None
    timing_trace: dict = field(default_factory=dict)
    candidate: dict | None = None
    candidate_original: dict | None = None


def word_key(word: Word) -> str:
    """Stable ownership across retiming, manual edits and resumed ASR chunks."""
    return f"{word.start_ms}:{word.end_ms}:{word.text}"


@dataclass
class Project:
    media_path: str = ""
    duration_ms: int = 0
    language: str = "ja"
    model: str = "small"
    device: str = "cpu"
    glossary: str = ""
    audio_stream: int = 0
    vad_threshold: float = 0.5
    words: list[Word] = field(default_factory=list)
    speech: list[list[int]] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)
    settings: TimingSettings = field(default_factory=TimingSettings)
    status: str = "new"
    audio_path: str = ""
    source_fingerprint: dict = field(default_factory=dict)
    # Recognition checkpoints are chunk-based: incomplete chunks are rerun.
    completed_chunks: list[int] = field(default_factory=list)
    boundary_policy: str = "legacy"
    review_events: list[dict] = field(default_factory=list)
    asr_strategy: str = "auto"
    native_groups: list[list[int]] = field(default_factory=list)
    native_resume_ms: int = 0
    native_complete: bool = False
    schema_version: int = 4

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Back up the exact old file on first migrated save, before replacement.
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("schema_version") in {1, 2, 3}:
                backup = path.with_suffix(path.suffix + f".v{previous['schema_version']}.bak")
                if not backup.exists():
                    shutil.copyfile(path, backup)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        replace_checkpoint(temporary, path)

    @classmethod
    def load(cls, path: Path) -> Project:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") not in {1, 2, 3, 4}:
            raise ValueError("不支援的專案版本")
        if raw["schema_version"] >= 3 and "asr_strategy" not in raw:
            raise ValueError("第三版專案缺少辨識策略，請使用完整專案或備份")
        if raw["schema_version"] == 1:
            radius = raw["settings"].get("boundary_search_ms", 400)
            raw["settings"].setdefault("start_search_ms", radius)
            raw["settings"].setdefault("end_search_ms", radius)
        raw.setdefault("asr_strategy", "fixed30-v1")
        if raw["schema_version"] < 4 and raw["asr_strategy"] == "auto":
            raw["asr_strategy"] = "mixed-vad-v1" if (raw["language"] == "mixed" and not raw["words"]
                                                       and not raw.get("completed_chunks")) else "fixed30-v1"
        if raw["schema_version"] == 4 and raw["asr_strategy"] == "native-v1" and not all(
                name in raw for name in ("native_groups", "native_resume_ms", "native_complete")):
            raise ValueError("原生句界專案缺少接續資料，請使用完整專案或備份")
        raw["schema_version"] = 4
        raw["settings"] = TimingSettings(**raw["settings"])
        raw["settings"].validate()
        raw["words"] = [Word(**w) for w in raw["words"]]
        raw["cues"] = [Cue(**c) for c in raw["cues"]]
        project = cls(**raw)
        project.validate()
        return project

    def validate(self) -> None:
        self.settings.validate()
        if self.asr_strategy not in {"auto", "fixed30-v1", "mixed-vad-v1", "native-v1"}:
            raise ValueError("不支援的辨識分段策略")
        if (type(self.native_resume_ms) is not int or not 0 <= self.native_resume_ms <= self.duration_ms
                or type(self.native_complete) is not bool):
            raise ValueError("原生句界接續時間無效")
        if self.asr_strategy == "native-v1":
            if self.language != "ja" or self.completed_chunks:
                raise ValueError("原生句界策略僅適用日文且不可混用舊分段進度")
            if (not isinstance(self.native_groups, list) or any(
                    not isinstance(group, list) or not group or any(type(i) is not int for i in group)
                    for group in self.native_groups) or
                    [i for group in self.native_groups for i in group] != list(range(len(self.words)))):
                raise ValueError("原生句界未完整對應原始辨識文字")
            if self.native_complete and self.native_resume_ms != self.duration_ms:
                raise ValueError("原生句界完成標記與進度不一致")
        elif self.native_groups or self.native_resume_ms or self.native_complete:
            raise ValueError("其他辨識策略不可混用原生句界資料")
        if self.boundary_policy not in {"legacy", "conservative"}:
            raise ValueError("不支援的邊界策略")
        if self.duration_ms < 0 or not 0 < self.vad_threshold < 1:
            raise ValueError("專案時長或 VAD 門檻無效")
        if len({c.id for c in self.cues}) != len(self.cues):
            raise ValueError("字幕 ID 重複，無法安全配對候選")
        for word in self.words:
            if not 0 <= word.start_ms <= word.end_ms <= self.duration_ms:
                raise ValueError("詞級時間超出媒體範圍")
            if not math.isfinite(word.probability):
                raise ValueError("詞級信心值無效")
        for cue in self.cues:
            if not 0 <= cue.start_ms < cue.end_ms <= self.duration_ms:
                raise ValueError("字幕時間超出媒體範圍")
            if not 0 <= cue.speech_start_ms < cue.speech_end_ms <= self.duration_ms:
                raise ValueError("基礎發聲時間超出媒體範圍")
