from autosub.domain import Cue
from autosub.evaluation import boundary_report, read_ass


def test_reference_parser_handles_commas_comments_and_translation(tmp_path):
    path = tmp_path / "reference.ass"
    path.write_text("""[Aegisub Project Garbage]
Video File: original.mp4
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,主角,,0,0,0,,{\\b1}翻譯,有逗號
Comment: 0,0:00:02.00,0:00:03.00,主角,,0,0,0,,忽略
Dialogue: 0,0:00:02.00,0:00:03.00,註解,,0,0,0,,忽略註解
Dialogue: 0,0:00:02.00,0:00:03.00,主角,,0,0,0,,
""", encoding="utf-8")
    cues, video = read_ass(path)
    assert video == "original.mp4"
    assert len(cues) == 1
    assert cues[0].text == "翻譯,有逗號"
    assert (cues[0].start_ms, cues[0].end_ms) == (1000, 2000)


def test_empty_output_has_zero_coverage_not_fake_accuracy(tmp_path):
    from autosub.evaluation import ReferenceCue
    report = boundary_report([ReferenceCue(1000, 2000, "test", "main")], [], 0, 3000)
    assert report["start_ms_reference_to_generated"]["within_50ms"] == 0
    assert report["start_ms_reference_to_generated"]["median_nearest_ms"] is None


def test_one_generated_boundary_cannot_match_two_reference_boundaries():
    from autosub.evaluation import ReferenceCue
    reference = [ReferenceCue(1000, 2000, "one", "a"), ReferenceCue(1010, 2200, "two", "b")]
    generated = [Cue("x", 1005, 2100, "generated", 1005, 2000)]
    report = boundary_report(reference, generated, 0, 3000)
    metric = report["start_ms_one_to_one"]["50ms"]
    assert metric["matches"] == 1
    assert metric["precision"] == 1
    assert metric["recall"] == 0.5
