import copy
import random

import pytest

from autosub.domain import Cue, Project, TimingSettings, Word
from autosub.timing import apply_timing, segment_words


def make_project(*ranges, duration=10000, settings=None):
    return Project(duration_ms=duration, settings=settings or TimingSettings(),
                   words=[Word(start, end, text) for start, end, text in ranges],
                   speech=[[start, end] for start, end, _ in ranges])


def base(start, end, text="テスト", **kwargs):
    return Cue(str(start), start, end, text, start, end, **kwargs)


def test_short_cue_cannot_cover_next_speech():
    project = Project(duration_ms=10000)
    cues = apply_timing(project, [base(1000, 1150), base(1350, 2000)])
    assert cues[0].start_ms == 980
    assert cues[0].end_ms == cues[1].start_ms == 1330
    assert "受邊界限制，未達最短時長" in cues[0].flags


def test_remaining_gap_measured_after_tail():
    cues = apply_timing(Project(duration_ms=5000), [base(1000, 2000), base(2400, 3000)])
    assert cues[0].end_ms == cues[1].start_ms == 2380


def test_gap_limit_is_strict_and_tail_cap_is_hard():
    p = Project(duration_ms=10000)
    cues = apply_timing(p, [base(1000, 2000), base(2520, 3000)])
    assert cues[0].end_ms == 2200  # exactly 300ms does not close
    p.settings.close_gap_ms = 1000
    cues = apply_timing(p, [base(1000, 2000), base(2600, 3000)])
    assert cues[0].end_ms == 2200  # cannot bridge beyond max tail 500ms


def test_minimum_duration_extends_and_stays_in_media():
    cues = apply_timing(Project(duration_ms=2000), [base(100, 200), base(1900, 1980)])
    assert cues[0].end_ms - cues[0].start_ms == 500
    assert cues[1].end_ms == 2000
    assert cues[1].flags


def test_elongation_does_not_receive_tail_or_gap_extension():
    cues = apply_timing(Project(duration_ms=5000), [base(1000, 1300, elongated=True), base(1500, 2500)])
    assert cues[0].end_ms == 1300
    assert "受邊界限制，未達最短時長" in cues[0].flags


def test_true_overlap_is_preserved_and_flagged():
    cues = apply_timing(Project(duration_ms=5000), [base(1000, 2000), base(1800, 2500)])
    assert cues[0].end_ms >= 2000
    assert all("時間重疊，請複核邊界／多人發話" in c.flags for c in cues)


def test_only_artificial_lead_overlap_is_removed():
    cues = apply_timing(Project(duration_ms=5000), [base(1000, 2000), base(2010, 2500)])
    assert cues[0].end_ms == cues[1].start_ms == 2000
    assert not any("時間重疊" in flag for c in cues for flag in c.flags)


def test_locked_manual_edit_survives_resegmentation():
    p = make_project((1000, 1500, "こんにちは。"), (3000, 3800, "世界。"))
    p.cues = apply_timing(p)
    p.cues[0].text = "人工修正"
    p.cues[0].locked = True
    p.cues[0].start_ms = 900
    p.cues[0].end_ms = 1800
    original = copy.deepcopy(p.cues[0])
    p.settings.max_chars = 2
    result = apply_timing(p)
    assert result[0] == original


def test_resegmentation_next_to_locked_cue_does_not_drop_text():
    p = make_project((1000, 1500, "one"), (1550, 2000, " two"), (2100, 2500, " three"))
    p.cues = [base(1000, 1500, "LOCKED", locked=True)]
    result = apply_timing(p)
    assert "two" in " ".join(c.text for c in result)
    assert "three" in " ".join(c.text for c in result)
    assert result[0].text == "LOCKED"


def test_elongation_annotation_survives_unlock_and_retime():
    p = make_project((1000, 1500, "長音。"))
    p.cues = [base(1000, 1500, "長音。", elongated=True)]
    assert apply_timing(p)[0].end_ms == 1500


@pytest.mark.parametrize("texts", [
    ["こんにちは", "世界", "です。"], ["今天", "玩", "遊戲。"],
    [" Hello", " world", " again."], [" 오늘", " 게임", " 합니다."],
])
def test_multilingual_text_survives_length_segmentation(texts):
    p = make_project(*[(i * 2800, i * 2800 + 2400, text) for i, text in enumerate(texts)])
    cues = segment_words(p)
    assert "".join(c.text.replace(" ", "") for c in cues) == "".join(texts).replace(" ", "")
    assert all(c.speech_end_ms - c.speech_start_ms <= 7000 for c in cues)


def test_vad_does_not_snap_internal_cues_to_entire_speech_region():
    p = make_project((3000, 4000, "one."), (5000, 6000, " two."))
    p.speech = [[1000, 9000]]
    cues = segment_words(p)
    assert [(c.speech_start_ms, c.speech_end_ms) for c in cues] == [(3000, 4000), (5000, 6000)]


def test_generated_sequences_obey_bounds_and_preserve_speech():
    rng = random.Random(84)
    for _ in range(100):
        cursor = 0
        raw = []
        for i in range(20):
            cursor += rng.randrange(20, 800)
            end = cursor + rng.randrange(30, 2000)
            raw.append(base(cursor, end))
            cursor = end
        p = Project(duration_ms=cursor + 50)
        result = apply_timing(p, raw)
        for i, cue in enumerate(result):
            assert 0 <= cue.start_ms <= cue.speech_start_ms < cue.speech_end_ms <= cue.end_ms <= p.duration_ms
            assert cue.end_ms <= cue.speech_end_ms + p.settings.max_tail_ms
            if i + 1 < len(result):
                assert cue.end_ms <= result[i + 1].start_ms


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        TimingSettings(lead_ms=51).validate()
    with pytest.raises(ValueError):
        TimingSettings(tail_ms=600).validate()


def test_vad_silence_splits_even_when_asr_word_gap_is_shorter():
    p = make_project((1000, 2000, "前半"), (2200, 3000, "後半"))
    p.speech = [[1000, 1800], [2500, 3000]]
    # ASR word gap = 200ms, VAD silence = 700ms.
    assert [c.text for c in segment_words(p)] == ["前半", "後半"]


def test_hard_cap_looks_back_to_pause_instead_of_last_word():
    p = make_project((0, 2000, "one"), (2200, 4200, " two"),
                     (4200, 6200, " three"), (6200, 8000, " four"))
    p.speech = [[0, 8000]]
    p.settings.soft_duration_ms = 5000
    cues = segment_words(p)
    assert cues[0].text == "one"
    assert "".join(c.text.replace(" ", "") for c in cues) == "onetwothreefour"
    assert all(c.speech_end_ms - c.speech_start_ms <= 7000 for c in cues)


def test_boundary_refinement_can_be_disabled():
    p = make_project((700, 1400, "テスト。"))
    p.speech = [[1000, 1300]]
    assert segment_words(p)[0].speech_start_ms == 1000
    p.settings.boundary_search_ms = 0
    c = segment_words(p)[0]
    assert (c.speech_start_ms, c.speech_end_ms) == (700, 1400)


def test_nearby_unrelated_speech_does_not_pull_edge_across_silence():
    p = make_project((1000, 1500, "テスト。"))
    p.speech = [[800, 900], [1000, 1500], [1550, 1700]]
    c = segment_words(p)[0]
    assert (c.speech_start_ms, c.speech_end_ms) == (1000, 1500)
