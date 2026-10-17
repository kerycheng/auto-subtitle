from autosub.recognition_windows import monolingual_windows


def test_short_fragments_are_kept_in_full_context_instead_of_individual_asr_calls():
    speech = [[5000, 5200], [7000, 8500], [9300, 9800], [10000, 12500], [13000, 18000]]
    assert monolingual_windows(speech, 30000) == [(4000, 19000, 0, 30000)]


def test_long_intro_and_outro_silence_are_not_fed_to_the_recognizer():
    assert monolingual_windows([[100000, 105000]], 180000) == [(99000, 106000, 0, 180000)]


def test_continuous_speech_gets_context_and_nonoverlapping_ownership():
    windows = monolingual_windows([[0, 80000]], 80000)
    assert windows == [(0, 26000, 0, 25000), (24000, 51000, 25000, 50000),
                       (49000, 76000, 50000, 75000), (74000, 80000, 75000, 80000)]


def test_acoustic_pause_preferred_near_target_instead_of_hard_cut():
    windows = monolingual_windows([[0, 19500], [20500, 45000]], 45000)
    assert windows[0] == (0, 20500, 0, 20000)
    assert windows[1] == (19500, 45000, 20000, 45000)


def test_long_silence_does_not_create_empty_asr_jobs():
    speech = [[0, 1000], [100000, 105000]]
    windows = monolingual_windows(speech, 120000)
    assert len(windows) == 2
    assert all(any(s < hi and e > lo for s, e in speech) for a, b, lo, hi in windows)
    assert all(0 <= a < b <= 120000 and b - a <= 27000 for a, b, _, _ in windows)
