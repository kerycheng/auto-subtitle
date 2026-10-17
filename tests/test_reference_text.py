import copy

import pytest

from autosub.domain import Project, Word, Cue, word_key
from autosub.reference_text import read_srt_events, reconstruct_overlap_bridges, compare_text_reference


def event(index, start, end, lines):
    return {'id': index, 'start_ms': start, 'end_ms': end, 'lines': lines,
            'text': '\n'.join(lines), 'source_rows': [index]}


def test_parse_preserves_multiline_bridge_and_source_bytes(tmp_path):
    p = tmp_path / 'reference.srt'
    p.write_text('1\n00:00:01,000 --> 00:00:02,000\n前。\n\n2\n00:00:02,000 --> 00:00:02,007\n次。\n前。\n\n3\n00:00:02,007 --> 00:00:03,000\n次。\n', encoding='utf-8-sig')
    before = p.read_bytes()
    raw = read_srt_events(p)
    original = copy.deepcopy(raw)
    fixed, trace = reconstruct_overlap_bridges(raw)
    assert len(raw) == 3 and len(fixed) == 2 and len(trace) == 1
    assert fixed[0]['end_ms'] == 2007 and fixed[1]['start_ms'] == 2000
    assert fixed[0]['source_rows'] == [1, 2] and fixed[1]['source_rows'] == [2, 3]
    assert raw == original and p.read_bytes() == before
    assert trace[0]['independently_confirmed'] is False


@pytest.mark.parametrize('problem', ['long', 'gap', 'same_reply', 'different_text'])
def test_reference_view_does_not_silently_deduplicate_ordinary_speech(problem):
    rows = [event(1, 1000, 2000, ['前。']), event(2, 2000, 2007, ['次。', '前。']), event(3, 2007, 3000, ['次。'])]
    if problem == 'long':
        rows[1]['end_ms'] = rows[2]['start_ms'] = 2100
    elif problem == 'gap':
        rows[2]['start_ms'] += 1
    elif problem == 'same_reply':
        rows = [event(1, 1000, 2000, ['はい']), event(2, 2000, 2007, ['はい']), event(3, 2007, 3000, ['はい'])]
    else:
        rows[1]['lines'] = ['知らない', '前。']
    fixed, trace = reconstruct_overlap_bridges(rows)
    assert fixed == rows and not trace


@pytest.mark.parametrize('content', ['', '1\n00:01:01,000 --> 00:01:00,000\n文字',
    '1\n00:99:01,000 --> 01:00:00,000\n文字', '1\nnot a clock\n文字'])
def test_invalid_srt_never_silently_skips_rows(tmp_path, content):
    p = tmp_path / 'bad.srt'
    p.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        read_srt_events(p)


def test_prefix_candidate_is_read_only_and_not_acoustic_gold():
    words = [Word(100, 500, '前の文章。'), Word(600, 700, '日本'), Word(700, 1500, 'には島がある。')]
    p = Project(words=words, duration_ms=2000, cues=[
        Cue('a', 100, 700, '前の文章。日本', 100, 700, source_tokens=[word_key(w) for w in words[:2]]),
        Cue('b', 700, 1500, 'には島がある。', 700, 1500, source_tokens=[word_key(words[2])])])
    before = copy.deepcopy(p)
    r = compare_text_reference(p, [event(1, 100, 500, ['前の文章。']), event(2, 600, 1500, ['日本には島がある。'])])
    assert r['mechanical_cer'] == 0
    assert r['prefix_carry_candidates'][0]['prefix'] == '日本'
    assert r['acoustic_timing_accuracy'] is None and p == before


def test_large_reference_is_rejected_before_quadratic_allocation():
    p = Project(words=[Word(0, 1000, 'あ' * 3000)])
    with pytest.raises(ValueError, match='exact-grid limit'):
        compare_text_reference(p, [event(1, 0, 1000, ['あ' * 3000])])


def test_cross_token_unicode_composition_scores_whole_text_but_does_not_guess_ownership():
    p = Project(words=[Word(0, 100, 'か'), Word(100, 200, '\u3099')])
    result = compare_text_reference(p, [event(1, 0, 200, ['が'])])
    assert result['mechanical_cer'] == 0
    assert result['ownership_mapping'].startswith('unrated')


def test_previous_ending_in_next_cue_is_a_separate_direction():
    words = [Word(100, 400, '今日は楽しい'), Word(400, 500, 'な'), Word(500, 900, 'いや次の話です')]
    p = Project(words=words, cues=[Cue('a',100,400,'今日は楽しい',100,400,source_tokens=[word_key(words[0])]),
        Cue('b',400,900,'ないや次の話です',400,900,source_tokens=[word_key(w) for w in words[1:]])])
    result = compare_text_reference(p, [event(1,100,500,['今日は楽しいな']),event(2,500,900,['いや次の話です'])])
    assert result['previous_suffix_carry_candidates'][0]['suffix'] == 'な'
    assert not result['prefix_carry_candidates']


def test_duplicate_zero_length_tokens_keep_text_score_but_ownership_is_unrated():
    p = Project(words=[Word(100,100,'ぇ'),Word(100,100,'ぇ')])
    before=copy.deepcopy(p)
    result=compare_text_reference(p,[event(1,0,200,['ぇぇ'])])
    assert result['mechanical_cer']==0 and p==before
    assert result['ownership_mapping']=='unrated-duplicate-word-identities'
    assert result['ambiguous_word_identities']==['100:100:ぇ']
    assert not result['previous_suffix_carry_candidates'] and not result['prefix_carry_candidates']
