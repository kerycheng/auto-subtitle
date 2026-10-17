from autosub.domain import Project, Word, Cue, word_key
from autosub.segmentation_audit import audit, uncovered_speech


def test_uncovered_vad_unions_words_and_speech_without_double_counting():
    p=Project(duration_ms=10000,speech=[[1000,4000],[3000,5000],[7000,9500]],words=[
        Word(1500,2000,"A"),Word(1800,3000,"B"),Word(5000,8000,"C"),Word(9000,9000,"D")])
    assert uncovered_speech(p)==[[3000,5000],[8000,9500]]
    assert uncovered_speech(p,minimum_ms=500)==[[1000,1500],[3000,5000],[8000,9500]]


def test_audit_distinguishes_known_word_cuts_and_preserves_token_ownership():
    words=[Word(100,300,"東"),Word(300,500,"京駅"),Word(1000,1500,"です。")]
    p=Project(duration_ms=3000,words=words)
    p.cues=[Cue("a",100,300,"東",100,300,source_tokens=[word_key(words[0])]),
            Cue("b",300,1500,"京駅です。",300,1500,source_tokens=list(map(word_key,words[1:])))]
    result=audit(p)
    assert result['known_word_internal_cuts']==1 and result['cuts'][0]['lexeme']=="東京駅"
    assert result['source_tokens_preserved'] and result['human_completion_rate'] is None


def test_audit_reports_ambiguous_duplicate_keys_and_all_nested_overlap_pairs():
    words=[Word(100,100,"字"),Word(100,100,"字")]
    p=Project(duration_ms=4000,words=words,cues=[
        Cue("long",0,3000,"字",0,3000,source_tokens=[word_key(words[0])]),
        Cue("short",500,1000,"字",500,1000,source_tokens=[word_key(words[1])]),
        Cue("late",2000,2500,"追加",2000,2500,source_tokens=[])])
    result=audit(p)
    assert result['duplicate_source_keys']==1
    assert result['ambiguous_cut_cues']==['long','short','late']
    assert result['overlap_pairs']==[['long','short'],['long','late']]
    assert result['source_tokens_preserved']


def test_empty_and_zero_duration_words_do_not_hide_vad_audition_targets():
    p=Project(duration_ms=2000,speech=[[0,2000]],words=[Word(100,100,"声")])
    assert uncovered_speech(p)==[[0,2000]]
