import copy
import json
from collections import Counter
from dataclasses import asdict

import pytest

from autosub.domain import Cue, Project, Word, word_key
from autosub.editing import review_cues
from autosub.review_proposals import apply_proposal, digest, proposals, restore_proposal


def sample():
    words=[Word(1000,2000,"声も入っています"),Word(2000,2500,"こちら")]
    p=Project(duration_ms=5000,words=words,speech=[[1000,2500]],asr_strategy="native-v1",
              native_groups=[[0],[1]],native_resume_ms=5000,native_complete=True,status="complete")
    p.cues=[Cue("first",980,2000,words[0].text,1000,2000,source_tokens=[word_key(words[0])]),
            Cue("short",2000,2700,words[1].text,2000,2500,source_tokens=[word_key(words[1])])]
    return p


def merge(p):
    return next(c for c in proposals(p) if c['kind']=='merge')


def test_proposals_are_deterministic_read_only_and_do_not_claim_semantic_verification():
    p=sample()
    before=copy.deepcopy(p)
    first=proposals(p)
    assert first==proposals(p) and p==before
    candidate=merge(p)
    assert candidate['decision']=='unreviewed'
    assert not candidate['evidence']['speaker_verified'] and not candidate['evidence']['semantic_verified']
    assert candidate['preview']['text']=='声も入っていますこちら'
    assert candidate['preview']['source_tokens']==list(map(word_key,p.words))


def test_apply_and_persistent_restore_preserve_text_words_locks_and_history(tmp_path):
    p=sample()
    p.review_events=[{'action':'earlier-user-action'}]
    original=copy.deepcopy(p)
    updated,receipt=apply_proposal(p,merge(p),'automated-test-not-human')
    assert p==original and len(updated.cues)==1
    assert updated.cues[0].locked and not updated.cues[0].reviewed
    assert ''.join(c.text for c in updated.cues)==''.join(c.text for c in p.cues)
    assert updated.words==p.words and updated.native_groups==p.native_groups
    assert Counter(updated.cues[0].source_tokens)==Counter(map(word_key,p.words))
    path=tmp_path/'adopted.autosub.json'
    updated.save(path)
    reloaded=Project.load(path)
    record=json.loads(json.dumps(receipt,ensure_ascii=False))
    restored=restore_proposal(reloaded,record)
    assert [asdict(c) for c in restored.cues]==[asdict(c) for c in original.cues]
    assert restored.review_events[0]==original.review_events[0]
    assert [e['action'] for e in restored.review_events[-2:]]==['apply_merge_proposal','restore_merge_proposal']
    assert len(reloaded.cues)==1
    with pytest.raises(ValueError):
        restore_proposal(restored,record)


@pytest.mark.parametrize('field,value',[
    ('locked',True),('reviewed',True),('edit_group','manual-split'),
    ('candidate_original',{'before':'manual'}),('text','已人工改字'),
])
def test_protected_or_modified_cues_never_offer_merges(field,value):
    p=sample()
    setattr(p.cues[0],field,value)
    assert not any(c['kind']=='merge' for c in proposals(p))


@pytest.mark.parametrize('gap',[-100,301,1500])
def test_overlap_and_large_gap_never_offer_merges(gap):
    p=sample()
    p.cues[1].speech_start_ms=2000+gap
    p.cues[1].speech_end_ms=2500+gap
    p.cues[1].start_ms=2000+gap
    p.cues[1].end_ms=2700+gap
    assert not any(c['kind']=='merge' for c in proposals(p))


def test_terminal_punctuation_not_automatically_joined_to_short_response():
    p=sample()
    p.words[0].text=p.cues[0].text='本当ですか？'
    p.cues[0].source_tokens=[word_key(p.words[0])]
    assert not any(c['kind']=='merge' for c in proposals(p))


def test_character_and_duration_caps_are_respected():
    p=sample()
    p.settings.max_chars=7
    assert not any(c['kind']=='merge' for c in proposals(p))
    p.settings.max_chars=42
    p.settings.max_duration_ms=1500
    p.settings.soft_duration_ms=1000
    assert not any(c['kind']=='merge' for c in proposals(p))


def test_duplicate_or_missing_word_keys_are_listen_only():
    p=sample()
    p.words.append(copy.deepcopy(p.words[1]))
    p.native_groups[1].append(2)
    assert not any(c['kind']=='merge' for c in proposals(p))
    p=sample()
    p.cues[1].source_tokens=None
    assert not any(c['kind']=='merge' for c in proposals(p))


def test_nonadjacent_source_overlap_prevents_merge_preview():
    p=sample()
    p.cues.append(Cue('overlap',2300,3000,'別人',2300,2800,source_tokens=[]))
    assert not any(c['kind']=='merge' for c in proposals(p))


@pytest.mark.parametrize('change', ['text','time','source','words','settings','reviewed'])
def test_stale_candidates_cannot_overwrite_changes(change):
    p=sample()
    candidate=merge(p)
    if change=='text': p.cues[0].text='人工修正'
    elif change=='time': p.cues[0].start_ms-=10
    elif change=='source': p.source_fingerprint={'size':100}
    elif change=='words': p.words[0].probability=.3
    elif change=='settings': p.settings.tail_ms=100
    else: p.cues[0].reviewed=True
    before=copy.deepcopy(p)
    with pytest.raises(ValueError,match='過期'):
        apply_proposal(p,candidate,'test')
    assert p==before


def test_tampered_preview_and_listen_items_cannot_be_applied():
    p=sample()
    candidate=merge(p)
    candidate['preview']['text']='不屬於音訊的文字'
    with pytest.raises(ValueError,match='改變'):
        apply_proposal(p,candidate,'test')
    listen=next(c for c in proposals(p) if c['kind']=='listen')
    with pytest.raises(ValueError):
        apply_proposal(p,listen,'test')
    with pytest.raises(ValueError,match='紀錄名稱'):
        apply_proposal(p,merge(p),' ')


def test_weak_voice_and_uncovered_speech_only_produce_listening_evidence():
    p=sample()
    p.speech=[[3000,4500]]
    before=copy.deepcopy(p)
    items=proposals(p)
    assert any(c['kind']=='listen-gap' and c['range_ms']==[3000,4500] for c in items)
    assert any(c['kind']=='listen' and c['evidence']['vad_coverage']==0 for c in items)
    assert all(not c['adoptable'] for c in items if c['kind']!='merge')
    assert not any(c['kind']=='merge' for c in items)
    assert p==before


def test_corrupt_receipt_and_later_edits_cannot_be_overwritten():
    p=sample()
    updated,receipt=apply_proposal(p,merge(p),'test')
    updated.cues[0].text+='人工校正'
    before=copy.deepcopy(updated)
    with pytest.raises(ValueError,match='後續修正'):
        restore_proposal(updated,receipt)
    assert updated==before
    updated,receipt=apply_proposal(p,merge(p),'test')
    receipt['before_cues'][0]['text']='不正確的備份'
    with pytest.raises(ValueError,match='損壞'):
        restore_proposal(updated,receipt)
    receipt['sha256']=digest({k:v for k,v in receipt.items() if k!='sha256'})
    with pytest.raises(ValueError,match='備份不符'):
        restore_proposal(updated,receipt)


@pytest.mark.parametrize('kind',['legacy','incomplete'])
def test_missing_native_provenance_is_not_guessed(kind):
    p=sample()
    if kind=='legacy':
        p.asr_strategy='fixed30-v1'
        p.native_groups=[]
        p.native_complete=False
        p.native_resume_ms=0
    else:
        p.native_complete=False
    with pytest.raises(ValueError,match='已完成'):
        proposals(p)


def test_overlapping_token_ownership_is_not_deduplicated_into_a_merge():
    p=sample()
    p.cues.append(Cue('shared',4000,4500,'別の行',4000,4500,source_tokens=[word_key(p.words[0])]))
    assert not any(c['kind']=='merge' for c in proposals(p))


def test_literal_whitespace_is_not_silently_trimmed_by_a_proposal():
    p=sample()
    p.cues[0].text+=' '
    assert not any(c['kind']=='merge' for c in proposals(p))


def test_two_independent_merges_can_be_restored_in_reverse_order():
    p=sample()
    for i,(a,b,text) in enumerate([(3000,3500,'次の話'),(3500,4000,'続き')],2):
        p.words.append(Word(a,b,text))
        p.native_groups.append([i])
        p.cues.append(Cue(str(i),a,b,text,a,b,source_tokens=[word_key(p.words[-1])]))
    p.speech.append([3000,4000])
    original=copy.deepcopy(p)
    first,r1=apply_proposal(p,merge(p),'test')
    second,r2=apply_proposal(first,merge(first),'test')
    with pytest.raises(ValueError,match='後續修正'):
        restore_proposal(second,r1)
    restored=restore_proposal(restore_proposal(second,r2),r1)
    assert restored.cues==original.cues and restored.words==original.words


def test_normal_ui_warning_refresh_does_not_invalidate_the_receipt():
    p=sample()
    # A stale dynamic flag in the input is removed at adoption, just as in the UI.
    p.cues[0].flags=['時間重疊，請複核邊界／多人發話']
    updated,receipt=apply_proposal(p,merge(p),'test')
    review_cues(updated)
    assert restore_proposal(updated,receipt).cues==p.cues
