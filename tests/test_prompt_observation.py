from types import SimpleNamespace
import pytest
from autosub.prompt_observation import observe_prompts, token_digest


class Tokenizer:
    def encode(self, text):
        return list(text.encode('ascii'))


class Model:
    def get_prompt(self, tokenizer, previous_tokens, hotwords=None):
        return ([1] + tokenizer.encode(' ' + hotwords) if hotwords else []) + previous_tokens + [2]

    def generate_with_fallback(self, encoder_output, prompt, tokenizer, options):
        return ('unaltered result', 0., .4, 1.)


def test_observation_is_transparent_and_records_prompt_actually_passed_to_decoder():
    model = Model(); tokenizer = Tokenizer()
    with observe_prompts(model, 'example') as rows:
        prompt = model.get_prompt(tokenizer, [], hotwords='example')
        result = model.generate_with_fallback(None, prompt, tokenizer, SimpleNamespace())
    assert result == ('unaltered result', 0., .4, 1.)
    assert [r['call'] for r in rows] == ['get_prompt', 'generate_with_fallback']
    assert all(r['full_vocabulary_present'] for r in rows)
    assert rows[0]['prompt_sha256'] == rows[1]['prompt_sha256'] == token_digest(prompt)
    assert rows[1]['returned_temperature'] == .4 and rows[1]['rejected_fallback_attempts'] is None
    assert 'get_prompt' not in vars(model) and 'generate_with_fallback' not in vars(model)


def test_reset_initial_tokens_are_observed_without_inventing_persistent_hint():
    model = Model(); tokenizer = Tokenizer()
    with observe_prompts(model, 'example') as rows:
        model.get_prompt(tokenizer, tokenizer.encode(' example'))
        model.get_prompt(tokenizer, [])
    assert [r['full_vocabulary_present'] for r in rows] == [True, False]


def test_failure_restores_existing_instance_overrides_and_keeps_failed_observation():
    model = Model()
    def fail(tokenizer, previous_tokens, hotwords=None):
        raise RuntimeError('synthetic failure')
    model.get_prompt = fail
    with pytest.raises(RuntimeError):
        with observe_prompts(model, 'example') as rows:
            model.get_prompt(Tokenizer(), [])
    assert model.get_prompt is fail and rows[0]['status'] == 'failed'
    assert rows[0]['error_type'] == 'RuntimeError'
    assert 'generate_with_fallback' not in vars(model)


@pytest.mark.parametrize('hint', ['', ' ', None])
def test_rejects_empty_observation_hint(hint):
    with pytest.raises(ValueError):
        with observe_prompts(Model(), hint):
            pass
