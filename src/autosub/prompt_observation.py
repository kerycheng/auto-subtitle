"""Read-only SDK prompt observations for isolated, serial ASR diagnostics.

Not imported by the application pipeline. No tokens or model outputs are changed.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json


def token_digest(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode('ascii')).hexdigest()


def contains_tokens(tokens, fragment):
    if not fragment:
        return False
    return any(tokens[i:i + len(fragment)] == fragment for i in range(len(tokens) - len(fragment) + 1))


@contextmanager
def observe_prompts(model, vocabulary):
    """Observe actual builder and decoder calls; restore instance attributes on exit.

    Each yielded list belongs to one transcription. Serial use only. A decoder
    call may try several fallback temperatures internally; those attempts are
    not exposed or inferred by this observer.
    """
    if not isinstance(vocabulary, str) or not vocabulary.strip():
        raise ValueError('An explicit nonempty vocabulary is required')
    observations = []
    originals = {}
    try:
        for name in ('get_prompt', 'generate_with_fallback'):
            method = getattr(model, name)
            originals[name] = (name in vars(model), vars(model).get(name), method)
            signature = inspect.signature(method)

            def wrapper(*args, _name=name, _method=method, _signature=signature, **kwargs):
                bound = _signature.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = bound.arguments
                tokenizer = arguments['tokenizer']
                hint = list(tokenizer.encode(' ' + vocabulary.strip()))
                row = {'call': _name, 'index': len(observations), 'status': 'running'}
                observations.append(row)
                try:
                    if _name == 'get_prompt':
                        result = _method(*args, **kwargs)
                        tokens = list(result)
                        row['previous_tokens_count'] = len(arguments['previous_tokens'])
                        row['hotwords_requested'] = bool(arguments.get('hotwords'))
                    else:
                        tokens = list(arguments['prompt'])
                        result = _method(*args, **kwargs)
                        row['returned_temperature'] = result[2]
                        row['rejected_fallback_attempts'] = None
                    row.update(status='complete', prompt_sha256=token_digest(tokens),
                               prompt_tokens_count=len(tokens), vocabulary_tokens_count=len(hint),
                               full_vocabulary_present=contains_tokens(tokens, hint))
                    return result
                except BaseException as exc:
                    row.update(status='failed', error_type=type(exc).__name__)
                    raise

            setattr(model, name, wrapper)
        yield observations
    finally:
        for name, (had_instance_value, value, _) in originals.items():
            if had_instance_value:
                setattr(model, name, value)
            else:
                delattr(model, name)
