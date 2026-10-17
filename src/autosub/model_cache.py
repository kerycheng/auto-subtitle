"""One resident ASR model for the desktop's sequential recognition jobs."""
from pathlib import Path
from threading import RLock

_lock = RLock()
_key = _model = None


def clear_model():
    global _key, _model
    with _lock:
        _key = _model = None


def get_model(factory, name, device, compute_type, download_root=None):
    global _key, _model
    local = Path(name)
    identity = None
    if local.is_dir():
        identity = tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                         for p in sorted(local.iterdir()) if p.is_file())
    key = (factory, str(local.resolve()) if identity is not None else name,
           device, compute_type, download_root, identity)
    with _lock:
        if key == _key and _model is not None:
            return _model, True
        # Release the previous cache reference before loading another large model.
        _key = _model = None
        loaded = factory(name, device=device, compute_type=compute_type, download_root=download_root)
        _key, _model = key, loaded
        return loaded, False
