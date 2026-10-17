"""Bounded retries for Windows checkpoint replacement; never delete the old file."""
from os import replace as _replace
from time import sleep as _sleep


def replace_checkpoint(temporary, destination):
    """Keep both snapshots on failure and propagate the original OS exception.

    Windows access/sharing violations may be transient (for example a reader
    without delete sharing). A persistent ACL or read-only error still fails
    after five attempts. Non-Windows permission errors and other I/O failures
    are not retried. No permission changes, deletion or copy fallback is used.
    """
    delays = (.05, .1, .2, .4)
    for attempt in range(len(delays) + 1):
        try:
            _replace(temporary, destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(delays):
                raise
            _sleep(delays[attempt])
