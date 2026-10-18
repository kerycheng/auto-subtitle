import json
from types import SimpleNamespace

import pytest
import numpy as np

from autosub.asr_observation import snapshot_hypothesis


def test_sdk_timestamp_variant_does_not_lose_text_or_invent_alias():
    hyp = SimpleNamespace(text="a blue window", score=-2.0, y_sequence=[5, 8], timestamp=[1, 4])
    record = snapshot_hypothesis(hyp)
    assert record["text"] == hyp.text
    assert record["observations"]["timestamp"] == [1, 4]
    assert "timestep" not in record["observations"]
    assert record["observation_errors"] == {}
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert hyp.timestamp == [1, 4]


def test_empty_decoded_text_is_valid_and_not_filled():
    record = snapshot_hypothesis(SimpleNamespace(text=""))
    assert record["text"] == "" and record["observations"] == {}


def test_optional_property_failure_keeps_mandatory_text():
    class Result:
        text = "bright paper"

        @property
        def score(self):
            raise RuntimeError("Unavailable")

    record = snapshot_hypothesis(Result())
    assert record["text"] == "bright paper"
    assert record["observation_errors"] == {"score": "RuntimeError"}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), [0] * 10001,
                                  {1: "unsupported key"}])
def test_unrecordable_optional_data_never_discards_text(value):
    record = snapshot_hypothesis(SimpleNamespace(text="glass bird", score=value))
    assert record["text"] == "glass bird"
    assert record["observations"] == {}
    assert record["observation_errors"] == {"score": "ValueError"}
    json.dumps(record, allow_nan=False)


def test_recursive_optional_data_is_bounded():
    values = []
    values.append(values)
    record = snapshot_hypothesis(SimpleNamespace(text="green leaf", tokens=values))
    assert record["text"] == "green leaf"
    assert record["observation_errors"] == {"tokens": "ValueError"}


def test_large_array_rejected_before_materializing_list():
    class LargeArray:
        size = 1000000

        def tolist(self):
            raise AssertionError("Should not allocate")

    record = snapshot_hypothesis(SimpleNamespace(text="linen", tokens=LargeArray()))
    assert record["text"] == "linen"
    assert record["observation_errors"] == {"tokens": "ValueError"}


def test_tensor_copy_failure_preserves_text():
    class Tensor:
        def numel(self):
            return 2

        def detach(self):
            raise RuntimeError("Device unavailable")

    record = snapshot_hypothesis(SimpleNamespace(text="cloud", y_sequence=Tensor()))
    assert record["text"] == "cloud"
    assert record["observation_errors"] == {"y_sequence": "RuntimeError"}


def test_cpu_tensor_data_copied_without_tensor_dependency():
    class Tensor:
        def numel(self):
            return 2

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [7, 9]

    record = snapshot_hypothesis(SimpleNamespace(text="cloud", y_sequence=Tensor()))
    assert record["observations"]["y_sequence"] == [7, 9]


def test_numpy_observations_are_json_safe_and_do_not_alias_input():
    values = np.array([3, 4], dtype=np.int64)
    record = snapshot_hypothesis(SimpleNamespace(text="paper", y_sequence=values,
                                                score=np.float32(-0.5)))
    values[0] = 99
    assert record["observations"] == {"y_sequence": [3, 4], "score": -0.5}
    assert json.loads(json.dumps(record, allow_nan=False)) == record


@pytest.mark.parametrize("hyp", [SimpleNamespace(), SimpleNamespace(text=None),
                                 SimpleNamespace(text=123)])
def test_missing_mandatory_text_is_not_silently_empty(hyp):
    with pytest.raises(ValueError):
        snapshot_hypothesis(hyp)
