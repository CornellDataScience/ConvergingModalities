from language_eval import windows
import pytest


@pytest.mark.parametrize('count', [1, 255, 256, 257, 511, 512, 1000])
def test_each_target_scored_once_with_causal_context(count):
    targets = []
    for begin, start, end in windows(count):
        assert 0 <= begin < start < end <= count+1
        assert start-begin <= 256
        assert end-start <= 256
        targets.extend(range(start, end))
    assert targets == list(range(1, count+1))
