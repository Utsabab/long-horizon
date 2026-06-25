import pytest
from experiments.harness import AcceptedLocalErrorDetector

def test_detector_triggers():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('open door', 0)
    d.add('open door', 0)
    d.add('open door', 0)
    assert d.check()

def test_detector_no_trigger_on_progress():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('open door', 0)
    d.add('open door', 0)
    d.add('open door', 1)
    assert not d.check()
