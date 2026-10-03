import random

import pytest

from control.place import CHAPTER_HOME, PlaceRequest, Placement, Shed, WorkerView, pick


def views(a_in=0, b_in=0, a_q=0, b_q=0, a_warm=True, b_warm=True, a_ok=True, b_ok=True, a_wait=0, b_wait=0):
    return {
        "WorkerA": WorkerView("WorkerA", a_warm, a_ok, a_in, a_q, a_wait),
        "WorkerB": WorkerView("WorkerB", b_warm, b_ok, b_in, b_q, b_wait),
    }


def req(chapter):
    return PlaceRequest("interactive", chapter)


def test_chapter_map():
    for ch in ("00", "03a", "03b", "05"):
        assert CHAPTER_HOME[ch] == "WorkerA"
    for ch in ("06", "06b", "07", "09", "10"):
        assert CHAPTER_HOME[ch] == "WorkerB"


def test_prefix_home_when_under_bound():
    assert pick(req("07"), views(b_in=21)) == Placement("WorkerB", "WorkerB", False, "prefix_then_load")
    assert pick(req("03a"), views(a_in=10, b_in=0)).chosen == "WorkerA"   # home wins over a lighter worker


def test_hop_when_home_at_load_bound():
    p = pick(req("07"), views(a_in=5, b_in=22))
    assert (p.preferred, p.chosen, p.hopped) == ("WorkerB", "WorkerA", True)


def test_stay_home_when_the_other_worker_is_busier():
    p = pick(req("07"), views(a_in=30, b_in=22))
    assert (p.chosen, p.hopped) == ("WorkerB", False)


def test_queue_depth_is_a_scorer():
    p = pick(req("07"), views(a_in=10, a_q=20, b_in=22, b_q=0))   # A: 30, B: 22
    assert p.chosen == "WorkerB"


def test_not_warm_is_not_usable():
    p = pick(req("07"), views(b_warm=False))
    assert (p.preferred, p.chosen, p.hopped) == ("WorkerB", "WorkerA", True)


def test_no_warm_worker_sheds_503():
    assert pick(req("07"), views(a_warm=False, b_ok=False)) == Shed(503, "no_warm_worker")


def test_unknown_chapter_goes_to_least_loaded_without_a_hop():
    p = pick(req(None), views(a_in=9, b_in=3))
    assert (p.preferred, p.chosen, p.hopped) == ("WorkerB", "WorkerB", False)


def test_least_loaded_and_p2c():
    assert pick(req("03a"), views(a_in=9, b_in=3), policy="least_loaded").chosen == "WorkerB"
    assert pick(req("03a"), views(a_in=9, b_in=3), policy="p2c", rng=random.Random(1)).chosen == "WorkerB"


def test_unknown_policy():
    with pytest.raises(ValueError):
        pick(req("07"), views(), policy="round_robin")
