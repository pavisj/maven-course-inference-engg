from control.overflow import OverflowLimiter, classify


def test_classify_stay_and_leave():
    assert classify(200) == "ok"
    assert classify(429, b'{"reason": "over_cap"}') == "stay"
    assert classify(500, b"Internal Server Error") == "stay"
    assert classify(503, b"Service Unavailable") == "leave"
    assert classify(529, b"overloaded") == "leave"
    assert classify(400, b"bad request") == "other"


def test_out_of_memory_text_stays_whatever_the_code():
    assert classify(503, b"CUDA error: out of memory") == "stay"
    assert classify(400, "torch.OutOfMemoryError: tried to allocate") == "stay"
    assert classify(503, b"slice_oom") == "stay"
    assert classify(503, b"no room at the inn") == "leave"   # "room" is not an OOM


def test_limiter_ten_per_minute():
    lim = OverflowLimiter()
    assert all(lim.allow(now=100.0 + i) for i in range(10))
    assert lim.allow(now=110.0) is False              # the 11th inside the minute
    assert lim.allow(now=160.5) is True               # the first one has left the window
