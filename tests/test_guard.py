from control.guard import MAX_MODEL_LEN, inspect


def msgs(user_text="What is a KV cache?"):
    return [{"role": "system", "content": "You are Tutor."}, {"role": "user", "content": user_text}]


def test_ok_and_counts_prompt_tokens():
    g = inspect({"messages": msgs(), "max_tokens": 512}, "interactive")
    assert (g.ok, g.code, g.reason) == (True, 200, "ok")
    assert 10 < g.prompt_tokens < 60


def test_missing_max_tokens_gets_tenant_budget():
    assert inspect({"messages": msgs()}, "interactive").max_tokens == 512
    assert inspect({"messages": msgs()}, "batch").max_tokens == 256


def test_bad_body():
    assert inspect({"max_tokens": 10}, "interactive").reason == "bad_body"
    assert inspect({"messages": []}, "interactive").reason == "bad_body"
    assert inspect({"messages": [{"role": "user"}]}, "interactive").reason == "bad_body"
    assert inspect({"messages": msgs(), "max_tokens": "512"}, "interactive").reason == "bad_body"
    assert inspect({"messages": msgs(), "max_tokens": 0}, "interactive").code == 400


def test_unknown_tenant():
    for tenant in (None, "", "admin"):
        g = inspect({"messages": msgs(), "max_tokens": 100}, tenant)
        assert (g.ok, g.code, g.reason) == (False, 400, "unknown_tenant")


def test_prompt_too_long():
    long_text = "token " * MAX_MODEL_LEN
    g = inspect({"messages": msgs(long_text), "max_tokens": 256}, "batch")
    assert (g.ok, g.code, g.reason) == (False, 400, "prompt_too_long")
    assert g.prompt_tokens + g.max_tokens > MAX_MODEL_LEN


def test_max_tokens_too_large():
    assert inspect({"messages": msgs(), "max_tokens": 513}, "interactive").reason == "max_tokens_too_large"
    assert inspect({"messages": msgs(), "max_tokens": 257}, "batch").reason == "max_tokens_too_large"
    assert inspect({"messages": msgs(), "max_tokens": 256}, "batch").ok


def test_unknown_model():
    msgs = [{"role": "user", "content": "What is TTFT?"}]
    assert inspect({"model": "gpt-4o", "messages": msgs}, "interactive").reason == "unknown_model"
    assert inspect({"model": "tutor", "messages": msgs}, "interactive").ok
    assert inspect({"messages": msgs}, "interactive").ok          # no model: the gateway serves the tutor model
