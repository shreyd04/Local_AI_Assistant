import pytest

from app.structured_output import (
    StructuredOutputError,
    generate_structured_response,
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content):
        self.message = FakeMessage(content)


def test_valid_structured_output(monkeypatch):

    valid_json = """
    {
        "answer": "A database index improves lookup performance.",
        "key_points": [
            "Faster lookups",
            "Reduced scanning",
            "Query optimization"
        ],
        "category": "explanation"
    }
    """

    calls = []

    def fake_chat(**kwargs):

        calls.append(kwargs)

        return FakeResponse(valid_json)

    monkeypatch.setattr(
        "app.structured_output.client.chat",
        fake_chat,
    )

    result = generate_structured_response(
        "Explain database indexes.",
        max_retries=1,
    )

    assert result.attempts == 1
    assert (
        result.data.category
        == "explanation"
    )

    assert len(calls) == 1


def test_retry_after_invalid_output(
    monkeypatch,
):

    invalid_json = """
    {
        "answer": "Test",
        "key_points": [],
        "category": "invalid_category"
    }
    """

    valid_json = """
    {
        "answer": "A process has its own address space.",
        "key_points": [
            "Independent memory",
            "Separate resources"
        ],
        "category": "comparison"
    }
    """

    responses = [
        FakeResponse(invalid_json),
        FakeResponse(valid_json),
    ]

    calls = []

    def fake_chat(**kwargs):

        calls.append(kwargs)

        return responses.pop(0)

    monkeypatch.setattr(
        "app.structured_output.client.chat",
        fake_chat,
    )

    result = generate_structured_response(
        "Compare processes and threads.",
        max_retries=1,
    )

    assert result.attempts == 2
    assert (
        result.data.category
        == "comparison"
    )

    assert len(calls) == 2


def test_graceful_failure_after_retry(
    monkeypatch,
):

    invalid_json = """
    {
        "answer": "Test",
        "key_points": [],
        "category": "invalid_category"
    }
    """

    calls = []

    def fake_chat(**kwargs):

        calls.append(kwargs)

        return FakeResponse(invalid_json)

    monkeypatch.setattr(
        "app.structured_output.client.chat",
        fake_chat,
    )

    with pytest.raises(
        StructuredOutputError
    ):
        generate_structured_response(
            "Test invalid output.",
            max_retries=1,
        )

    assert len(calls) == 2