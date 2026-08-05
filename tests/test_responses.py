"""
Tests for piazza_bot response types (Answer, Followup).

These lightweight value types are used by ``bot.respond_to_post`` to decide
whether a handler result should be posted as an answer or a followup, and to
format the text that gets published back to Piazza. They had no coverage.
"""

from src.piazza_bot.responses import Answer, Followup


class TestAnswer:
    """Tests for the Answer response type."""

    def test_stores_text(self):
        answer = Answer("The exam is on Friday.")
        assert answer.text == "The exam is on Friday."

    def test_str_returns_raw_text(self):
        answer = Answer("Office hours are at 2 PM.")
        assert str(answer) == "Office hours are at 2 PM."

    def test_get_formatted_text_prefixes_answer(self):
        answer = Answer("Use the online portal.")
        assert answer.get_formatted_text() == "Answer: Use the online portal."

    def test_handles_empty_text(self):
        answer = Answer("")
        assert str(answer) == ""
        assert answer.get_formatted_text() == "Answer: "

    def test_text_is_used_for_joining(self):
        # bot.respond_to_post joins answers via ``answer.text``.
        answers = [Answer("First."), Answer("Second.")]
        joined = "<p></p><p>---</p><p></p>".join(a.text for a in answers)
        assert joined == "First.<p></p><p>---</p><p></p>Second."


class TestFollowup:
    """Tests for the Followup response type."""

    def test_stores_text(self):
        followup = Followup("Could you clarify the deadline?")
        assert followup.text == "Could you clarify the deadline?"

    def test_str_returns_raw_text(self):
        followup = Followup("See the syllabus.")
        assert str(followup) == "See the syllabus."

    def test_get_formatted_text_prefixes_followup(self):
        followup = Followup("Please rephrase.")
        assert followup.get_formatted_text() == "Followup: Please rephrase."

    def test_wrapping_raw_string_in_followup(self):
        # bot.respond_to_post wraps non-Answer/Followup handler results in a
        # Followup, then posts ``followup.text``.
        raw = "I am not sure, escalating to staff."
        followup = Followup(raw)
        assert followup.text == raw


class TestResponseTypeDiscrimination:
    """Answer and Followup must be distinguishable via isinstance.

    bot.respond_to_post routes handler results by ``isinstance`` checks, so the
    two types must not be interchangeable.
    """

    def test_answer_is_not_followup(self):
        assert not isinstance(Answer("x"), Followup)

    def test_followup_is_not_answer(self):
        assert not isinstance(Followup("x"), Answer)

    def test_routing_by_isinstance(self):
        responses = [Answer("a1"), Followup("f1"), Answer("a2")]
        answers = [r for r in responses if isinstance(r, Answer)]
        followups = [r for r in responses if isinstance(r, Followup)]
        assert [a.text for a in answers] == ["a1", "a2"]
        assert [f.text for f in followups] == ["f1"]
