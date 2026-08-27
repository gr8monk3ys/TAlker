"""
Tests for src/piazza_bot/bot.py (PiazzaBot).

PiazzaBot is constructed by injecting a `network` object directly (the
constructor never performs a real Piazza login), so a MagicMock() stands in
for the network in every test below. `create_bot` (which does perform a real
login) is intentionally not exercised.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, call

import pandas as pd
import piazza_api
import pytest

from src.piazza_bot.bot import DATA_DIR, PiazzaBot
from src.piazza_bot.responses import Answer, Followup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def network():
    """A MagicMock network with a sane default user profile."""
    net = MagicMock()
    net.get_user_profile.return_value = {"user_id": "instructor_1"}
    return net


@pytest.fixture
def bot(network):
    return PiazzaBot(network)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Prevent tests from actually blocking on time.sleep."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def make_post(
    nr=1,
    uid_a="student_1",
    content="Hello world",
    status="unresolved",
    created="2024-01-01T00:00:00Z",
    bucket_name="",
    is_announcement=0,
    children=None,
):
    return {
        "nr": nr,
        "history": [{"content": content, "uid_a": uid_a}],
        "status": status,
        "created": created,
        "bucket_name": bucket_name,
        "config": {"is_announcement": is_announcement},
        "children": children if children is not None else [],
    }


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_network_and_profile(self, network):
        bot = PiazzaBot(network)
        assert bot.network is network
        network.get_user_profile.assert_called_once_with()
        assert bot.user_profile == {"user_id": "instructor_1"}

    def test_starts_with_no_handlers(self, bot):
        assert bot.post_handlers == []

    def test_starts_with_empty_dataframe(self, bot):
        assert list(bot.df.columns) == [
            "username",
            "content",
            "post_id",
            "status",
            "timestamp",
        ]
        assert len(bot.df) == 0


# ---------------------------------------------------------------------------
# get_posts
# ---------------------------------------------------------------------------


class TestGetPosts:
    def test_stops_when_no_more_posts(self, bot, network):
        network.iter_all_posts.side_effect = [["post1", "post2"], []]

        result = bot.get_posts(time_limit=3600)

        assert result == ["post1", "post2"]
        assert network.iter_all_posts.call_count == 2
        network.iter_all_posts.assert_called_with(limit=PiazzaBot.POST_LOOKBACK_LIMIT)

    def test_accumulates_multiple_batches(self, bot, network):
        network.iter_all_posts.side_effect = [["a"], ["b", "c"], []]

        result = bot.get_posts(time_limit=3600)

        assert result == ["a", "b", "c"]

    def test_stops_when_time_limit_already_elapsed(self, bot, network, monkeypatch):
        # First call to time.time() sets start_time, second call (top of loop)
        # reports elapsed time greater than the limit -> break before ever
        # touching the network.
        times = iter([100.0, 200.0])
        monkeypatch.setattr(time, "time", lambda: next(times))

        result = bot.get_posts(time_limit=5)

        assert result == []
        network.iter_all_posts.assert_not_called()

    @pytest.mark.parametrize(
        "exc",
        [
            piazza_api.exceptions.RequestError("boom"),
            piazza_api.exceptions.AuthenticationError("nope"),
            ConnectionError("dropped"),
        ],
    )
    def test_retries_with_backoff_on_recoverable_errors(
        self, bot, network, exc, monkeypatch
    ):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

        network.iter_all_posts.side_effect = [exc, exc, []]

        result = bot.get_posts(time_limit=3600)

        assert result == []
        # Two failures then a clean empty batch.
        assert network.iter_all_posts.call_count == 3
        # backoff starts at 4 and doubles each failure: 4, then 8.
        assert sleeps == [4, 8]

    def test_backoff_reset_after_success(self, bot, network, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
        exc = ConnectionError("dropped")

        # fail once, succeed, fail once again -> backoff should have reset
        # back to 4 after the successful batch, not continued doubling to 16.
        # (A successful, non-empty batch also incurs the fixed 2s pacing
        # sleep at the bottom of the loop, so the middle entry is 2, not a
        # backoff value.)
        network.iter_all_posts.side_effect = [exc, ["post"], exc, []]

        result = bot.get_posts(time_limit=3600)

        assert result == ["post"]
        assert sleeps == [4, 2, 4]

    def test_breaks_when_backoff_exceeds_time_limit(self, bot, network, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
        network.iter_all_posts.side_effect = ConnectionError("always fails")

        result = bot.get_posts(time_limit=5)

        assert result == []
        # backoff starts at 4 (sleep(4) happens), then becomes 8 which is >
        # time_limit(5), so the loop breaks without a second sleep.
        assert sleeps == [4]
        assert network.iter_all_posts.call_count == 1


# ---------------------------------------------------------------------------
# get_user_info
# ---------------------------------------------------------------------------


class TestGetUserInfo:
    def test_returns_parsed_info(self, bot, network):
        network.get_users.return_value = [{"name": "Alice"}]
        post = make_post(
            nr=42, uid_a="uid-1", content="hi", status="unresolved", created="ts-1"
        )

        info = bot.get_user_info(post)

        assert info == {
            "username": "Alice",
            "content": "hi",
            "post_id": 42,
            "status": "unresolved",
            "timestamp": "ts-1",
        }
        network.get_users.assert_called_once_with(["uid-1"])

    def test_missing_uid_a_returns_none(self, bot, network):
        post = make_post(nr=7)
        del post["history"][0]["uid_a"]

        result = bot.get_user_info(post)

        assert result is None
        network.get_users.assert_not_called()

    def test_falls_back_to_anonymous_when_no_user_info(self, bot, network):
        network.get_users.return_value = []
        post = make_post(uid_a="uid-2")

        info = bot.get_user_info(post)

        assert info["username"] == "Anonymous"


# ---------------------------------------------------------------------------
# already_answered / should_skip_post
# ---------------------------------------------------------------------------


class TestAlreadyAnswered:
    def test_no_children_returns_false(self, bot):
        post = make_post(children=[])
        assert bot.already_answered(post) is False

    def test_matches_instructor_uid_returns_true(self, bot, network):
        network.get_user_profile.return_value = {"user_id": "instructor_1"}
        bot = PiazzaBot(network)
        post = make_post(children=[{"uid": "someone_else"}, {"uid": "instructor_1"}])

        assert bot.already_answered(post) is True

    def test_no_matching_child_returns_false(self, bot):
        post = make_post(children=[{"uid": "someone_else"}])
        assert bot.already_answered(post) is False


class TestShouldSkipPost:
    def test_skips_pinned_posts(self, bot):
        post = make_post(bucket_name="Pinned")
        assert bot.should_skip_post(post) is True

    def test_skips_announcements(self, bot):
        post = make_post(is_announcement=1)
        assert bot.should_skip_post(post) is True

    def test_skips_already_answered_posts(self, bot, network):
        network.get_user_profile.return_value = {"user_id": "instructor_1"}
        bot = PiazzaBot(network)
        post = make_post(children=[{"uid": "instructor_1"}])
        assert bot.should_skip_post(post) is True

    def test_does_not_skip_normal_unanswered_post(self, bot):
        post = make_post(bucket_name="", is_announcement=0, children=[])
        assert bot.should_skip_post(post) is False


# ---------------------------------------------------------------------------
# post_answer / post_followup / post_response
# ---------------------------------------------------------------------------


class TestPostResponse:
    def test_post_answer_calls_network(self, bot, network):
        fake_post = MagicMock()
        network.get_post.return_value = fake_post

        bot.post_answer(99, "the answer")

        network.get_post.assert_called_once_with(99)
        fake_post.create_instructor_answer.assert_called_once_with(
            "the answer", revision=0
        )

    def test_post_followup_calls_network(self, bot, network):
        fake_post = MagicMock()
        network.get_post.return_value = fake_post

        bot.post_followup(99, "the followup")

        network.get_post.assert_called_once_with(99)
        fake_post.create_followup.assert_called_once_with("the followup")

    def test_dispatches_answer(self, bot, network):
        fake_post = MagicMock()
        network.get_post.return_value = fake_post

        bot.post_response(5, Answer("a"))

        fake_post.create_instructor_answer.assert_called_once_with("a", revision=0)
        fake_post.create_followup.assert_not_called()

    def test_dispatches_followup(self, bot, network):
        fake_post = MagicMock()
        network.get_post.return_value = fake_post

        bot.post_response(5, Followup("f"))

        fake_post.create_followup.assert_called_once_with("f")
        fake_post.create_instructor_answer.assert_not_called()

    def test_unrecognized_type_is_a_noop(self, bot, network):
        # A plain string is neither Answer nor Followup -> nothing should
        # happen, and it must not raise.
        bot.post_response(5, "just a string")

        network.get_post.assert_not_called()


# ---------------------------------------------------------------------------
# respond_to_post
# ---------------------------------------------------------------------------


class TestRespondToPost:
    def _post(self, network, nr=1, uid_a="student_1"):
        network.get_users.return_value = [{"name": "Bob"}]
        return make_post(nr=nr, uid_a=uid_a)

    def test_none_responses_are_filtered_out(self, bot, network):
        post = self._post(network)
        handler = MagicMock(return_value=None)
        bot.register_post_handler(handler)
        fake_post_obj = MagicMock()
        network.get_post.return_value = fake_post_obj

        bot.respond_to_post(post)

        handler.assert_called_once()
        fake_post_obj.create_instructor_answer.assert_not_called()
        fake_post_obj.create_followup.assert_not_called()

    def test_plain_return_value_is_wrapped_in_followup(self, bot, network):
        post = self._post(network)
        bot.register_post_handler(lambda info: "plain text reply")
        fake_post_obj = MagicMock()
        network.get_post.return_value = fake_post_obj

        bot.respond_to_post(post)

        fake_post_obj.create_followup.assert_called_once_with("plain text reply")
        fake_post_obj.create_instructor_answer.assert_not_called()

    def test_single_answer_is_posted_directly(self, bot, network):
        post = self._post(network)
        bot.register_post_handler(lambda info: Answer("the answer"))
        fake_post_obj = MagicMock()
        network.get_post.return_value = fake_post_obj

        bot.respond_to_post(post)

        fake_post_obj.create_instructor_answer.assert_called_once_with(
            "the answer", revision=0
        )

    def test_multiple_answers_are_joined_with_separator(self, bot, network):
        post = self._post(network)
        bot.register_post_handler(lambda info: Answer("first"))
        bot.register_post_handler(lambda info: Answer("second"))
        fake_post_obj = MagicMock()
        network.get_post.return_value = fake_post_obj

        bot.respond_to_post(post)

        fake_post_obj.create_instructor_answer.assert_called_once_with(
            "first<p></p><p>---</p><p></p>second", revision=0
        )

    def test_multiple_followups_are_each_posted_separately(self, bot, network):
        post = self._post(network)
        bot.register_post_handler(lambda info: Followup("f1"))
        bot.register_post_handler(lambda info: Followup("f2"))
        fake_post_obj = MagicMock()
        network.get_post.return_value = fake_post_obj

        bot.respond_to_post(post)

        assert fake_post_obj.create_followup.call_args_list == [call("f1"), call("f2")]

    def test_no_handlers_posts_nothing(self, bot, network):
        post = self._post(network)

        bot.respond_to_post(post)

        network.get_post.assert_not_called()


# ---------------------------------------------------------------------------
# register_post_handler
# ---------------------------------------------------------------------------


class TestRegisterPostHandler:
    def test_registers_and_returns_handler_unchanged(self, bot):
        def handler(post_info):
            return None

        returned = bot.register_post_handler(handler)

        assert returned is handler
        assert bot.post_handlers == [handler]

    def test_multiple_handlers_registered_in_order(self, bot):
        def h1(info):
            return None

        def h2(info):
            return None

        bot.register_post_handler(h1)
        bot.register_post_handler(h2)

        assert bot.post_handlers == [h1, h2]


# ---------------------------------------------------------------------------
# process_all_posts / process_new_posts
# ---------------------------------------------------------------------------


class TestProcessPosts:
    def test_process_all_posts_builds_dataframe_and_writes_csv(
        self, bot, network, monkeypatch
    ):
        written = {}

        def fake_to_csv(self, path, *args, **kwargs):
            written["path"] = path
            written["frame"] = self.copy()

        monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

        skip_post = make_post(nr=1, bucket_name="Pinned")
        good_post = make_post(nr=2, uid_a="uid-2")
        network.iter_all_posts.side_effect = [[skip_post, good_post], []]
        network.get_users.return_value = [{"name": "Carol"}]

        bot.process_all_posts()

        assert Path(written["path"]) == DATA_DIR / "posts.csv"
        assert list(written["frame"]["post_id"]) == [2]
        assert list(written["frame"]["username"]) == ["Carol"]
        assert list(bot.df["post_id"]) == [2]

    def test_process_new_posts_uses_short_time_limit_and_writes_csv(
        self, bot, network, monkeypatch
    ):
        written = {}

        def fake_to_csv(self, path, *args, **kwargs):
            written["path"] = path
            written["frame"] = self.copy()

        monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

        get_posts_calls = []
        original_get_posts = bot.get_posts

        def spy_get_posts(time_limit=3600):
            get_posts_calls.append(time_limit)
            return original_get_posts(time_limit=time_limit)

        monkeypatch.setattr(bot, "get_posts", spy_get_posts)
        network.iter_all_posts.side_effect = [[]]

        bot.process_new_posts()

        assert get_posts_calls == [300]
        assert Path(written["path"]) == DATA_DIR / "posts.csv"
        assert list(written["frame"].columns) == list(bot.df.columns)

    def test_process_all_posts_skips_posts_with_missing_user_info(
        self, bot, network, monkeypatch
    ):
        monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *a, **k: None)

        post_without_uid = make_post(nr=3)
        del post_without_uid["history"][0]["uid_a"]
        network.iter_all_posts.side_effect = [[post_without_uid], []]

        bot.process_all_posts()

        assert len(bot.df) == 0
