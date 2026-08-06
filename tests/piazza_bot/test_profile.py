"""
Tests for src/piazza_bot/profile.py (Profile).

Profile.__init__ performs real work at construction time: it reads Piazza
credentials from the environment and performs a real login via
`piazza_api.Piazza`. Every test below patches `os.getenv`, `load_dotenv`, and
the `Piazza` class so no real network/env access happens.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.piazza_bot.profile import Profile

ENV = {
    "PIAZZA_EMAIL": "prof@example.edu",
    "PIAZZA_PASSWORD": "hunter2",
    "PIAZZA_COURSE_ID": "course_123",
}


def make_post(
    nr=1,
    post_type="question",
    subject="Some Title",
    content="Some content",
    created="2024-01-01T00:00:00Z",
    tags=None,
    is_answered=False,
    num_favorites=0,
    history=True,
):
    post = {
        "nr": nr,
        "type": post_type,
        "created": created,
        "tags": tags if tags is not None else ["homework"],
        "is_answered": is_answered,
        "num_favorites": num_favorites,
    }
    if history:
        post["history"] = [{"subject": subject, "content": content}]
    return post


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_raises_when_credentials_missing(self, monkeypatch):
        monkeypatch.delenv("PIAZZA_EMAIL", raising=False)
        monkeypatch.delenv("PIAZZA_PASSWORD", raising=False)
        monkeypatch.delenv("PIAZZA_COURSE_ID", raising=False)

        with patch("src.piazza_bot.profile.load_dotenv"):
            with pytest.raises(ValueError, match="Missing Piazza credentials"):
                Profile()

    def test_raises_when_only_some_credentials_present(self, monkeypatch):
        monkeypatch.setenv("PIAZZA_EMAIL", "a@b.com")
        monkeypatch.delenv("PIAZZA_PASSWORD", raising=False)
        monkeypatch.delenv("PIAZZA_COURSE_ID", raising=False)

        with patch("src.piazza_bot.profile.load_dotenv"):
            with pytest.raises(ValueError, match="Missing Piazza credentials"):
                Profile()

    def test_successful_login_stores_network_and_credentials(self, monkeypatch):
        for key, value in ENV.items():
            monkeypatch.setenv(key, value)

        fake_piazza_instance = MagicMock()
        fake_network = MagicMock()
        fake_piazza_instance.network.return_value = fake_network
        fake_piazza_cls = MagicMock(return_value=fake_piazza_instance)

        with (
            patch("src.piazza_bot.profile.load_dotenv"),
            patch("src.piazza_bot.profile.Piazza", fake_piazza_cls),
        ):
            profile = Profile()

        assert profile.email == ENV["PIAZZA_EMAIL"]
        assert profile.password == ENV["PIAZZA_PASSWORD"]
        assert profile.course_id == ENV["PIAZZA_COURSE_ID"]
        fake_piazza_instance.user_login.assert_called_once_with(
            email=ENV["PIAZZA_EMAIL"], password=ENV["PIAZZA_PASSWORD"]
        )
        fake_piazza_instance.network.assert_called_once_with(ENV["PIAZZA_COURSE_ID"])
        assert profile.network is fake_network

    def test_reraises_and_logs_on_login_failure(self, monkeypatch):
        for key, value in ENV.items():
            monkeypatch.setenv(key, value)

        fake_piazza_instance = MagicMock()
        fake_piazza_instance.user_login.side_effect = ConnectionError("no route")
        fake_piazza_cls = MagicMock(return_value=fake_piazza_instance)

        with (
            patch("src.piazza_bot.profile.load_dotenv"),
            patch("src.piazza_bot.profile.Piazza", fake_piazza_cls),
        ):
            with pytest.raises(ConnectionError, match="no route"):
                Profile()


# ---------------------------------------------------------------------------
# get_posts
# ---------------------------------------------------------------------------


@pytest.fixture
def profile(monkeypatch):
    """A Profile with a mocked, already-authenticated network."""
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)

    fake_piazza_instance = MagicMock()
    fake_network = MagicMock()
    fake_piazza_instance.network.return_value = fake_network
    fake_piazza_cls = MagicMock(return_value=fake_piazza_instance)

    with (
        patch("src.piazza_bot.profile.load_dotenv"),
        patch("src.piazza_bot.profile.Piazza", fake_piazza_cls),
    ):
        p = Profile()
    p.network = fake_network
    return p


class TestGetPosts:
    def test_builds_dataframe_from_posts(self, profile):
        profile.network.iter_all_posts.return_value = [
            make_post(
                nr=1,
                subject="Q1",
                content="body1",
                tags=["hw1", "urgent"],
                is_answered=True,
                num_favorites=3,
            ),
            make_post(nr=2, subject="Q2", content="body2"),
        ]

        df = profile.get_posts(time_limit=100)

        profile.network.iter_all_posts.assert_called_once_with(limit=50)
        assert isinstance(df, pd.DataFrame)
        assert list(df["id"]) == [1, 2]
        assert list(df["title"]) == ["Q1", "Q2"]
        assert list(df["content"]) == ["body1", "body2"]
        assert list(df["tags"]) == ["hw1, urgent", "homework"]
        assert list(df["is_answered"]) == [True, False]
        assert list(df["num_favorites"]) == [3, 0]

    def test_empty_post_list_returns_empty_dataframe(self, profile):
        profile.network.iter_all_posts.return_value = []

        df = profile.get_posts()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_missing_history_falls_back_to_defaults(self, profile):
        profile.network.iter_all_posts.return_value = [make_post(nr=9, history=False)]

        df = profile.get_posts()

        assert df.iloc[0]["title"] == "No Title"
        assert df.iloc[0]["content"] == "No Content"

    def test_missing_optional_fields_use_defaults(self, profile):
        bare_post = {"nr": 5, "type": "note", "history": [{}]}

        profile.network.iter_all_posts.return_value = [bare_post]

        df = profile.get_posts()

        row = df.iloc[0]
        assert row["created"] == "Unknown"
        assert row["tags"] == ""
        assert bool(row["is_answered"]) is False
        assert row["num_favorites"] == 0

    def test_propagates_and_logs_network_error(self, profile):
        profile.network.iter_all_posts.side_effect = RuntimeError("piazza down")

        with pytest.raises(RuntimeError, match="piazza down"):
            profile.get_posts()


# ---------------------------------------------------------------------------
# process_post
# ---------------------------------------------------------------------------


class TestProcessPost:
    def test_posts_response_via_network(self, profile):
        profile.process_post({"id": 42})

        profile.network.create_followup.assert_called_once_with(
            42, "This is a test response."
        )

    def test_missing_id_raises_key_error(self, profile):
        with pytest.raises(KeyError):
            profile.process_post({"not_id": 1})

        profile.network.create_followup.assert_not_called()

    def test_propagates_network_error(self, profile):
        profile.network.create_followup.side_effect = ConnectionError("dropped")

        with pytest.raises(ConnectionError, match="dropped"):
            profile.process_post({"id": 7})
