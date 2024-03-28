import logging
import time
import piazza_api
from .utils import PostInfo, ignore_error
from .responses import Answer, Followup

PostInfo = collections.namedtuple("PostInfo", ["username", "text", "post_id", "status"])

class ignore_error:
    def __init__(self, *error_types):
        self.error_types = error_types

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except self.error_types as e:
                logging.error(f"Ignoring error {e} in function {func.__name__}")
        return wrapped


def test_ignore_error():
    @ignore_error(KeyError, ValueError)
    def ignored():
        raise ValueError()

    try:
        ignored()
    except ValueError:
        assert False

    @ignore_error(ValueError, IOError)
    def not_ignored():
        raise KeyError()

    try:
        not_ignored()
        assert False
    except KeyError:
        pass


class PiazzaBot:
    POST_LOOKBACK_LIMIT = 50
    """
    The number of most recent posts to examine for responses. Adjust this
    value based on the expected volume of discussions and queries.
    """

    def __init__(self, piazza_instance, user_profile, course_network):
        self._piazza = piazza_instance
        self._user_profile = user_profile
        self._network = course_network
        self._post_handlers = []

    @classmethod
    def create_bot(cls, email, password, course_id):
        piazza = piazza_api.Piazza()
        piazza.user_login(email=email, password=password)
        user_profile = piazza.get_user_profile()
        network = piazza.network(course_id)
        return cls(piazza, user_profile, network)

    def _fetch_new_posts(self):
        """Retrieve recent posts from the course network."""
        return self._network.iter_all_posts(limit=self.POST_LOOKBACK_LIMIT)

    @ignore_error(IOError)
    def _process_new_posts(self):
        """Evaluate and respond to new posts on the course network."""
        for post in self._fetch_new_posts():
            if self._should_skip_post(post):
                continue
            self._respond_to_post(post)

    @ignore_error(KeyError, IndexError)
    def _respond_to_post(self, post):
        post_info = self._extract_post_info(post)
        responses = (handler(post_info) for handler in self._post_handlers)
        responses = [response for response in responses if response is not None]

        answers, followups = [], []
        for response in responses:
            if isinstance(response, Answer):
                answers.append(response)
            elif isinstance(response, Followup):
                followups.append(response)
            else:
                followups.append(Followup(response))

        if answers:
            joined_answers = "<p></p><p>---</p><p></p>".join(answer.text for answer in answers)
            self._network.create_instructor_answer(post, joined_answers, revision=0)

        for followup in followups:
            self._network.create_followup(post, followup.text)

    def _extract_post_info(self, post):
        """Extracts relevant information from a post for processing."""
        history = post["history"][0]
        content = history["content"]
        status = post["status"]
        user_id = history["uid"]
        username = self._network.get_users([user_id])[0]["name"]

        return PostInfo(username=username, text=content, post_id=post["nr"], status=status)

    @ignore_error(KeyError)
    def _should_skip_post(self, post):
        """Determines if a post should be ignored based on its content or status."""
        if post["bucket_name"] == "Pinned" or post["config"].get("is_announcement", 0) == 1:
            return True

        for child in post["children"]:
            if child.get("uid") == self._user_profile["user_id"]:
                logging.info(f"Skipping post @{post['nr']} - already commented.")
                return True

        return self._already_answered(post)

    @staticmethod
    def _already_answered(post):
        """Checks if the post already has an instructor's answer."""
        return any(change for change in post["change_log"] if change["type"] == "i_answer")

    def register_post_handler(self, handler_func):
        self._post_handlers.append(handler_func)
        return handler_func

    def run(self):
        """Continuously monitor and respond to new posts on the Piazza network."""
        while True:
            self._process_new_posts()
            time.sleep(5)