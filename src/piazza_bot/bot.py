import logging
import time
import pandas as pd
import piazza_api
from .responses import Answer, Followup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PiazzaBot:
    POST_LOOKBACK_LIMIT = 10

    @classmethod
    def create_bot(cls, email, password, course_id):
        piazza = piazza_api.Piazza()
        piazza.user_login(email=email, password=password)
        course = piazza.network(course_id)
        return cls(course)

    def __init__(self, network):
        self.network = network
        self.user_profile = network.get_user_profile()
        self.post_handlers = []
        self.df = pd.DataFrame(
            columns=["username", "content", "post_id", "status", "timestamp"]
        )

    def get_posts(self, time_limit=3600):
        """
        Fetch posts from Piazza within the given time limit.
        Args:
            time_limit: Time limit in seconds (default: 1 hour)
        """
        all_posts = []
        start_time = time.time()
        backoff_time = 4

        while True:
            current_time = time.time()
            if current_time - start_time > time_limit:
                logger.info("Reached time limit for fetching posts.")
                break

            try:
                batch_of_posts = self.network.iter_all_posts(
                    limit=self.POST_LOOKBACK_LIMIT
                )
                if not batch_of_posts:
                    logger.info("No more posts to fetch.")
                    break

                all_posts.extend(batch_of_posts)
                backoff_time = 4  # Reset backoff time after success

            except (piazza_api.exceptions.RequestError,
                   piazza_api.exceptions.AuthenticationError,
                   ConnectionError) as e:
                logger.error(
                    "RequestError: %s. Retrying in %s seconds.",
                    e,
                    backoff_time
                )
                time.sleep(backoff_time)
                backoff_time *= 2
                if backoff_time > time_limit:
                    logger.info(
                        "Backoff time exceeded time limit. Ending fetch."
                    )
                    break
                continue

            time.sleep(2)

        return all_posts

    def post_answer(self, post_id, content):
        """Post an answer to the given post."""
        post = self.network.get_post(post_id)
        post.create_instructor_answer(content, revision=0)

    def post_followup(self, post_id, content):
        """Post a followup to the given post."""
        post = self.network.get_post(post_id)
        post.create_followup(content)

    def post_response(self, post_id, response_obj):
        """Post a response object to the given post."""
        if isinstance(response_obj, Answer):
            self.post_answer(post_id, response_obj.text)
        elif isinstance(response_obj, Followup):
            self.post_followup(post_id, response_obj.text)

    def get_user_info(self, post):
        """Extract user information from a post."""
        history = post["history"][0]
        content = history["content"]
        status = post["status"]
        user_id = history.get("uid_a")  # Changed from "uid" to "uid_a"
        created_timestamp = post.get("created")

        if user_id is None:
            logger.error(
                "User ID ('uid_a') not found in post history for post_id: %s",
                post.get('nr')
            )
            return None

        user_info = self.network.get_users([user_id]) if user_id else []
        if not user_info:
            logger.error("No user information found for user_id: %s", user_id)
            username = "Anonymous"
        else:
            username = user_info[0]["name"]

        return {
            "username": username,
            "content": content,
            "post_id": post["nr"],
            "status": status,
            "timestamp": created_timestamp,
        }

    def already_answered(self, post):
        """Check if we've already answered this post."""
        if not post.get("children"):
            return False

        for child in post["children"]:
            if child.get("uid") == self.user_profile["user_id"]:
                logger.info("Skipping post @%s - already commented.", post['nr'])
                return True

        return False

    def should_skip_post(self, post):
        """Determines if a post should be ignored based on its content or status."""
        if (
            post["bucket_name"] == "Pinned"
            or post["config"].get("is_announcement", 0) == 1
        ):
            return True

        return self.already_answered(post)

    def process_all_posts(self):
        """Evaluate and respond to all posts on the course network."""
        posts_data = []
        for post in self.get_posts():
            if self.should_skip_post(post):
                continue
            post_info = self.get_user_info(post)
            if post_info is not None:
                posts_data.append(post_info)
        self.df = pd.DataFrame(posts_data)
        self.df.to_csv("../data/posts.csv")

    def process_new_posts(self):
        """Evaluate and respond to new posts on the course network."""
        posts_data = []
        for post in self.get_posts(time_limit=300):
            if self.should_skip_post(post):
                continue
            post_info = self.get_user_info(post)
            if post_info is not None:
                posts_data.append(post_info)
        self.df = pd.DataFrame(posts_data)
        self.df.to_csv("../data/posts.csv")

    def respond_to_post(self, post):
        post_info = self.get_user_info(post)
        responses = (handler(post_info) for handler in self.post_handlers)
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
            joined_answers = "<p></p><p>---</p><p></p>".join(
                answer.text for answer in answers
            )
            self.post_answer(post["nr"], joined_answers)

        for followup in followups:
            self.post_followup(post["nr"], followup.text)

    def register_post_handler(self, handler_func):
        self.post_handlers.append(handler_func)
        return handler_func

    def run(self):
        """Continuously monitor and respond to new posts on the Piazza network."""
        while True:
            self.process_all_posts()
            time.sleep(5)
