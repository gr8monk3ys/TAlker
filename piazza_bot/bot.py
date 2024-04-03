import logging
import time, os, sys, collections
import piazza_api
from responses import *
import pandas as pd

class PiazzaBot:
    POST_LOOKBACK_LIMIT = 10

    def __init__(self, piazza_instance, user_profile, course_network):
        self.piazza = piazza_instance
        self.user_profile = user_profile
        self.network = course_network
        self.post_handlers = []
        self.df = pd.DataFrame(columns=['username', 'content', 'post_id', 'status', 'timestamp'])

    @classmethod
    def create_bot(cls, email, password, course_id):
        piazza = piazza_api.Piazza()
        piazza.user_login(email=email, password=password)
        user_profile = piazza.get_user_profile()
        network = piazza.network(course_id)
        return cls(piazza, user_profile, network)

    def fetch_all_posts(self):
        all_posts = []
        start_time = time.time()  # Capture the start time
        time_limit = 120  # Set a time limit of 60 seconds
        backoff_time = 4  # Start with a 2-second wait

        while True:
            current_time = time.time()
            if current_time - start_time > time_limit:
                print("Reached time limit for fetching posts.")
                break  # Exit the loop if the time limit has been reached
            
            try:
                batch_of_posts = self.network.iter_all_posts(limit=self.POST_LOOKBACK_LIMIT)
                if not batch_of_posts:
                    print("No more posts to fetch.")
                    break  # No more posts to fetch
                all_posts.extend(batch_of_posts)
                backoff_time = 4  # Reset backoff time after a successful request
            except piazza_api.exceptions.RequestError as e:
                logging.error(f"RequestError: {e}. Retrying in {backoff_time} seconds.")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponentially increase the wait time
                if backoff_time > time_limit:
                    print("Backoff time exceeded time limit. Ending fetch operation.")
                    break  # Break the loop if the backoff time itself exceeds the time limit
                continue  # Skip the rest of the loop and retry
            time.sleep(2)  # Always wait at least 1 second between successful requests
        return all_posts

    def fetch_new_posts(self):
        """Retrieve recent posts from the course network."""
        return self.network.iter_all_posts(limit=self.POST_LOOKBACK_LIMIT)

    def process_all_posts(self):
        """Evaluate and respond to all posts on the course network."""
        for post in self.fetch_all_posts():
            # Assuming you've defined logic to decide if a post should be skipped
            if self.should_skip_post(post):
                continue
            post_info = self.extract_post_info(post)
            if post_info is not None:  # Ensure post_info is not None before appending
                self.df = self.df._append(post_info, ignore_index=True)
        self.df.to_csv('../data/posts.csv')

    def process_new_posts(self):
        """Evaluate and respond to new posts on the course network."""
        for post in self.fetch_new_posts():
            if self.should_skip_post(post):
                continue
            # self.respond_to_post(post)
            post_info = self.extract_post_info(post)
            if post_info is not None:  # Make sure post_info is not None before appending
                self.df = self.df._append(post_info, ignore_index=True)
        self.df.to_csv('../data/posts.csv')

    def respond_to_post(self, post):
        post_info = self.extract_post_info(post)
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
            joined_answers = "<p></p><p>---</p><p></p>".join(answer.text for answer in answers)
            self.network.create_instructor_answer(post, joined_answers, revision=0)

        for followup in followups:
            self.network.create_followup(post, followup.text)

    def extract_post_info(self, post):
        """Extracts relevant information from a post for processing."""
        history = post["history"][0]
        content = history["content"]
        status = post["status"]
        user_id = history.get("uid_a")  # Changed from "uid" to "uid_a"
        created_timestamp = post.get("created")
        
        if user_id is None:
            logging.error(f"User ID ('uid_a') not found in post history for post_id: {post.get('nr')}")
            return None

        # Assuming get_users() method can handle None gracefully or you ensure user_id is not None before calling
        user_info = self.network.get_users([user_id]) if user_id else []
        if not user_info:
            logging.error(f"No user information found for user_id: {user_id}")
            username = "Anonymous"
        else:
            username = user_info[0]["name"]

        return {
        'username': username,
        'content': content,
        'post_id': post["nr"],
        'status': status,
        'timestamp': created_timestamp
        }

    def should_skip_post(self, post):
        """Determines if a post should be ignored based on its content or status."""
        if post["bucket_name"] == "Pinned" or post["config"].get("is_announcement", 0) == 1:
            return True

        for child in post["children"]:
            if child.get("uid") == self.user_profile["user_id"]:
                logging.info(f"Skipping post @{post['nr']} - already commented.")
                return True

        return self.already_answered(post)

    @staticmethod
    def already_answered(post):
        """Checks if the post already has an instructor's answer."""
        return any(change for change in post["change_log"] if change["type"] == "i_answer")

    def register_post_handler(self, handler_func):
        self.post_handlers.append(handler_func)
        return handler_func

    def run(self):
        """Continuously monitor and respond to new posts on the Piazza network."""
        while True:
            self.process_all_posts()
            time.sleep(5)