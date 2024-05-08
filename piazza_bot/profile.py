import logging, os, re, sys
from bot import Answer, Followup, PiazzaBot
import parameters
import pandas as pd
import sys
sys.path.insert(0, '../dashboard')  # Adjust the path to where llm.py is relative to profile.py
from llm import Llm_chain


class Profile:
    def die(self, message):
        sys.stderr.write("{}\n".format(message))
        sys.exit(1)

    @classmethod
    def create_bot(cls):
        """Prepare and return a PiazzaBot instance based on configuration parameters."""
        config_sources = {
            "email": parameters.email,
            "password": parameters.password,
            "course_id": parameters.class_code,
        }

        config = {}
        for var, source in config_sources.items():
            if not source:
                logging.error(f"Missing configuration for {var}. Please check your parameters.")
                sys.exit(1)
            config[var] = source

        return PiazzaBot.create_bot(**config)

    def main(self):
        logging.basicConfig(level=logging.DEBUG)
        bot = self.create_bot()
        processed_posts = set()

        while True:
            all_posts = bot.fetch_all_posts()
            new_posts = [post for post in all_posts if post['id'] not in processed_posts]

            for post in new_posts:
                if not bot.already_answered(post):
                    response = llm_chain.generate_response(post['history'], post['classroom'])
                    bot.post_response(post['id'], response)
                    processed_posts.add(post['id'])

if __name__ == '__main__':
    profile = Profile()
    profile.main()
