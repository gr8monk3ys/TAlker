import logging, os, re, sys
from bot import Answer, Followup, PiazzaBot
import parameters
import pandas as pd

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
        bot.process_all_posts()

if __name__ == '__main__':
    profile = Profile()
    profile.main()