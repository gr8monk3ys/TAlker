import os
import logging
from typing import Dict, Any, List
import pandas as pd
from piazza_api import Piazza
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Profile:
    """Profile class for managing Piazza interactions."""

    def __init__(self):
        """Initialize Profile with Piazza credentials."""
        load_dotenv()  # Load environment variables
        
        # Get credentials from environment variables
        self.email = os.getenv("PIAZZA_EMAIL")
        self.password = os.getenv("PIAZZA_PASSWORD")
        self.course_id = os.getenv("PIAZZA_COURSE_ID")

        if not all([self.email, self.password, self.course_id]):
            raise ValueError(
                "Missing Piazza credentials. Please check your .env file contains "
                "PIAZZA_EMAIL, PIAZZA_PASSWORD, and PIAZZA_COURSE_ID"
            )

        try:
            # Initialize Piazza
            self.p = Piazza()
            self.p.user_login(email=self.email, password=self.password)
            logger.info(f"Successfully logged in as {self.email}")
            
            # Get course network
            self.network = self.p.network(self.course_id)
            logger.info(f"Connected to course {self.course_id}")
        except Exception as e:
            logger.error(f"Failed to authenticate with Piazza: {str(e)}")
            raise

    def get_posts(self, time_limit: int = 3600) -> pd.DataFrame:
        """
        Get posts from Piazza within the time limit.
        
        Args:
            time_limit (int): Time limit in seconds (default: 1 hour)
            
        Returns:
            pd.DataFrame: DataFrame containing posts
        """
        try:
            # Get all posts
            posts = self.network.iter_all_posts(limit=50)  # Limit to 50 posts for testing
            
            # Convert posts to list of dictionaries
            posts_list = []
            for post in posts:
                post_dict = {
                    "id": post["nr"],
                    "type": post["type"],
                    "title": post.get("history", [{}])[0].get("subject", "No Title"),
                    "content": post.get("history", [{}])[0].get("content", "No Content"),
                    "created": post.get("created", "Unknown"),
                    "tags": ", ".join(post.get("tags", [])),
                    "is_answered": post.get("is_answered", False),
                    "num_favorites": post.get("num_favorites", 0),
                }
                posts_list.append(post_dict)
            
            # Create DataFrame
            df = pd.DataFrame(posts_list)
            logger.info(f"Retrieved {len(df)} posts from Piazza")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching posts: {str(e)}")
            raise

    def process_post(self, post_data: Dict[str, Any]) -> None:
        """
        Process a post and generate a response.
        
        Args:
            post_data (Dict[str, Any]): Post data containing id and other information
        """
        try:
            # Get the post ID
            post_id = post_data["id"]
            
            # Generate response using LLM (to be implemented)
            response = "This is a test response."
            
            # Post the response
            self.network.create_followup(post_id, response)
            logger.info(f"Posted response to post {post_id}")
            
        except Exception as e:
            logger.error(f"Error processing post {post_data.get('id')}: {str(e)}")
            raise

    def main(self) -> None:
        """Main execution loop."""
        processed_posts = set()
        while True:
            new_posts = self.get_posts(time_limit=300)
            for post in new_posts.itertuples():
                if post.id not in processed_posts:
                    self.process_post(post._asdict())
                    processed_posts.add(post.id)


if __name__ == "__main__":
    profile = Profile()
    profile.main()
