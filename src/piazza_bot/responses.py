class Answer:
    def __init__(self, text):
        self.text = text

    def __str__(self) -> str:
        return self.text

    def get_formatted_text(self) -> str:
        """Return the text formatted as an answer."""
        return f"Answer: {self.text}"


class Followup:
    def __init__(self, text):
        self.text = text

    def __str__(self) -> str:
        return self.text

    def get_formatted_text(self) -> str:
        """Return the text formatted as a followup."""
        return f"Followup: {self.text}"
