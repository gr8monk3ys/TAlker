import requests

print(
    requests.post(
        "http://0.0.0.0:10000",
        json={"message": "Welcome to the TA Bot, how can I be of assistance today?"},
        timeout=30
    ).json()
)
