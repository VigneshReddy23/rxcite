"""Load test for the rxcite API (Locust).

  locust -f bench/locustfile.py --headless -u 50 -r 10 -t 60s \
      --host http://localhost:8000 --csv results/load_search   # RXCITE_ENDPOINT=search
Questions are drawn from the Stage 2 test set, so requests look like real traffic.
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    json.loads(line)["question"]
    for line in Path("eval/questions.jsonl").read_text().splitlines()
    if line
]
ENDPOINT = os.environ.get("RXCITE_ENDPOINT", "search")  # "search" (no LLM) or "ask"


class RxciteUser(HttpUser):
    wait_time = between(0.5, 1.5)  # think time between a user's requests

    @task
    def query(self) -> None:
        self.client.post(f"/{ENDPOINT}", json={"question": random.choice(QUESTIONS)}, name=ENDPOINT)
