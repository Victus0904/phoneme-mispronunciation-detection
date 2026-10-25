"""End-to-end checks for the app: scoring the bundled example clips at every
strictness level, input validation, and the static files. Runs the real model
(first run downloads it), no network calls to the LLM services.

Usage: python -m unittest discover tests      (~1-2 min on CPU)
"""
import io
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient

import server
from src.pairs import PAIR_SETS
from src.teacher import build_facts, template_feedback

LEVELS = ["relaxed", "standard", "strict", "very_strict"]
EXAMPLES = {e["wav"]: e for e in json.load(open(os.path.join(ROOT, "examples", "manifest.json")))}


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)
        cls.cache = {}

    def setUp(self):
        server._hits.clear()  # rate limits are per minute; don't let tests trip them

    def score(self, wav, level="standard", sentence=None):
        key = (wav, level, sentence)
        if key not in self.cache:
            with open(os.path.join(ROOT, "examples", wav), "rb") as f:
                r = self.client.post("/api/score", files={"audio": (wav, f, "audio/wav")},
                                     data={"sentence": sentence or EXAMPLES[wav]["sentence"], "strictness": level})
            self.assertEqual(r.status_code, 200)
            self.cache[key] = r.json()
        return self.cache[key]

    # ------------------------------------------------------------ scoring
    def test_info_lists_strictness_levels(self):
        info = self.client.get("/api/info").json()
        self.assertEqual(set(info["strictness"]), set(LEVELS))
        recalls = [info["strictness"][k]["recall"] for k in LEVELS]
        self.assertEqual(recalls, sorted(recalls), "stricter levels must catch more errors")

    def test_result_shape(self):
        j = self.score("substitution_there.wav")
        self.assertTrue(j["ok"])
        self.assertIn("id", j)
        self.assertEqual(j["overall"]["strictness"], "standard")
        self.assertEqual(j["overall"]["n_phones"], sum(len(w["phones"]) for w in j["words"]))
        self.assertTrue(0 <= j["fluency"]["fluency_10"] <= 10)
        for w in j["words"]:
            self.assertLessEqual(w["start_s"], w["end_s"])

    def test_obvious_substitution_is_flagged(self):
        there = self.score("substitution_there.wav")["words"][0]
        self.assertEqual(there["phones"][0]["arpa"], "DH")
        self.assertTrue(there["phones"][0]["flagged"])

    def test_clean_clip_has_no_flags_at_standard(self):
        self.assertEqual(self.score("clean_more.wav")["overall"]["n_flagged"], 0)

    def test_stricter_levels_flag_more(self):
        for wav in EXAMPLES:
            flags = [self.score(wav, k)["overall"]["n_flagged"] for k in LEVELS]
            scores = [self.score(wav, k)["overall"]["score_100"] for k in LEVELS]
            self.assertEqual(flags, sorted(flags), wav)
            self.assertEqual(scores, sorted(scores, reverse=True), wav)

    def test_unknown_level_falls_back_to_standard(self):
        self.assertEqual(self.score("clean_more.wav", "nonsense")["overall"]["strictness"], "standard")

    # ------------------------------------------------------------ input validation
    def test_rejects_long_sentence(self):
        j = self.score("clean_more.wav", sentence="word " * 60)
        self.assertFalse(j["ok"])

    def test_rejects_unreadable_audio(self):
        r = self.client.post("/api/score", files={"audio": ("x.wav", io.BytesIO(b"not audio"), "audio/wav")},
                             data={"sentence": "hello"})
        self.assertFalse(r.json()["ok"])

    def test_rejects_oversized_upload(self):
        big = io.BytesIO(b"\0" * (server.MAX_UPLOAD_BYTES + 10))
        r = self.client.post("/api/score", files={"audio": ("x.wav", big, "audio/wav")}, data={"sentence": "hello"})
        self.assertFalse(r.json()["ok"])

    def test_feedback_only_for_known_results(self):
        self.assertFalse(self.client.post("/api/feedback", json={"id": "made-up"}).json()["ok"])

    def test_rate_limit(self):
        n = server.LIMITS["phonemes"]
        codes = [self.client.get("/api/phonemes", params={"sentence": "hello"}).json()["ok"] for _ in range(n + 1)]
        self.assertTrue(all(codes[:n]))
        self.assertFalse(codes[-1])

    # ------------------------------------------------------------ static files
    def test_static_files(self):
        for path in ["/", "/manifest.webmanifest", "/sw.js", "/icons/icon-192.png", "/examples/clean_more.wav"]:
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_no_path_traversal(self):
        for path in ["/examples/..%2Fserver.py", "/icons/..%2F..%2Fserver.py", "/examples/manifest.json.bak"]:
            self.assertEqual(self.client.get(path).status_code, 404, path)

    # ------------------------------------------------------------ pairs + coach
    def test_pairs_are_calibrated(self):
        with open(os.path.join(ROOT, "models", "pairs_calibration.json")) as f:
            bias = json.load(f)["bias_ab"]
        for s in PAIR_SETS:
            self.assertIn(s["id"], bias)
            for a, b in s["pairs"]:
                self.assertNotEqual(a, b)

    def test_template_feedback_without_llm(self):
        text = template_feedback(build_facts(self.score("substitution_there.wav")), "Maya")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 40)


if __name__ == "__main__":
    unittest.main()
