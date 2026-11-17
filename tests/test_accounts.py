"""Accounts, sessions and first-language customisation (API level).
Uses a temporary accounts file; the scoring check runs the real model.

Usage: python -m unittest discover tests
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient

import server
from src import accounts, l1_profiles


class AccountsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["ACCOUNTS_STORE"] = os.path.join(cls.tmp.name, "accounts.json")
        os.environ["SESSION_SECRET"] = "test-secret"
        accounts.reset_store()
        accounts._secret = None

    @classmethod
    def tearDownClass(cls):
        accounts.reset_store()
        os.environ.pop("ACCOUNTS_STORE", None)
        os.environ.pop("SESSION_SECRET", None)
        accounts._secret = None
        cls.tmp.cleanup()

    def setUp(self):
        server._hits.clear()
        self.c = TestClient(server.app)

    def test_signup_login_profile_logout_delete(self):
        r = self.c.post("/api/signup", json={"email": "Ana@Example.com", "password": "correct horse", "name": "Ana"})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["user"]["email"], "ana@example.com")
        self.assertIn(accounts.COOKIE, r.cookies)
        cookie = r.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        me = self.c.get("/api/me").json()
        self.assertEqual(me["user"]["name"], "Ana")
        self.assertIsNone(me["profile"])  # language not chosen yet

        r = self.c.post("/api/profile", json={"l1": "es"})
        self.assertTrue(r.json()["ok"])
        self.assertEqual(r.json()["profile"]["label"], "Spanish")
        self.assertEqual(self.c.get("/api/me").json()["user"]["l1"], "es")
        self.assertEqual(self.c.post("/api/profile", json={"l1": "klingon"}).status_code, 400)

        self.c.post("/api/logout")
        self.assertIsNone(self.c.get("/api/me").json()["user"])
        self.assertEqual(self.c.post("/api/profile", json={"l1": "fr"}).status_code, 401)

        self.assertEqual(self.c.post("/api/login", json={"email": "ana@example.com", "password": "wrong pass"}).status_code, 401)
        r = self.c.post("/api/login", json={"email": "ANA@example.com", "password": "correct horse"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["user"]["l1"], "es")  # kept across sessions

        self.assertTrue(self.c.post("/api/account/delete").json()["ok"])
        self.assertIsNone(accounts.get_user("ana@example.com"))
        self.assertEqual(self.c.post("/api/login", json={"email": "ana@example.com", "password": "correct horse"}).status_code, 401)

    def test_progress_sync_follows_the_account(self):
        self.assertEqual(self.c.get("/api/progress").status_code, 401)
        self.c.post("/api/signup", json={"email": "sync@example.com", "password": "correct horse"})
        p = self.c.get("/api/progress").json()
        self.assertEqual((p["attempts"], p["pairs"]), ([], {}))
        attempts = [{"t": 1700000000000 + i, "s": f"sentence {i}", "score": 80} for i in range(600)]
        r = self.c.post("/api/progress", json={"attempts": attempts + ["junk", {"no_t": 1}],
                                               "pairs": {"ship_sheep": {"ok": 3, "n": 5}}})
        self.assertTrue(r.json()["ok"], r.text)
        self.assertEqual(r.json()["n"], accounts.PROGRESS_MAX_ATTEMPTS)  # newest 500, junk dropped
        # a second device (fresh cookie jar) sees the same history after logging in
        other = TestClient(server.app)
        other.post("/api/login", json={"email": "sync@example.com", "password": "correct horse"})
        p = other.get("/api/progress").json()
        self.assertEqual(p["attempts"][-1]["s"], "sentence 599")
        self.assertEqual(p["pairs"]["ship_sheep"], {"ok": 3, "n": 5})
        self.assertEqual(self.c.post("/api/progress", json={"attempts": "nope"}).status_code, 400)
        self.c.post("/api/account/delete")
        self.assertEqual(accounts.get_progress("sync@example.com")["attempts"], [])  # deleted with the account

    def test_signup_validation(self):
        self.assertEqual(self.c.post("/api/signup", json={"email": "not-an-email", "password": "longenough"}).status_code, 400)
        self.assertEqual(self.c.post("/api/signup", json={"email": "a@b.co", "password": "short"}).status_code, 400)
        self.assertEqual(self.c.post("/api/signup", json={"email": "dup@b.co", "password": "longenough"}).status_code, 201)
        self.assertEqual(TestClient(server.app).post("/api/signup", json={"email": "dup@b.co", "password": "longenough"}).status_code, 400)

    def test_signup_keeps_guest_language(self):
        r = self.c.post("/api/signup", json={"email": "guest@b.co", "password": "longenough", "l1": "hi"})
        self.assertEqual(r.json()["user"]["l1"], "hi")

    def test_passwords_are_not_stored_in_plain_text(self):
        self.c.post("/api/signup", json={"email": "plain@b.co", "password": "my secret pw"})
        with open(os.environ["ACCOUNTS_STORE"], encoding="utf-8") as f:
            raw = f.read()
        self.assertNotIn("my secret pw", raw)

    def test_forged_or_expired_tokens_are_rejected(self):
        self.assertIsNone(accounts.read_token("garbage"))
        good = accounts.make_token("x@y.co")
        self.assertEqual(accounts.read_token(good), "x@y.co")
        self.assertIsNone(accounts.read_token(accounts.make_token("x@y.co", days=-1)))
        os.environ["SESSION_SECRET"] = "other-secret"; accounts._secret = None
        try:
            self.assertIsNone(accounts.read_token(good))
        finally:
            os.environ["SESSION_SECRET"] = "test-secret"; accounts._secret = None

    def test_language_profiles(self):
        langs = {l["code"]: l for l in self.c.get("/api/languages").json()["languages"]}
        for code in ("ar", "zh", "hi", "ko", "es", "vi"):
            self.assertTrue(langs[code]["measured"], code)
        self.assertFalse(langs["de"]["measured"])
        es = self.c.get("/api/l1/es").json()["profile"]
        self.assertEqual(es["focus"][0]["sound"], "Z")  # Spanish speakers' most-missed sound in L2-ARCTIC
        self.assertIn("marked wrong", es["focus"][0]["evidence"])
        known_lessons = {"th", "rl", "vw", "ee", "vowels", "endings", "jch", "pbf", "er", "oh"}
        from src.pairs import PAIR_SETS
        known_pairs = {s["id"] for s in PAIR_SETS}
        for l in l1_profiles.LANGUAGES:
            p = l1_profiles.profile(l["code"])
            self.assertTrue(p["focus"], l["code"])
            self.assertLessEqual(set(p["lessons"]), known_lessons, l["code"])
            self.assertLessEqual(set(p["pairs"]), known_pairs, l["code"])
        self.assertEqual(self.c.get("/api/l1/xx").status_code, 404)

    def test_score_adds_first_language_notes(self):
        wav = os.path.join(ROOT, "examples", "substitution_there.wav")
        with open(wav, "rb") as f:
            j = self.c.post("/api/score", files={"audio": ("a.wav", f, "audio/wav")},
                            data={"sentence": "THERE WAS NOTHING LEFT FOR ME", "l1": "es"}).json()
        self.assertTrue(j["ok"], j)
        self.assertEqual(j["l1"], "es")
        dh = j["words"][0]["phones"][0]
        self.assertEqual(dh["arpa"], "DH")
        self.assertIn("Spanish", dh.get("l1_note", ""))


class DirStoreTest(unittest.TestCase):
    """The hosted store: one file per user, committed to the volume after each write."""

    class FakeVolume:
        def __init__(self):
            self.commits = self.reloads = 0

        def commit(self):
            self.commits += 1

        def reload(self):
            self.reloads += 1

    def test_roundtrip_commit_and_no_emails_in_paths(self):
        with tempfile.TemporaryDirectory() as d:
            vol = self.FakeVolume()
            st = accounts.DirStore(d, volume=vol, reload_every=0)
            self.assertIsNone(st.get("a@example.com"))
            st.put("a@example.com", {"email": "a@example.com", "l1": "es"})
            self.assertEqual(vol.commits, 1)
            self.assertEqual(st.get("a@example.com")["l1"], "es")
            self.assertGreaterEqual(vol.reloads, 2)
            self.assertFalse(any("example" in f for f in os.listdir(d)))
            st.delete("a@example.com")
            self.assertIsNone(st.get("a@example.com"))
            self.assertEqual(vol.commits, 2)
            st.delete("a@example.com")  # deleting twice is harmless
            self.assertEqual(vol.commits, 2)


if __name__ == "__main__":
    unittest.main()
