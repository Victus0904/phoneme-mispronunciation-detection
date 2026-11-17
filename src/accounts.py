"""User accounts: sign-up, login, signed session cookies, profile (first language).

Storage is pluggable via the ACCOUNTS_STORE environment variable:
  (unset)                      a JSON file at data/accounts.json (local app)
  /some/path.json              a JSON file at that path
  modal-dict:<name>            a modal.Dict (hosted on Modal; shared by all containers)

Passwords are hashed with scrypt (stdlib hashlib) and a random per-user salt;
plain passwords are never stored or logged. Sessions are stateless signed
tokens (HMAC-SHA256 over user id + expiry) kept in an HttpOnly cookie. The
signing key comes from SESSION_SECRET; without it, a random key is created
once in data/session_secret (fine for one local machine; on Modal it is a
Modal Secret so every container agrees).
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SESSION_DAYS = 30
COOKIE = "pc_session"
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,24}$")


class AccountError(ValueError):
    """A message safe to show to the user."""


# ---------------------------------------------------------------- storage

class FileStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, ValueError):
            return {}

    def get(self, key):
        with self.lock:
            return self._load().get(key)

    def put(self, key, value):
        with self.lock:
            d = self._load()
            d[key] = value
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f)
            os.replace(tmp, self.path)

    def delete(self, key):
        with self.lock:
            d = self._load()
            if d.pop(key, None) is not None:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(d, f)


class ModalDictStore:
    def __init__(self, name):
        import modal
        self.d = modal.Dict.from_name(name, create_if_missing=True)

    def get(self, key):
        return self.d.get(key)

    def put(self, key, value):
        self.d[key] = value

    def delete(self, key):
        self.d.pop(key, None)


_store = None


def store():
    global _store
    if _store is None:
        spec = os.environ.get("ACCOUNTS_STORE", "")
        if spec.startswith("modal-dict:"):
            _store = ModalDictStore(spec.split(":", 1)[1])
        else:
            _store = FileStore(spec or os.path.join(DATA_DIR, "accounts.json"))
    return _store


def reset_store():
    """Tests: re-read ACCOUNTS_STORE."""
    global _store
    _store = None


# ---------------------------------------------------------------- secrets

_secret = None


def _signing_key():
    global _secret
    if _secret is None:
        env = os.environ.get("SESSION_SECRET")
        if env:
            _secret = env.encode()
        else:
            path = os.path.join(DATA_DIR, "session_secret")
            try:
                with open(path, "rb") as f:
                    _secret = f.read().strip()
            except FileNotFoundError:
                os.makedirs(DATA_DIR, exist_ok=True)
                _secret = secrets.token_hex(32).encode()
                with open(path, "wb") as f:
                    f.write(_secret)
    return _secret


def _hash_pw(password, salt):
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


def _key(email):
    return "user:" + email


# ---------------------------------------------------------------- accounts

def normalize_email(email):
    return (email or "").strip().lower()


def public(user):
    """What the browser may see."""
    return {"name": user.get("name", ""), "email": user["email"], "l1": user.get("l1"),
            "created": user.get("created")}


def signup(email, password, name=""):
    email = normalize_email(email)
    name = (name or "").strip()[:60]
    if not EMAIL_RE.match(email):
        raise AccountError("Please enter a valid email address.")
    if not isinstance(password, str) or len(password) < 8:
        raise AccountError("Please choose a password of at least 8 characters.")
    if len(password) > 200:
        raise AccountError("That password is too long.")
    if store().get(_key(email)) is not None:
        raise AccountError("An account with that email already exists. Try logging in.")
    salt = secrets.token_bytes(16)
    user = {"email": email, "name": name, "l1": None, "created": int(time.time()),
            "salt": base64.b64encode(salt).decode(), "pw": base64.b64encode(_hash_pw(password, salt)).decode()}
    store().put(_key(email), user)
    return user


def login(email, password):
    email = normalize_email(email)
    user = store().get(_key(email))
    if user is None or not isinstance(password, str) or len(password) > 200:
        _hash_pw("x" * 8, b"0" * 16)  # similar timing whether or not the account exists
        raise AccountError("Email or password is incorrect.")
    expected = base64.b64decode(user["pw"])
    if not hmac.compare_digest(expected, _hash_pw(password, base64.b64decode(user["salt"]))):
        raise AccountError("Email or password is incorrect.")
    return user


def get_user(email):
    return store().get(_key(normalize_email(email)))


def update_profile(email, **fields):
    user = get_user(email)
    if user is None:
        raise AccountError("Please log in again.")
    for k in ("name", "l1"):
        if k in fields:
            user[k] = fields[k]
    store().put(_key(user["email"]), user)
    return user


def delete_user(email):
    store().delete(_key(normalize_email(email)))


# ---------------------------------------------------------------- sessions

def make_token(email, days=SESSION_DAYS):
    payload = f"{normalize_email(email)}|{int(time.time()) + days * 86400}"
    sig = hmac.new(_signing_key(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()


def read_token(token):
    """Email for a valid, unexpired token, else None."""
    try:
        email, exp, sig = base64.urlsafe_b64decode(token.encode()).decode().rsplit("|", 2)
    except Exception:
        return None
    good = hmac.new(_signing_key(), f"{email}|{exp}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(good, sig) or int(exp) < time.time():
        return None
    return email
