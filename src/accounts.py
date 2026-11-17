"""User accounts: sign-up, login, signed session cookies, profile (first language).

Storage is pluggable via the ACCOUNTS_STORE environment variable:
  (unset)                      a JSON file at data/accounts.json (local app)
  /some/path.json              a JSON file at that path
  dir:/some/dir                one JSON file per user in that directory
  modal-volume:<name>:/mount   one file per user on a mounted, writable Modal Volume
                               (hosted; durable, committed after every write)
  modal-dict:<name>            a modal.Dict (legacy: entries expire after 7 idle days,
                               so don't use it for accounts)

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


class DirStore:
    """One JSON file per key, named by a hash of the key (so emails never become paths).

    With a Modal Volume, `volume.commit()` makes each write durable and visible to
    other containers, and `volume.reload()` (at most every `reload_every` s) picks
    up their writes."""

    def __init__(self, path, volume=None, reload_every=2.0):
        self.path = path
        self.volume = volume
        self.reload_every = reload_every
        self._last_reload = 0.0
        self.lock = threading.Lock()
        os.makedirs(path, exist_ok=True)

    def _file(self, key):
        return os.path.join(self.path, hashlib.sha256(key.encode("utf-8")).hexdigest()[:40] + ".json")

    def _refresh(self, force=False):
        if self.volume is not None and (force or time.time() - self._last_reload > self.reload_every):
            try:
                self.volume.reload()
            except Exception as e:  # e.g. a file briefly open in this container
                print(f"[accounts] volume reload skipped: {e}")
            self._last_reload = time.time()

    def _commit(self):
        if self.volume is not None:
            self.volume.commit()

    def get(self, key):
        with self.lock:
            self._refresh()
            try:
                with open(self._file(key), encoding="utf-8") as f:
                    return json.load(f)
            except (FileNotFoundError, ValueError):
                return None

    def put(self, key, value):
        with self.lock:
            path = self._file(key)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(value, f)
            os.replace(tmp, path)
            self._commit()

    def delete(self, key):
        with self.lock:
            self._refresh(force=True)
            try:
                os.remove(self._file(key))
            except FileNotFoundError:
                return
            self._commit()


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
        elif spec.startswith("modal-volume:"):
            import modal
            _, name, mount = spec.split(":", 2)
            _store = DirStore(os.path.join(mount, "accounts"), volume=modal.Volume.from_name(name))
        elif spec.startswith("dir:"):
            _store = DirStore(spec.split(":", 1)[1])
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
    email = normalize_email(email)
    store().delete(_key(email))
    store().delete(_progress_key(email))


# ---------------------------------------------------------------- progress
# Practice history (sentences, scores, per-sound averages; never audio) and
# minimal-pair tallies, synced so they follow the account across devices.

PROGRESS_MAX_BYTES = 400_000
PROGRESS_MAX_ATTEMPTS = 500


def _progress_key(email):
    return "progress:" + email


def get_progress(email):
    return store().get(_progress_key(normalize_email(email))) or {"attempts": [], "pairs": {}, "updated": 0}


def put_progress(email, attempts, pairs):
    if not isinstance(attempts, list) or not isinstance(pairs, dict):
        raise AccountError("Couldn't save your progress.")
    attempts = [a for a in attempts if isinstance(a, dict) and isinstance(a.get("t"), (int, float))]
    attempts = sorted(attempts, key=lambda a: a["t"])[-PROGRESS_MAX_ATTEMPTS:]
    pairs = {str(k)[:40]: {"ok": int(v.get("ok", 0)), "n": int(v.get("n", 0))}
             for k, v in list(pairs.items())[:100] if isinstance(v, dict)}
    data = {"attempts": attempts, "pairs": pairs, "updated": int(time.time())}
    while len(json.dumps(data)) > PROGRESS_MAX_BYTES and data["attempts"]:
        data["attempts"] = data["attempts"][len(data["attempts"]) // 4 + 1:]  # drop the oldest quarter
    store().put(_progress_key(normalize_email(email)), data)
    return data


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
