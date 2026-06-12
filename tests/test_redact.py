from ragnite.claude.redact import (
    contains_secret,
    is_ignored,
    is_sensitive_path,
    load_ragniteignore,
    redact,
)

SECRET_SAMPLES = [
    "export OPENAI_API_KEY=sk-proj1234567890abcdefghij",
    "anthropic key sk-ant-api03-aaaaaaaaaaaaaaaaaaaa",
    "aws AKIAIOSFODNN7EXAMPLE here",
    "github token ghp_abcdefghijklmnopqrstuvwxyz123456",
    "github_pat_11AAAAAA0abcdefghijklmnopq",
    "slack xoxb-1234567890-abcdefghijklm",
    "Authorization: Bearer abcdef1234567890abcdef",
    "password = hunter2hunter2",
    "api_key: 'supersecretvalue123'",
    "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9P",
]


def test_secret_patterns_are_redacted():
    for sample in SECRET_SAMPLES:
        cleaned = redact(sample)
        assert "[REDACTED" in cleaned, sample
        assert contains_secret(sample)


def test_connection_url_password_redacted():
    cleaned = redact("DATABASE_URL=postgres://app:S3cretPass@db.internal:5432/prod")
    assert "S3cretPass" not in cleaned
    assert "postgres://app:[REDACTED]@db.internal:5432/prod" in cleaned


def test_private_key_block_redacted():
    key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\nxyz\n-----END RSA PRIVATE KEY-----"
    assert redact(key) == "[REDACTED PRIVATE KEY]"


def test_normal_text_untouched():
    text = "The database listens on port 5432 and the API uses RS256 tokens."
    assert redact(text) == text
    assert not contains_secret(text)


def test_sensitive_paths():
    for path in (
        ".env",
        ".env.production",
        "id_rsa",
        "id_ed25519.pub",
        "server.pem",
        "keystore.jks",
        "credentials.json",
        ".npmrc",
    ):
        assert is_sensitive_path(path), path
    for path in ("main.py", "README.md", "docker-compose.yml", "environment.md"):
        assert not is_sensitive_path(path), path


def test_ragniteignore(tmp_path):
    (tmp_path / ".ragniteignore").write_text("# comment\nprivate/\n*.secret.md\nnotes\n", encoding="utf-8")
    patterns = load_ragniteignore(tmp_path)
    assert is_ignored("private/plan.md", patterns)
    assert is_ignored("docs/roadmap.secret.md", patterns)
    assert is_ignored("notes", patterns)
    assert not is_ignored("src/main.py", patterns)
    assert load_ragniteignore(tmp_path / "nowhere") == []
