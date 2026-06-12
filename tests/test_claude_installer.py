import json

from ragnite.claude.installer import HOOK_SPECS, install_into, merge_hooks, merge_mcp


def _settings(root):
    return json.loads((root / ".claude" / "settings.local.json").read_text(encoding="utf-8"))


def test_install_creates_all_artifacts(tmp_path):
    actions = install_into(tmp_path)
    assert (tmp_path / ".claude" / "skills" / "ragnite" / "SKILL.md").exists()
    assert (tmp_path / ".mcp.json").exists()
    assert (tmp_path / ".ragnite" / "config.toml").exists()
    assert (tmp_path / ".ragnite" / "session.json").exists()
    settings = _settings(tmp_path)
    assert set(settings["hooks"]) == {event for event, *_ in HOOK_SPECS}
    skill = (tmp_path / ".claude" / "skills" / "ragnite" / "SKILL.md").read_text(encoding="utf-8")
    assert "/ragnite invoke" in skill and "/ragnite init" in skill
    mcp = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert "ragnite" in mcp["mcpServers"]
    assert len(actions) >= 4

    session = json.loads((tmp_path / ".ragnite" / "session.json").read_text(encoding="utf-8"))
    assert session["active"] is False  # install never auto-activates


def test_install_preserves_existing_settings_and_servers(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    existing_settings = {
        "permissions": {"allow": ["Bash(npm test:*)"]},
        "hooks": {
            "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo existing"}]}]
        },
    }
    (claude_dir / "settings.local.json").write_text(json.dumps(existing_settings), encoding="utf-8")
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "other-server"}}}), encoding="utf-8"
    )

    install_into(tmp_path)

    settings = _settings(tmp_path)
    assert settings["permissions"] == {"allow": ["Bash(npm test:*)"]}  # untouched
    post_tool = settings["hooks"]["PostToolUse"]
    commands = [h["command"] for entry in post_tool for h in entry["hooks"]]
    assert "echo existing" in commands  # pre-existing hook preserved
    assert any("ragnite.cli claude hook post-tool" in c for c in commands)  # ours added

    mcp = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert "other" in mcp["mcpServers"] and "ragnite" in mcp["mcpServers"]


def test_install_is_idempotent(tmp_path):
    install_into(tmp_path)
    first = _settings(tmp_path)
    install_into(tmp_path)
    second = _settings(tmp_path)
    assert first == second  # no duplicated hook entries on re-install
    for event in second["hooks"]:
        ragnite_entries = [
            entry
            for entry in second["hooks"][event]
            for h in entry["hooks"]
            if "ragnite.cli claude hook" in h["command"]
        ]
        assert len(ragnite_entries) == 1


def test_merge_functions_are_pure_and_idempotent():
    settings, changed = merge_hooks({})
    assert changed
    _, changed_again = merge_hooks(settings)
    assert not changed_again

    mcp, changed = merge_mcp({})
    assert changed and "ragnite" in mcp["mcpServers"]
    _, changed_again = merge_mcp(mcp)
    assert not changed_again


def test_gitignore_exact_line_check(tmp_path):
    # ".ragniteignore" in .gitignore must NOT mask the missing ".ragnite/" entry
    (tmp_path / ".gitignore").write_text(".ragniteignore\n", encoding="utf-8")
    install_into(tmp_path)
    content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".ragnite/" in content.splitlines()
    # second install must not duplicate the entry
    install_into(tmp_path)
    lines = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines.count(".ragnite/") == 1


def test_install_does_not_overwrite_existing_config_toml(tmp_path):
    ragnite_dir = tmp_path / ".ragnite"
    ragnite_dir.mkdir()
    (ragnite_dir / "config.toml").write_text("[invoke]\nstrict = true\n", encoding="utf-8")
    install_into(tmp_path)
    assert "strict = true" in (ragnite_dir / "config.toml").read_text(encoding="utf-8")
