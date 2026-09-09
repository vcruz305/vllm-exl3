"""Unit tests asserting upstream attribution and strict repository sanitization."""

import os
import re
import tomllib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_internal_hostnames_or_usernames():
    """Ensure zero occurrences of forbidden hostnames in documentation or source files."""
    forbidden = ["cruz" + "-spark", "we" + "sche" + "-spark", "9f" + "73", "mark" + "us"]
    text_extensions = {".md", ".py", ".cu", ".cuh", ".cpp", ".toml", ".cff"}
    violations = []
    for root, dirs, files in os.walk(REPO_ROOT):
        if ".git" in root or "__pycache__" in root or ".pytest_cache" in root or ".agent-sync" in root or "dist" in root or "build" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in text_extensions or file in {"NOTICE", "LICENSE", "CITATION"}:
                path = os.path.join(root, file)
                try:
                    text = open(path, "r", encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                for term in forbidden:
                    if term in text:
                        violations.append(f"{term} found in {os.path.relpath(path, REPO_ROOT)}")
    assert not violations, "Forbidden internal terms detected:\n" + "\n".join(violations)


def test_upstream_attribution_present():
    """Require visible MiaAI-Lab and ExLlamaV3 attribution plus detailed notices."""
    readme = open(os.path.join(REPO_ROOT, "README.md"), "r", encoding="utf-8").read()
    assert "MiaAI-Lab" in readme or "Mia's AI Lab" in readme
    assert "turboderp" in readme or "ExLlamaV3" in readme

    notices_path = os.path.join(REPO_ROOT, "THIRD_PARTY_NOTICES.md")
    assert os.path.exists(notices_path), "THIRD_PARTY_NOTICES.md missing"
    notices = open(notices_path, "r", encoding="utf-8").read()
    assert "Mia's AI Lab" in notices
    assert "Turboderp" in notices or "turboderp" in notices
    assert "4b8d3c7" in notices, "historical E2 source commit must remain identified"


def test_package_version_and_metadata():
    """Verify package identity, current release, and project license structurally."""
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
        project = tomllib.load(f)["project"]
    assert project["name"] == "vllm-exl3"
    assert project["version"] == "0.4.2"
    assert project["license"] == "AGPL-3.0-only"

    setup_content = open(os.path.join(REPO_ROOT, "setup.py"), "r", encoding="utf-8").read()
    assert 'name="vllm-exl3"' in setup_content


def test_canonical_turboderp_org_urls():
    """Verify ExLlamaV3 repository URLs use the canonical turboderp-org namespace."""
    bad_pattern = re.compile(r"https?://github\.com/turboderp/exllamav3[^\s\)\"\`\>]*")
    violations = []
    text_extensions = {".md", ".py", ".toml", ".cff"}
    for root, dirs, files in os.walk(REPO_ROOT):
        if ".git" in root or "__pycache__" in root or ".pytest_cache" in root or "dist" in root or "build" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in text_extensions or file in {"NOTICE", "LICENSE", "CITATION"}:
                path = os.path.join(root, file)
                try:
                    text = open(path, "r", encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                matches = bad_pattern.findall(text)
                if matches:
                    violations.append(f"{os.path.relpath(path, REPO_ROOT)}: {matches}")
    assert not violations, "Legacy turboderp/exllamav3 URLs found:\n" + "\n".join(violations)


def test_local_markdown_links_resolve():
    """Verify relative links in top-level documentation point to existing files."""
    for md_file in ["README.md", "THIRD_PARTY_NOTICES.md", "ACKNOWLEDGMENTS.md", "AGENTS.md"]:
        path = os.path.join(REPO_ROOT, md_file)
        if not os.path.exists(path):
            continue
        content = open(path, "r", encoding="utf-8").read()
        for link in re.findall(r"\[.*?\]\((?!https?://)(.*?)\)", content):
            clean = link.split("#")[0].strip()
            if clean:
                assert os.path.exists(os.path.join(REPO_ROOT, clean)), f"Broken link in {md_file}: {link}"
