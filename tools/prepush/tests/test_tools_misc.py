"""gitleaks 설치기·버전 경고·올리기 도우미의 순수 부분."""

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

import gitleaks_tool
import push_check_main
import versions


def test_gitleaks_pin_matches_ci_workflow():
    """CI(secret-scan.yml)와 같은 버전·같은 linux 체크섬을 쓰는지 — 한쪽만 올리면 여기서 걸린다."""
    wf = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "secret-scan.yml").read_text(encoding="utf-8")
    assert f"GITLEAKS_VERSION: {gitleaks_tool.VERSION}" in wf
    assert f"GITLEAKS_SHA256: {gitleaks_tool.SHA256['linux_x64']}" in wf
    pre_commit = (Path(__file__).resolve().parents[3] / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert f"rev: v{gitleaks_tool.VERSION}" in pre_commit


def test_verify_sha256():
    data = b"gitleaks"
    gitleaks_tool.verify_sha256(data, hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="체크섬"):
        gitleaks_tool.verify_sha256(data, "0" * 64)


def test_extract_binary_takes_only_the_exe():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("LICENSE", "mit")
        zf.writestr("../../evil/gitleaks.exe", "EXE")  # 경로는 무시하고 내용만 읽는다
    assert gitleaks_tool.extract_binary(buf.getvalue(), "gitleaks.exe") == b"EXE"
    with pytest.raises(ValueError):
        gitleaks_tool.extract_binary(buf.getvalue(), "nope.exe")


def test_tools_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_OFFICE_TOOLS_DIR", str(tmp_path))
    assert gitleaks_tool.exe_path().parent == tmp_path / f"gitleaks-{gitleaks_tool.VERSION}"


def test_requirement_specs():
    text = "# c\n-r base.txt\nstreamlit>=1.43,<1.67   # 메모\nlangchain_core>=0.3.46,<1.0\nst-click-detector==0.1.3\n"
    specs = versions.requirement_specs(text)
    assert specs["streamlit"] == ">=1.43,<1.67"
    assert specs["langchain-core"] == ">=0.3.46,<1.0"
    assert specs["st-click-detector"] == "==0.1.3"


def test_latest_allowed_skips_prereleases_and_out_of_range():
    vs = ["1.43.2", "1.66.1", "1.67.0", "1.66.2rc1", "junk"]
    assert versions.latest_allowed(vs, ">=1.43,<1.67") == "1.66.1"
    assert versions.latest_allowed(vs, ">=2") is None


def test_compare_only_warns_on_difference():
    lines = versions.compare({"streamlit": "1.43.2", "langgraph": "0.5.4", "langchain-core": None},
                             {"streamlit": "1.66.1", "langgraph": "0.5.4", "langchain-core": "0.3.80"})
    assert len(lines) == 2
    assert lines[0].startswith("streamlit: 이 PC 1.43.2 / CI 1.66.1")
    assert "설치 안 됨" in lines[1]


def test_pypi_cache_used_without_network(tmp_path, monkeypatch):
    import json
    import time

    cache = tmp_path / "c.json"
    cache.write_text(json.dumps({"streamlit": {"at": time.time(), "versions": ["1.0"]}}), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("네트워크를 쓰면 안 됨")

    monkeypatch.setattr(versions.urllib.request, "urlopen", boom)
    assert versions.pypi_versions("streamlit", cache) == ["1.0"]


@pytest.mark.parametrize("remote", [
    "https://github.com/jeong-yun/Project.git",
    "https://github.com/jeong-yun/Project",
    "git@github.com:jeong-yun/Project.git",
    "ssh://git@github.com/jeong-yun/Project.git",
])
def test_web_url(remote):
    assert push_check_main.web_url(remote) == "https://github.com/jeong-yun/Project"
    assert push_check_main.actions_url("https://github.com/jeong-yun/Project").endswith(
        "/actions/workflows/promote-to-main.yml?query=branch%3Acheck%2Fmain")


def test_web_url_unknown_host():
    assert push_check_main.web_url("https://gitlab.com/a/b.git") is None
