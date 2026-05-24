from __future__ import annotations
import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
import os

try:
    import git as _git
except ImportError:
    _git = None
logger = logging.getLogger(__name__)


def _make_commit_message(messages) -> str:
    if isinstance(messages, str):
        return messages
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), None
    )
    if last_user and isinstance(last_user, str):
        title = last_user.replace("\n", " ").strip()[:30]
    body = json.dumps(messages, indent=2, ensure_ascii=False)
    return f"{title}\n\n{body}"


def _parse_messages_from_commit(raw: str) -> list[dict]:
    if not raw or raw in ("snapshot", "update"):
        return []
    try:
        parts = raw.split("\n\n", 1)
        body = parts[1] if len(parts) == 2 else parts[0]
        msgs = json.loads(body)
        return msgs if isinstance(msgs, list) else []
    except (json.JSONDecodeError, IndexError):
        return []


def git_add_filtered(repo):
    ignores = [
        "_*",
        "*.bin",
        "*.gguf",
        "*.safetensors",
        "*.pt",
        "*.pth",
        ".cache/",
        ".log.json",
        "*.egg-info/",
        ".eggs/",
        "build/",
        "dist/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache/",
        ".tox/",
        ".nox/",
        ".coverage",
        "htmlcov/",
        ".venv/",
        "venv/",
        "env/",
        ".DS_Store",
        "Thumbs.db",
    ]
    args = ["-A", "--", "."]
    for pat in ignores:
        args.append(f":(exclude){pat}")
    repo.git.add(*args, with_exceptions=False)


@dataclass(frozen=True)
class LedgerPoint:
    branch: str
    commit: str
    worktree: str


def _repo(path: str) -> _git.Repo:
    return _git.Repo(path, search_parent_directories=True)


def create_worktree(
    repo_dir: str,
    *,
    worktree_dir: str | None = None,
    ref: str = "HEAD",
    prefix: str = "agent",
    skip_if_missing: bool = False,
) -> LedgerPoint | None:
    if _git is None:
        logger.warning("pip install GitPython")
        return None
    try:
        try:
            repo = _repo(repo_dir)
        except _git.InvalidGitRepositoryError:
            if skip_if_missing:
                return None
            repo = _git.Repo.init(repo_dir)
            gi = os.path.join(repo_dir, ".gitignore")
            if not os.path.exists(gi):
                with open(gi, "w") as f:
                    ignores = [
                        "__pycache__/",
                        "*.pyc",
                        ".DS_Store",
                        "*.bin",
                        "*.safetensors",
                        "*.gguf",
                        "*.pt",
                        "*.pth",
                        ".cache/",
                        "_*",
                    ]
                    f.write("\n".join(ignores))
        with repo.config_writer() as cw:
            if not cw.has_option("user", "email"):
                cw.set_value("user", "email", "agent@local")
            if not cw.has_option("user", "name"):
                cw.set_value("user", "name", "agent")
        git_add_filtered(repo)
        if not repo.head.is_valid() or repo.index.diff("HEAD") or repo.untracked_files:
            repo.index.commit("snapshot")
        repo.git.worktree("prune")
        base = repo.commit(ref)
        _uuid = uuid.uuid4().hex
        branch = f"{prefix}/{base.hexsha}-{_uuid}"
        if worktree_dir is None:
            worktree_dir = Path(repo.git.rev_parse("--show-toplevel")).parent / _uuid
        worktree_dir = os.path.abspath(worktree_dir)
        repo.git.worktree("add", "-b", branch, worktree_dir, base.hexsha)
        point = LedgerPoint(branch=branch, commit=base.hexsha, worktree=worktree_dir)
        logger.info(point)
        return point
    except Exception as e:
        logger.exception(e)
        return None


def commit_worktree(
    point: LedgerPoint | None, message: str = "update"
) -> LedgerPoint | None:
    if point is None:
        return None
    try:
        repo = _repo(point.worktree)
        git_add_filtered(repo)
        has_changes = bool(repo.index.diff("HEAD") or repo.untracked_files)
        diff_info = ""
        if has_changes:
            try:
                diff_stat = repo.git.diff("HEAD", stat=True)
                diff_patch = repo.git.diff("HEAD")
                diff_info = (
                    f"\n--- DIFF STAT ---\n{diff_stat}\n--- PATCH ---\n{diff_patch}"
                )
            except Exception as de:
                diff_info = f"\n(Could not capture diff: {de})"
        else:
            return LedgerPoint(
                branch=point.branch,
                commit=repo.head.commit.hexsha
                if repo.head.is_valid()
                else point.commit,
                worktree=point.worktree,
            )
        commit = repo.index.commit(_make_commit_message(message))
        new_point = LedgerPoint(
            branch=point.branch, commit=commit.hexsha, worktree=point.worktree
        )
        logger.info("%s%s", new_point, diff_info)
        return new_point
    except Exception as e:
        logger.exception(e)
        return None


def cleanup_worktree(point: LedgerPoint, *, remove_branch: bool = False) -> None:
    root_repo = None
    try:
        worktree_repo = _repo(point.worktree)
        root_path = worktree_repo.git.rev_parse("--show-toplevel")
        root_repo = _repo(root_path)
        worktree_repo.close()
    except Exception as e:
        logger.warning("Could not map or resolve root repository tracking: %s", e)
    if root_repo:
        try:
            root_repo.git.worktree("remove", "--force", point.worktree)
        except Exception as e:
            logger.warning("git worktree remove failed: %s", e)
    if os.path.exists(point.worktree):
        try:
            shutil.rmtree(point.worktree, ignore_errors=True)
        except Exception as e:
            logger.warning("File-system cleanup failed for %s: %s", point.worktree, e)
    if remove_branch and root_repo:
        try:
            root_repo.git.branch("-D", point.branch)
        except Exception as e:
            logger.warning("git branch deletion failed: %s", e)


def resume_worktree(
    repo_dir: str,
    commit: str,
    *,
    worktree_dir: str | None = None,
    prefix: str = "resume",
) -> tuple[LedgerPoint, list[dict]] | None:
    if _git is None:
        logger.warning("pip install GitPython")
        return (None, [])
    try:
        repo = _repo(repo_dir)
        base = repo.commit(commit)
        repo.git.worktree("prune")
        _uuid = uuid.uuid4().hex
        branch = f"{prefix}/{base.hexsha}-{_uuid}"
        if worktree_dir is None:
            worktree_dir = Path(repo.git.rev_parse("--show-toplevel")).parent / _uuid
        worktree_dir = os.path.abspath(worktree_dir)
        repo.git.worktree("add", "-b", branch, worktree_dir, base.hexsha)
        point = LedgerPoint(branch=branch, commit=base.hexsha, worktree=worktree_dir)
        logger.info("Resumed worktree at %s", point)
        messages: list[dict] = []
        raw = base.message.strip()
        try:
            parts = raw.split("\n\n", 1)
            body = parts[1] if len(parts) == 2 else parts[0]
            messages = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("Could not parse message history from commit %s", commit)
        return (point, messages)
    except Exception as e:
        logger.exception(e)
        return (None, [])


def current_point(worktree_dir: str) -> LedgerPoint:
    repo = _repo(worktree_dir)
    try:
        branch_name = repo.active_branch.name
    except (TypeError, AttributeError):
        branch_name = "DETACHED"
    commit_sha = repo.head.commit.hexsha if repo.head.is_valid() else ""
    return LedgerPoint(branch=branch_name, commit=commit_sha, worktree=worktree_dir)
