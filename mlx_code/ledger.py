from __future__ import annotations
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
import os
import git as _git

logger = logging.getLogger(__name__)

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
    worktree_dir: str|None = None,
    ref: str = "HEAD",
    prefix: str = "agent",
    skip_if_missing = False, # □ must be turned True before commit!
) -> LedgerPoint | None:
    try:
        try:
            repo = _repo(repo_dir)
        except _git.InvalidGitRepositoryError:
            if skip_if_missing:
                return
            repo = _git.Repo.init(repo_dir)
            gi = os.path.join(repo_dir, '.gitignore')
            if not os.path.exists(gi):
                with open(gi, 'w') as f:
                    ignores = [
                        "__pycache__/",
                        "*.pyc",
                        ".DS_Store",
                        "*.bin",
                        "*.safetensors",
                        "*.gguf",
                        "*.pt",
                        "*.pth",
                        ".cache/"
                    ]
                    f.write('\n'.join(ignores))


        with repo.config_writer() as cw:
            if not cw.has_option("user", "email"):
                cw.set_value("user", "email", "agent@local")
            if not cw.has_option("user", "name"):
                cw.set_value("user", "name", "agent")

        repo.git.add("-A")
        if not repo.head.is_valid() or repo.index.diff("HEAD") or repo.untracked_files:
            repo.index.commit("snapshot")

        repo.git.worktree("prune")
        base = repo.commit(ref)

        _uuid = uuid.uuid4().hex
        branch = f"{prefix}/{base.hexsha}-{_uuid}"
        worktree_dir = Path(repo.git.rev_parse("--show-toplevel")).parent / _uuid if worktree_dir is None else worktree_dir
        worktree_dir = os.path.abspath(worktree_dir)

        repo.git.worktree(
            "add",
            "-b",
            branch,
            worktree_dir,
            base.hexsha,
        )

        point = LedgerPoint(
            branch=branch,
            commit=base.hexsha,
            worktree=worktree_dir,
        )

        logger.info(point)

        return point
    except Exception as e:
        logger.exception(e)
        return None


def commit_worktree(
    point: LedgerPoint | None,
    message: str = "update",
) -> LedgerPoint | None:
    if point is None:
        return
    try:
        repo = _repo(point.worktree)

        repo.git.add("-A")
        if not repo.index.diff("HEAD"):
            return LedgerPoint(branch=point.branch,
                               commit=repo.head.commit.hexsha,
                               worktree=point.worktree)
        commit = repo.index.commit(message[:512])


        new_point = LedgerPoint(
            branch=point.branch,
            commit=commit.hexsha,
            worktree=point.worktree,
        )

        logger.info(new_point)

        return new_point

    except Exception as e:
        logger.exception(e)
        return None

def restore_worktree(
    repo_dir: str,
    worktree_dir: str,
    point: LedgerPoint,
) -> LedgerPoint:
    repo = _repo(repo_dir)

    try:
        repo.git.worktree("prune")
    except Exception:
        pass

    repo.git.worktree(
        "add",
        worktree_dir,
        point.branch,
    )

    restored = LedgerPoint(
        branch=point.branch,
        commit=point.commit,
        worktree=worktree_dir,
    )

    logger.info(restored)
    return restored

def cleanup_worktree(
    point: LedgerPoint,
    *,
    remove_branch: bool = False,
) -> None:
    repo = _repo(point.worktree)
    root = repo.git.rev_parse("--show-toplevel")
    root_repo = _repo(root)
    try:
        root_repo.git.worktree(
            "remove",
            "--force",
            point.worktree,
        )
    except Exception as e:
        logger.warning("git cleanup: %s", e)

    try:
        shutil.rmtree(point.worktree, ignore_errors=True)
    except Exception:
        pass

    if remove_branch:
        try:
            root_repo.git.branch("-D", point.branch)
        except Exception as e:
            logger.warning("git cleanup branch: %s", e)


def current_point(worktree_dir: str) -> LedgerPoint:
    repo = _repo(worktree_dir)
    
    try:
        branch_name = repo.active_branch.name
    except (TypeError, AttributeError):
        branch_name = "DETACHED"

    return LedgerPoint(
        branch=branch_name,
        commit=repo.head.commit.hexsha,
        worktree=worktree_dir,
    )
