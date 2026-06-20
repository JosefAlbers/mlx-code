from __future__ import annotations
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
logger = logging.getLogger(__name__)
_ADD_EXCLUDES = ['_*', '*.bin', '*.gguf', '*.safetensors', '*.pt', '*.pth', '.cache/', '.log.json', '*.egg-info/', '.eggs/', 'build/', 'dist/', '__pycache__/', '*.pyc', '*.pyo', '*.pyd', '.pytest_cache/', '.tox/', '.nox/', '.coverage', 'htmlcov/', '.venv/', 'venv/', 'env/', '.DS_Store', 'Thumbs.db']

class GitError(RuntimeError):
    pass

def _git(cwd: str, *args: str, check: bool=True) -> str:
    try:
        r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=check)
        return r.stdout.strip()
    except subprocess.CalledProcessError as exc:
        args_str = repr(' '.join(args))
        raise GitError(f'git {args_str} failed in {cwd!r}: {exc.stderr.strip()}') from exc
    except FileNotFoundError as exc:
        if not os.path.exists(cwd):
            raise GitError(f'git working directory does not exist: {cwd!r}') from exc
        raise GitError('git executable not found in PATH') from exc

def _make_commit_message(messages) -> str:
    if isinstance(messages, str):
        return messages
    filtered = [m for m in messages if m.get('role') != 'commit']
    last_user = next((m['content'] for m in reversed(filtered) if m.get('role') == 'user'), None)
    title = last_user.replace('\n', ' ').strip()[:60] if isinstance(last_user, str) else 'update'
    body = json.dumps(filtered, indent=2, ensure_ascii=False)
    return f'{title}\n\n--- BEGIN MESSAGES ---\n{body}'

def _parse_messages_from_commit(raw: str) -> list[dict]:
    if not raw or raw in ('snapshot', 'update'):
        return []
    marker = '--- BEGIN MESSAGES ---\n'
    idx = raw.find(marker)
    if idx != -1:
        payload = raw[idx + len(marker):]
    else:
        parts = raw.split('\n\n', 1)
        payload = parts[1] if len(parts) == 2 else parts[0]
    try:
        msgs = json.loads(payload)
        return msgs if isinstance(msgs, list) else []
    except json.JSONDecodeError:
        logger.warning('_parse_messages_from_commit: could not parse JSON from commit body')
        return []

def _count_user_turns(commit_body: str) -> int:
    messages = _parse_messages_from_commit(commit_body)
    if messages:
        return sum((1 for m in messages if m.get('role') == 'user'))
    return 0

def git_add_filtered(cwd: str) -> None:
    excludes = [f':(exclude){p}' for p in _ADD_EXCLUDES]
    try:
        _git(cwd, 'add', '-A', '--', '.', *excludes)
    except GitError as e:
        logger.warning('git add warning (ignored): %s', e)

@dataclass(frozen=True)
class LedgerPoint:
    branch: str
    commit: str
    worktree: str

def create_worktree(repo_dir: str, *, worktree_dir: str | None=None, ref: str='HEAD', prefix: str='agent', skip_if_missing: bool=False) -> LedgerPoint | None:
    try:
        repo_dir = os.path.abspath(repo_dir)
        try:
            root = _git(repo_dir, 'rev-parse', '--show-toplevel')
        except GitError:
            if skip_if_missing:
                return None
            _git(repo_dir, 'init')
            root = repo_dir
            gi = os.path.join(root, '.gitignore')
            if not os.path.exists(gi):
                Path(gi).write_text('\n'.join(['_log.json']))
        if not _git(root, 'config', 'user.email', check=False):
            _git(root, 'config', 'user.email', 'agent@local')
        if not _git(root, 'config', 'user.name', check=False):
            _git(root, 'config', 'user.name', 'agent')
        git_add_filtered(root)
        head_valid = bool(_git(root, 'rev-parse', '--verify', 'HEAD', check=False))
        has_changes = bool(_git(root, 'status', '--porcelain'))
        if not head_valid or has_changes:
            _git(root, 'commit', '--allow-empty', '-m', 'snapshot')
        _git(root, 'worktree', 'prune')
        base_sha = _git(root, 'rev-parse', ref)
        _uuid = uuid.uuid4().hex
        branch = f'{prefix}--{base_sha[:12]}-{_uuid[:12]}'
        if worktree_dir is None:
            worktree_dir = str(Path(root).parent / _uuid)
        worktree_dir = os.path.abspath(worktree_dir)
        _git(root, 'worktree', 'add', '-b', branch, worktree_dir, base_sha)
        point = LedgerPoint(branch=branch, commit=base_sha, worktree=worktree_dir)
        logger.info(point)
        return point
    except Exception as e:
        logger.exception('create_worktree failed: %s', e)
        return None

def commit_worktree(point: LedgerPoint | None, message: str='update') -> tuple[LedgerPoint | None, str]:
    if point is None:
        return (None, '')
    try:
        git_add_filtered(point.worktree)
        if not _git(point.worktree, 'status', '--porcelain'):
            sha = _git(point.worktree, 'rev-parse', 'HEAD', check=False) or point.commit
            return (LedgerPoint(branch=point.branch, commit=sha, worktree=point.worktree), '')
        diff_stat = ''
        try:
            stat = _git(point.worktree, 'diff', '--stat', 'HEAD')
            diff_stat = stat
            patch = _git(point.worktree, 'diff', 'HEAD')
            diff_info = f'\n--- DIFF STAT ---\n{stat}\n--- PATCH ---\n{patch}'
        except Exception as de:
            diff_info = f'\n(Could not capture diff: {de})'
        _git(point.worktree, 'commit', '-m', _make_commit_message(message))
        new_sha = _git(point.worktree, 'rev-parse', 'HEAD')
        new_point = LedgerPoint(branch=point.branch, commit=new_sha, worktree=point.worktree)
        logger.info('%s%s', new_point, diff_info)
        return (new_point, diff_stat)
    except Exception as e:
        logger.exception('commit_worktree failed: %s', e)
        return (None, '')

def cleanup_worktree(point: LedgerPoint, *, remove_branch: bool=False) -> None:
    root: str | None = None
    if os.path.exists(point.worktree):
        git_common = _git(point.worktree, 'rev-parse', '--git-common-dir', check=False).strip()
        if git_common and os.path.isdir(git_common):
            root = str(Path(git_common).parent.resolve())
        if not root:
            root = _git(point.worktree, 'rev-parse', '--show-toplevel', check=False) or None
        if root:
            try:
                _git(root, 'worktree', 'remove', '--force', point.worktree)
            except Exception as e:
                logger.warning('git worktree remove failed: %s', e)
        try:
            shutil.rmtree(point.worktree, ignore_errors=True)
        except Exception as e:
            logger.warning('Filesystem cleanup failed for %s: %s', point.worktree, e)
    else:
        logger.warning('cleanup_worktree: worktree path %r no longer exists; skipping removal', point.worktree)
        try:
            root = _git(os.path.dirname(point.worktree), 'rev-parse', '--show-toplevel', check=False) or None
        except Exception:
            pass
    if root:
        try:
            _git(root, 'worktree', 'prune')
        except Exception as e:
            logger.warning('git worktree prune failed: %s', e)
    if remove_branch:
        if root:
            try:
                _git(root, 'branch', '-D', point.branch)
            except Exception as e:
                logger.warning('git branch deletion failed for %r: %s', point.branch, e)
        else:
            logger.warning('cleanup_worktree: could not locate repo root; branch %r was NOT deleted', point.branch)

def resume_worktree(repo_dir: str, commit: str, *, worktree_dir: str | None=None, prefix: str='resume') -> tuple[LedgerPoint | None, list[dict]]:
    try:
        root = _git(os.path.abspath(repo_dir), 'rev-parse', '--show-toplevel')
        base_sha = _git(root, 'rev-parse', commit)
        _git(root, 'worktree', 'prune')
        _uuid = uuid.uuid4().hex
        branch = f'{prefix}--{base_sha[:12]}-{_uuid[:12]}'
        if worktree_dir is None:
            worktree_dir = str(Path(root).parent / _uuid)
        worktree_dir = os.path.abspath(worktree_dir)
        _git(root, 'worktree', 'add', '-b', branch, worktree_dir, base_sha)
        point = LedgerPoint(branch=branch, commit=base_sha, worktree=worktree_dir)
        logger.info('Resumed worktree at %s', point)
        raw = _git(root, 'log', '-1', '--format=%B', base_sha)
        messages = _parse_messages_from_commit(raw)
        if not messages:
            logger.warning('Could not parse message history from commit %s', commit)
        return (point, messages)
    except Exception as e:
        logger.exception('resume_worktree failed: %s', e)
        return (None, [])

def git_new_branch(worktree: str, branch_name: str) -> LedgerPoint:
    git_add_filtered(worktree)
    if _git(worktree, 'status', '--porcelain'):
        _git(worktree, 'commit', '--allow-empty', '-m', 'snapshot')
    sha = _git(worktree, 'rev-parse', 'HEAD')
    full_name = f'{branch_name}--{sha[:12]}'
    _git(worktree, 'switch', '-c', full_name)
    point = LedgerPoint(branch=full_name, commit=sha, worktree=worktree)
    logger.info('Created branch %s at %s', full_name, sha[:8])
    return point

def git_new_branch_at(worktree: str, branch_name: str, ref: str) -> LedgerPoint:
    git_add_filtered(worktree)
    if _git(worktree, 'status', '--porcelain'):
        _git(worktree, 'commit', '--allow-empty', '-m', 'snapshot')
    sha = _git(worktree, 'rev-parse', ref)
    full_name = f'{branch_name}--{sha[:12]}'
    _git(worktree, 'branch', full_name, sha)
    _git(worktree, 'switch', full_name)
    point = LedgerPoint(branch=full_name, commit=sha, worktree=worktree)
    logger.info('Created branch %s at %s (from ref %s)', full_name, sha[:8], ref)
    return point

def git_switch_branch(worktree: str, branch_name: str) -> LedgerPoint:
    git_add_filtered(worktree)
    if _git(worktree, 'status', '--porcelain'):
        _git(worktree, 'commit', '--allow-empty', '-m', 'snapshot')
    _git(worktree, 'switch', branch_name)
    sha = _git(worktree, 'rev-parse', 'HEAD')
    point = LedgerPoint(branch=branch_name, commit=sha, worktree=worktree)
    logger.info('Switched to branch %s at %s', branch_name, sha[:8])
    return point

def current_point(worktree_dir: str) -> LedgerPoint:
    worktree_dir = os.path.abspath(worktree_dir)
    branch = _git(worktree_dir, 'symbolic-ref', '--short', 'HEAD', check=False) or 'DETACHED'
    sha = _git(worktree_dir, 'rev-parse', 'HEAD', check=False)
    return LedgerPoint(branch=branch, commit=sha, worktree=worktree_dir)

def get_commit_history_with_stats(worktree: str, limit: int=50) -> list[dict]:
    try:
        cmd = ['git', 'log', '--name-status', '--format=COMMIT:%H%nSHORT:%h%nREFS:%d%nSUBJECT:%s%nBODY_START%n%b%nBODY_END', f'-n{limit}']
        result = subprocess.run(cmd, cwd=worktree, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        commits = []
        current_commit = None
        in_body = False
        body_lines = []
        for line in output.split('\n'):
            if line.startswith('COMMIT:'):
                if current_commit:
                    body_text = '\n'.join(body_lines)
                    current_commit['user_turns'] = _count_user_turns(body_text)
                    commits.append(current_commit)
                current_commit = {'sha': line[7:], 'short_sha': '', 'refs': '', 'subject': '', 'files': []}
                in_body = False
                body_lines = []
            elif line.startswith('SHORT:'):
                if current_commit:
                    current_commit['short_sha'] = line[6:]
            elif line.startswith('REFS:'):
                if current_commit:
                    current_commit['refs'] = line[5:].strip()
            elif line.startswith('SUBJECT:'):
                if current_commit:
                    current_commit['subject'] = line[8:]
            elif line == 'BODY_START':
                in_body = True
            elif line == 'BODY_END':
                in_body = False
            elif current_commit:
                if in_body:
                    body_lines.append(line)
                elif line.strip():
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        current_commit['files'].append(f'{parts[0]} {parts[1]}')
                    else:
                        current_commit['files'].append(line)
        if current_commit:
            body_text = '\n'.join(body_lines)
            current_commit['user_turns'] = _count_user_turns(body_text)
            commits.append(current_commit)
        return commits
    except Exception as e:
        logger.warning(f'get_commit_history_with_stats failed: {e}')
        return []

def find_rev_commit(worktree: str, n: int, *, limit: int=0) -> str | None:
    limit_args = ['-n', str(limit)] if limit > 0 else []
    try:
        result = subprocess.run(['git', 'log', '--format=COMMIT:%H%n%b%nEND_BODY', *limit_args], cwd=worktree, capture_output=True, text=True, check=True)
        output = result.stdout
        best_exact: str | None = None
        best_below: str | None = None
        current_sha: str | None = None
        body_lines: list[str] = []
        for line in output.splitlines():
            if line.startswith('COMMIT:'):
                if current_sha is not None:
                    body = '\n'.join(body_lines)
                    count = _count_user_turns(body)
                    if count == n and best_exact is None:
                        best_exact = current_sha
                    if count < n and best_below is None:
                        best_below = current_sha
                current_sha = line[7:]
                body_lines = []
            elif line == 'END_BODY':
                pass
            else:
                body_lines.append(line)
        if current_sha is not None:
            body = '\n'.join(body_lines)
            count = _count_user_turns(body)
            if count == n and best_exact is None:
                best_exact = current_sha
            if count < n and best_below is None:
                best_below = current_sha
        chosen = best_exact or best_below
        if chosen:
            logger.info('find_rev_commit(n=%d): chose %s (exact=%s, below=%s)', n, chosen[:8], bool(best_exact), bool(best_below))
        else:
            logger.warning('find_rev_commit(n=%d): no suitable commit found', n)
        return chosen
    except Exception as e:
        logger.warning('find_rev_commit failed: %s', e)
        return None

def get_diff_between_refs(worktree: str, ref1: str='HEAD~1', ref2: str='HEAD') -> str:
    try:
        return _git(worktree, 'diff', ref1, ref2)
    except GitError:
        try:
            return _git(worktree, 'diff', '4b825dc642cb6eb9a060e54bf899d15d4a0d0e1d', ref2)
        except GitError as e2:
            raise GitError(f'Could not diff {ref1}..{ref2}: {e2}') from e2

def resolve_ref_short(worktree: str, ref: str) -> str:
    try:
        return _git(worktree, 'rev-parse', '--short', ref, check=False)
    except GitError:
        return ''

def get_branch_base_sha(worktree: str) -> str | None:
    branch = _git(worktree, 'symbolic-ref', '--short', 'HEAD', check=False)
    if branch:
        match = re.search('--([0-9a-f]{12})', branch)
        if match:
            sha_part = match.group(1)
            try:
                full_sha = _git(worktree, 'rev-parse', sha_part, check=False)
                if full_sha:
                    return full_sha
            except GitError:
                pass
    root_sha = _git(worktree, 'rev-list', '--max-parents=0', 'HEAD', check=False)
    return root_sha or None

def merge_branch_into_worktree(parent_gwt, child_gwt) -> tuple[bool, str]:
    if parent_gwt is None or child_gwt is None:
        out = f'Not git: parent_gwt={parent_gwt!r} child_gwt={child_gwt!r}'
        logger.debug(out)
        return (False, out)
    try:
        out = _git(parent_gwt.worktree, 'merge', '--no-edit', '--no-ff', child_gwt.branch)
        logger.info(out)
        return (True, out)
    except GitError as exc:
        logger.error('Failed merge')
        try:
            _git(parent_gwt.worktree, 'merge', '--abort', check=False)
        except Exception as _exc:
            logger.error('Failed merge --abort')
        return (False, str(exc))