"""Repository loader: clone and load source files from the target codebase."""

import os
import re
import shutil
import subprocess
import sys

from git import Repo

REPOS_DIR = "cloned_repos"

# Broad set so any folder (gradle, config, docs, src) has analyzable files
SUPPORTED_EXTENSIONS = (
    ".java", ".kt", ".kts", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".rb",
    ".yaml", ".yml", ".properties",
    ".xml", ".json", ".md", ".adoc",
    ".http", ".env", ".toml", ".cfg", ".ini",
)

# Generic paths to exclude from any repo (build artifacts, deps, caches)
EXCLUDE_PATHS = (
    "build", "dist", "target", "out", ".next", ".nuxt",
    "node_modules", "__pycache__", ".venv", "venv", ".env",
    "gradle/wrapper", ".git",
)


def should_exclude(file_path: str, exclude_tests: bool = True) -> bool:
    """Return True if the file should be excluded from loading."""
    normalized = file_path.replace("\\", "/")
    if exclude_tests and ("/test/" in normalized or "/tests/" in normalized or "src/test" in normalized):
        return True
    for excluded in EXCLUDE_PATHS:
        if excluded in normalized or normalized.startswith(excluded):
            return True
    return False


def _repo_name_from_url(url: str) -> str:
    """Extract repo name from GitHub/GitLab URL for folder naming."""
    url = (url or "").strip()
    match = re.search(r"github\.com[/:]([^/]+)/([^/.]+)", url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    match = re.search(r"gitlab\.com[/:]([^/]+)/([^/.]+)", url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    match = re.search(r"bitbucket\.org[/:]([^/]+)/([^/.]+)", url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return "repo"


def _enable_git_longpaths_if_windows() -> None:
    """On Windows, enable Git long path support to avoid 'Filename too long' on checkout."""
    if sys.platform != "win32":
        return
    try:
        subprocess.run(
            ["git", "config", "--global", "core.longpaths", "true"],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass  # non-fatal; clone may still work for repos with shorter paths


def clone_repo(url: str | None = None, force: bool = False) -> str | None:
    """
    Clone the repository. Returns the local path, or None if url is missing.
    Always clones into REPOS_DIR / <repo_name_from_url>.
    On Windows, enables core.longpaths so repos with very long paths can be checked out.
    """
    target_url = (url or "").strip()
    if not target_url:
        return None
    _enable_git_longpaths_if_windows()
    os.makedirs(REPOS_DIR, exist_ok=True)
    folder = _repo_name_from_url(target_url)
    path = os.path.join(REPOS_DIR, folder)
    if os.path.exists(path):
        if force:
            shutil.rmtree(path)
        else:
            # Re-clone if existing clone is empty (e.g. interrupted previous clone)
            try:
                entries = [e for e in os.listdir(path) if e != ".git"]
                if not entries:
                    shutil.rmtree(path)
                else:
                    return path
            except (OSError, PermissionError):
                return path
    Repo.clone_from(target_url, path)
    return path


def get_folder_structure(repo_path: str) -> list[dict]:
    """
    Build a folder tree for UI display.
    Returns list of {label, path, children} for st.tree.
    """
    root_path = repo_path
    result = []

    def add_node(rel_path: str, name: str) -> dict:
        full = os.path.join(root_path, rel_path) if rel_path else root_path
        children = []
        try:
            entries = sorted(os.listdir(full))
        except (OSError, PermissionError):
            return {"label": name, "path": rel_path or ".", "children": []}

        for entry in entries:
            if entry.startswith(".") and entry != ".github":
                continue
            child_rel = os.path.join(rel_path, entry) if rel_path else entry
            child_full = os.path.join(root_path, child_rel)
            if os.path.isdir(child_full) and entry != ".git":
                children.append(add_node(child_rel, entry))
            elif os.path.isfile(child_full):
                pass  # Skip files in tree, only show folders

        return {"label": name, "path": rel_path or ".", "children": sorted(children, key=lambda x: (x["label"].lower(),))}

    result.append(add_node("", os.path.basename(repo_path)))
    return result


def get_selectable_folders(repo_path: str) -> list[str]:
    """Return flat list of folder paths for selection."""
    folders = ["."]
    for root, dirs, _ in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d != ".git" and (not d.startswith(".") or d == ".github")]
        rel = os.path.relpath(root, repo_path).replace("\\", "/")
        for d in dirs:
            path = f"{rel}/{d}" if rel != "." else d
            folders.append(path)
    return sorted(set(folders), key=lambda x: (x == ".", x))


def load_code_files(base_path: str, subfolder: str | None = None) -> list[dict]:
    """
    Load supported source files from the repository.
    base_path: repo root (required). subfolder: optional path within repo (e.g. 'src/main/java' or '.' for all).
    Returns list of dicts with keys: file (path), content.
    """
    if not base_path or not os.path.isdir(base_path):
        return []
    code_files = []
    subfolder = (subfolder or ".").strip().replace("\\", "/").rstrip("/") if subfolder else "."

    for root, _, files in os.walk(base_path):
        rel_root = os.path.relpath(root, base_path).replace("\\", "/")
        if subfolder != ".":
            if rel_root != subfolder and not rel_root.startswith(subfolder + "/"):
                continue

        for file in files:
            if not any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_path).replace("\\", "/")
            if should_exclude(rel_path, exclude_tests=True):
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (IOError, OSError):
                continue

            code_files.append({"file": rel_path, "content": content})

    return code_files
