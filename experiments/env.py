"""TextQuestsEnv loader.

Resolves the textquests root (submodule, env var, or explicit path) and
imports TextQuestsEnv so callers never have to touch sys.path themselves.

Typical usage
-------------
from env import load_env_class, find_game_path, resolve_root

root = resolve_root(cfg)                  # from config dict or TEXTQUESTS_ROOT env var
TextQuestsEnv = load_env_class(root)      # imports the class, adds root to sys.path
game_path = find_game_path('zork1', root) # returns absolute Path to game folder
env = TextQuestsEnv(str(game_path), seed=0)
"""

import importlib
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_ROOT = _REPO_ROOT / 'textquests'


def resolve_root(config=None):
    """Return the textquests root Path.

    Priority: TEXTQUESTS_ROOT env var > config['textquests_root'] > local submodule.
    Returns None if none of the above exist.
    """
    configured = os.getenv('TEXTQUESTS_ROOT') or (
        config.get('textquests_root') if isinstance(config, dict) else None
    )
    if configured:
        return Path(configured).expanduser().resolve()
    if _LOCAL_ROOT.exists():
        return _LOCAL_ROOT.resolve()
    return None


def load_env_class(root=None):
    """Import and return the TextQuestsEnv class.

    Adds the textquests root to sys.path so ``src.textquests_env`` is importable.
    Raises ImportError with a helpful message if the class cannot be found.
    """
    candidates = []
    if root is not None:
        candidates.append(Path(root).expanduser().resolve())
    if _LOCAL_ROOT.exists():
        local = _LOCAL_ROOT.resolve()
        if local not in candidates:
            candidates.append(local)

    errors = []
    for candidate in candidates:
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        try:
            module = importlib.import_module('src.textquests_env')
            return module.TextQuestsEnv
        except (ImportError, AttributeError) as exc:
            errors.append(f'{candidate}: {exc}')

    raise ImportError(
        'Could not import TextQuestsEnv.\n'
        'If you just cloned the repo, run:  git submodule update --init\n'
        'Or set TEXTQUESTS_ROOT to point at your textquests checkout.\n'
        + '\n'.join(errors)
    )


def find_game_path(game_name, root=None):
    """Return the absolute Path to a named game folder inside textquests/data/.

    Raises FileNotFoundError with a helpful message if the folder is not found.
    """
    roots = []
    if root is not None:
        roots.append(Path(root).expanduser().resolve())
    if _LOCAL_ROOT.exists():
        local = _LOCAL_ROOT.resolve()
        if local not in roots:
            roots.append(local)

    for r in roots:
        candidate = r / 'data' / 'jiminy-cricket' / game_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Game folder '{game_name}' not found under textquests/data/jiminy-cricket/.\n"
        'Run:  git submodule update --init\n'
        'or set TEXTQUESTS_ROOT to point at your textquests checkout.'
    )
