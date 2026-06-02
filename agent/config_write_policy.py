"""Pre-write permission guard for the Hermes config (Phase 1 trust boundary).

The Hermes gateway and the agent run as the SAME OS user. A post-hoc janitor that
reverts drift after the fact cannot stop the agent from re-escalating between sweeps.
This module enforces the boundary at WRITE TIME: any write that would change a guarded
permission key on the authoritative config.yaml is refused unless either (a) a valid
root-owned marker authorizes that exact key, or (b) the write RESTORES the safe default
(the make-safer / snap-back direction).

The marker is a ROOT-OWNED file in a root-owned directory. Because the agent runs as a
non-root user with no passwordless sudo, it cannot create or modify the marker, so it
cannot grant itself permissions. A marker that the agent's own user could create is, by
definition, not a real grant and is rejected (the st_uid != 0 check).

Wired at the atomic_yaml_write / atomic_roundtrip_yaml_update sinks, so gateway hot-path
writers that bypass the named chokepoints are still caught.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The permission/autonomy keys the agent must never be able to widen on its own.
GUARDED_KEYS = (
    "approvals.destructive_slash_confirm",
    "approvals.mcp_reload_confirm",
    "command_allowlist",
    "kanban.dispatch_in_gateway",
    "delegation.subagent_auto_approve",
    "cron.max_parallel_jobs",
)

# The safe default for each guarded key. A marker-less write is permitted ONLY when it
# restores a guarded key to its safe default (the snap-back / make-safer direction); the
# agent can make a guarded key safer on its own but never less safe.
SAFE_DEFAULTS = {
    "approvals.destructive_slash_confirm": True,
    "approvals.mcp_reload_confirm": True,
    "command_allowlist": [],
    "kanban.dispatch_in_gateway": False,
    "delegation.subagent_auto_approve": False,
    "cron.max_parallel_jobs": 1,
}

# Root-owned grant marker. Override path ONLY via env, for tests.
MARKER_PATH = Path(os.environ.get("HERMES_TRUST_MARKER", "/etc/hermes/trust-grant.marker"))

_MISSING = object()


class PermissionGuardError(PermissionError):
    """Raised when a guarded permission key is written without a valid trust marker."""


def _get_nested(d: Any, dotted: str, default: Any = _MISSING) -> Any:
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _authoritative_config_path() -> Path | None:
    """Resolve the active profile's config.yaml. Lazy import avoids a circular import."""
    try:
        from hermes_cli.config import get_config_path
        return Path(get_config_path())
    except Exception:
        return None


def _marker_authorizes(key: str, marker_path: str | os.PathLike | None = None) -> bool:
    """A grant is valid only if the marker exists, is ROOT-OWNED, names this exact key,
    carries Drew's token, and has not expired. Anything else (including a marker the
    agent's own user created) is rejected."""
    mp = Path(marker_path) if marker_path else MARKER_PATH
    try:
        st = mp.stat()
    except Exception:
        return False
    # Must be owned by root. A same-user-creatable marker is not a grant.
    if getattr(st, "st_uid", -1) != 0:
        return False
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    exp = data.get("expires_at")
    try:
        if exp is None or time.time() > float(exp):
            return False
    except (TypeError, ValueError):
        return False
    if key not in (data.get("granted_keys") or []):
        return False
    if not data.get("drew_token"):
        return False
    return True


def _log_event(event: str, key: str, old: Any, new: Any) -> None:
    rec = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        "key": key,
        "on_disk": old if old is not _MISSING else None,
        "attempted": new,
    }
    try:
        home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
        logpath = home / "logs" / "autonomy-config-guard.log"
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with open(logpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _log_refused(key: str, old: Any, new: Any) -> None:
    _log_event("guarded_write_refused", key, old, new)
    logger.warning("Refused guarded config write: %s (on-disk=%r attempted=%r)", key, old, new)


def assert_write_allowed(
    target_path: str | os.PathLike,
    new_config: dict,
    *,
    authoritative_path: str | os.PathLike | None = None,
    marker_path: str | os.PathLike | None = None,
) -> None:
    """Refuse the write if it changes a guarded key on the authoritative config without
    either a valid marker for that key or restoring its safe default. ABSENT-in-new means
    no change, so partial-dict and gateway hot-path writers are never false-refused.

    Raises PermissionGuardError on refusal. Returns None when allowed.
    """
    auth = Path(authoritative_path) if authoritative_path else _authoritative_config_path()
    if auth is None:
        return
    tp = Path(target_path)
    try:
        same = tp.resolve() == auth.resolve()
    except Exception:
        same = str(tp) == str(auth)
    if not same:
        return  # only the authoritative config.yaml is guarded
    on_disk = _load_yaml(tp)
    for gk in GUARDED_KEYS:
        new_val = _get_nested(new_config, gk, _MISSING)
        if new_val is _MISSING:
            continue  # absent = no change
        old_val = _get_nested(on_disk, gk, _MISSING)
        if new_val != old_val:
            # Allow a marker-less write that RESTORES the safe default (snap-back /
            # make-safer direction). Only making a key LESS safe needs a marker.
            if gk in SAFE_DEFAULTS and new_val == SAFE_DEFAULTS[gk]:
                continue
            if not _marker_authorizes(gk, marker_path):
                _log_refused(gk, old_val, new_val)
                raise PermissionGuardError(
                    f"Refused write to guarded permission key '{gk}' (no valid root-owned "
                    f"trust marker). on-disk={old_val!r} attempted={new_val!r}"
                )


def snap_back_guarded_keys(*, fail_closed: bool = False, marker_path: str | os.PathLike | None = None) -> list:
    """Boot-time fail-safe. For each guarded key that has drifted off its safe default
    and is NOT authorized by a valid marker, restore the safe default and log loudly.

    fail_closed=True instead raises PermissionGuardError so the gateway refuses to start
    on unauthorized drift (Drew chose fail-SAFE, the default). Returns the list of
    (key, old, safe) it restored.
    """
    auth = _authoritative_config_path()
    if auth is None:
        return []
    auth = Path(auth)
    if not auth.exists():
        return []
    config = _load_yaml(auth)
    drifted = []
    for gk, safe in SAFE_DEFAULTS.items():
        cur = _get_nested(config, gk, _MISSING)
        if cur is _MISSING:
            continue
        if cur != safe and not _marker_authorizes(gk, marker_path):
            drifted.append((gk, cur, safe))
    if not drifted:
        return []
    if fail_closed:
        detail = "; ".join(f"{k}={c!r} safe={s!r}" for k, c, s in drifted)
        raise PermissionGuardError(
            f"Boot refused: guarded permission keys drifted with no valid marker: {detail}"
        )
    for gk, cur, safe in drifted:
        _set_nested(config, gk, safe)
        logger.warning("Boot snap-back: restored %s from %r to safe default %r", gk, cur, safe)
        _log_event("guarded_key_snapped_back", gk, cur, safe)
    try:
        from utils import atomic_yaml_write
        atomic_yaml_write(auth, config, sort_keys=False)
    except Exception as e:
        logger.error("Boot snap-back failed to write corrected config: %s", e)
    return drifted
