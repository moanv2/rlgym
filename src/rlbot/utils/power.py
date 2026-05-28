"""Keep the machine awake during long training runs.

`caffeinate` only exists on macOS, so this is cross-platform:
  - Windows: Win32 ``SetThreadExecutionState`` (no external process)
  - macOS:   ``caffeinate``
  - Linux:   best-effort ``systemd-inhibit``

On an unsupported platform (or if the helper binary is missing) it logs a warning
and no-ops rather than failing training.

Usage:
    with keep_awake("rlbot exp_007_polish"):
        learner.learn()
"""
from __future__ import annotations

import contextlib
import platform
import subprocess
from collections.abc import Iterator

from rlbot.utils.logging import get_logger

_log = get_logger("rlbot.power")


@contextlib.contextmanager
def keep_awake(reason: str = "rlbot training", keep_display: bool = False) -> Iterator[None]:
    """Inhibit system sleep for the duration of the ``with`` block."""
    system = platform.system()
    if system == "Windows":
        yield from _keep_awake_windows(keep_display)
    elif system == "Darwin":
        yield from _keep_awake_macos(reason, keep_display)
    elif system == "Linux":
        yield from _keep_awake_linux(reason)
    else:
        _log.warning(f"keep_awake: unsupported platform {system!r} — the system may sleep.")
        yield


def _keep_awake_windows(keep_display: bool) -> Iterator[None]:
    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if keep_display:
        flags |= ES_DISPLAY_REQUIRED

    kernel32 = ctypes.windll.kernel32
    # Pin the signature so the 0x80000000 flag isn't mangled into a negative c_int.
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint

    kernel32.SetThreadExecutionState(flags)
    _log.info("keep_awake: Windows sleep inhibited (SetThreadExecutionState).")
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)  # clear the required flags
        _log.info("keep_awake: released sleep inhibition.")


def _keep_awake_macos(reason: str, keep_display: bool) -> Iterator[None]:
    args = ["caffeinate", "-i", "-s"]  # prevent idle + (on AC) system sleep
    if keep_display:
        args.append("-d")
    try:
        proc = subprocess.Popen(args)
    except FileNotFoundError:
        _log.warning("keep_awake: `caffeinate` not found — the system may sleep.")
        yield
        return
    _log.info(f"keep_awake: macOS caffeinate started (pid {proc.pid}).")
    try:
        yield
    finally:
        _stop(proc)
        _log.info("keep_awake: caffeinate stopped.")


def _keep_awake_linux(reason: str) -> Iterator[None]:
    # systemd-inhibit holds the lock until its child exits, so we give it a sleeper.
    args = [
        "systemd-inhibit", "--what=idle:sleep", "--who=rlbot",
        f"--why={reason}", "sleep", "infinity",
    ]
    try:
        proc = subprocess.Popen(args)
    except FileNotFoundError:
        _log.warning("keep_awake: `systemd-inhibit` not found — the system may sleep.")
        yield
        return
    _log.info(f"keep_awake: Linux systemd-inhibit started (pid {proc.pid}).")
    try:
        yield
    finally:
        _stop(proc)
        _log.info("keep_awake: systemd-inhibit stopped.")


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=5)
