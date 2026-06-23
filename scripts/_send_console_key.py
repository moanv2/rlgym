"""Inject a single keystroke into another process's Windows console input buffer.

Used to drive rlgym_ppo's in-loop keyboard controls (p=pause, c=checkpoint,
q=quit, any key=resume) on a *detached* background training run, whose console
we can't type into directly. Works by AttachConsole(target_pid) then
WriteConsoleInput of a key-down/up pair.

    python scripts/_send_console_key.py <pid> <char> [down|tap]

mode (default "tap"):
  down  — inject ONLY a key-down. Use for rlgym_ppo's pause: 'p' down with no
          following key-up, so the "any key to resume" loop is NOT triggered.
  tap   — key-down + key-up (a normal keypress). Use to RESUME a paused run.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

KEY_EVENT = 0x0001
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3


class KEY_EVENT_RECORD(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", wintypes.BOOL),
        ("wRepeatCount", wintypes.WORD),
        ("wVirtualKeyCode", wintypes.WORD),
        ("wVirtualScanCode", wintypes.WORD),
        ("UnicodeChar", wintypes.WCHAR),
        ("dwControlKeyState", wintypes.DWORD),
    ]


class INPUT_RECORD(ctypes.Structure):
    _fields_ = [
        ("EventType", wintypes.WORD),
        ("KeyEvent", KEY_EVENT_RECORD),
    ]


MAPVK_VK_TO_VSC = 0


def _rec(char: str, down: bool) -> INPUT_RECORD:
    vk = ord(char.upper())
    scan = ctypes.windll.user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
    r = INPUT_RECORD()
    r.EventType = KEY_EVENT
    k = r.KeyEvent
    k.bKeyDown = down
    k.wRepeatCount = 1
    k.wVirtualKeyCode = vk
    k.wVirtualScanCode = scan
    k.UnicodeChar = char
    k.dwControlKeyState = 0
    return r


def main() -> int:
    pid = int(sys.argv[1])
    char = sys.argv[2][:1]
    mode = sys.argv[3] if len(sys.argv) > 3 else "tap"

    k32 = ctypes.windll.kernel32
    k32.FreeConsole()
    if not k32.AttachConsole(pid):
        err = ctypes.get_last_error()
        print(f"AttachConsole({pid}) failed (err={err}) — target has no attachable console", file=sys.stderr)
        return 2

    h = k32.CreateFileW(
        "CONIN$", GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None,
    )
    if h == wintypes.HANDLE(-1).value or h == 0xFFFFFFFFFFFFFFFF:
        print(f"CreateFileW(CONIN$) failed (err={ctypes.get_last_error()})", file=sys.stderr)
        return 3

    if mode == "down":
        records = (INPUT_RECORD * 1)(_rec(char, True))
        n = 1
    else:
        records = (INPUT_RECORD * 2)(_rec(char, True), _rec(char, False))
        n = 2
    written = wintypes.DWORD(0)
    ok = k32.WriteConsoleInputW(h, records, n, ctypes.byref(written))
    print(f"WriteConsoleInput ok={bool(ok)} written={written.value} char={char!r} mode={mode} -> pid {pid}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
