"""Cross-platform clipboard utilities for error reporting."""

import os
import subprocess
import sys
from typing import Optional


def copy_to_clipboard(text: str) -> bool:
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except ImportError:
        pass

    if sys.platform == "darwin":
        return _copy_macos(text)
    elif sys.platform == "win32":
        return _copy_windows(text)
    elif sys.platform == "linux":
        return _copy_linux(text)

    return False


def _copy_macos(text: str) -> bool:
    try:
        process = subprocess.Popen(
            ["pbcopy"],
            stdin=subprocess.PIPE,
            close_fds=True,
        )
        process.communicate(input=text.encode("utf-8"))
        return process.returncode == 0
    except Exception:
        return False


def _copy_windows(text: str) -> bool:
    try:
        process = subprocess.Popen(
            ["clip"],
            stdin=subprocess.PIPE,
            close_fds=True,
        )
        process.communicate(input=text.encode("utf-8"))
        return process.returncode == 0
    except Exception:
        return False


def _copy_linux(text: str) -> bool:
    try:
        process = subprocess.Popen(
            ["wl-copy"],
            stdin=subprocess.PIPE,
            close_fds=True,
        )
        process.communicate(input=text.encode("utf-8"))
        if process.returncode == 0:
            return True
    except Exception:
        pass

    try:
        process = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            close_fds=True,
        )
        process.communicate(input=text.encode("utf-8"))
        return process.returncode == 0
    except Exception:
        pass

    return False


def reset_terminal_state() -> None:
    sys.stdout.write("\033[?25h")
    sys.stdout.write("\033[0m")
    sys.stdout.write("\033[!p")
    sys.stdout.flush()

    if sys.platform != "win32":
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            if os.isatty(fd):
                tty.setcbreak(fd)
                termios.tcsetattr(fd, termios.TCSANOW, termios.tcgetattr(fd))
        except Exception:
            pass


def format_error_for_clipboard(
    error: Exception,
    command: Optional[str] = None,
    include_traceback: bool = True,
) -> str:
    import traceback

    lines = [
        "=" * 60,
        "TurboQuant Error Report",
        "=" * 60,
        "",
    ]

    if command:
        lines.append(f"Command: {command}")
        lines.append("")

    lines.append(f"Error: {type(error).__name__}: {error}")
    lines.append("")

    if include_traceback:
        lines.append("Traceback:")
        lines.append("-" * 40)
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        lines.extend([line.rstrip() for line in tb_lines])
        lines.append("")

    lines.append("=" * 60)
    lines.append("System Information:")
    lines.append(f"  Python: {sys.version}")
    lines.append(f"  Platform: {sys.platform}")
    lines.append("=" * 60)

    return "\n".join(lines)


def copy_error_to_clipboard(
    error: Exception,
    command: Optional[str] = None,
    notify: bool = True,
) -> bool:
    formatted = format_error_for_clipboard(error, command)
    success = copy_to_clipboard(formatted)

    if notify:
        if success:
            print("\n📋 Error copied to clipboard!", file=sys.stderr)
        else:
            print(
                "\n⚠️  Could not copy error to clipboard automatically.",
                file=sys.stderr,
            )
            print(
                "   To manually copy, select the error text above.",
                file=sys.stderr,
            )

    return success
