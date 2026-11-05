"""Micrograph Maker を研究室の共有アプリとして起動します（同じ学内ネットワークの PC からブラウザで使えます）。

    python serve.py                 ポート 8510 で起動
    python serve.py --port 8600     ポートを変える

初回は学生用のパスワードを聞きます（password.txt に保存。空欄ならパスワードなし）。
入れたファイルはこの PC のメモリ上だけで処理され、保存も外部への送信もしません。
起動している間、この PC は自動ではスリープしません（このウィンドウを閉じると元に戻ります）。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from appinfo import APP_NAME  # noqa: E402

PASSWORD_FILE = HERE / "password.txt"
MIN_STREAMLIT = (1, 30)


def ready_to_run() -> bool:
    """This Python has what the app needs (the launchers ask before making a separate environment)."""
    try:
        from importlib.metadata import version

        nums = [int(x) for x in version("streamlit").split(".")[:2]]
    except Exception:
        return False
    if tuple(nums) < MIN_STREAMLIT:
        return False
    return all(importlib.util.find_spec(m) is not None for m in ("numpy", "PIL", "matplotlib"))


def lan_ip() -> str:
    """This PC's address on the network ("" when it cannot be told).  Connecting a UDP socket sends
    nothing: it only picks the interface the PC would use."""
    for probe in ("10.255.255.255", "172.31.255.255", "192.168.255.255", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 1))
                ip = s.getsockname()[0]
            if ip and not ip.startswith(("127.", "0.")):
                return ip
        except OSError:
            continue
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        return ""
    return "" if ip.startswith("127.") else ip


def password() -> str:
    """The students' password, read the way the app reads it: EDSFIG_PASSWORD, else password.txt."""
    pw = os.environ.get("EDSFIG_PASSWORD", "")
    if not pw:
        try:
            pw = PASSWORD_FILE.read_text(encoding="utf-8-sig")
        except OSError:
            pw = ""
    return pw.strip()


def ask_password() -> None:
    """The first time: a password for the students (kept in password.txt; empty = none)."""
    if PASSWORD_FILE.exists() or os.environ.get("EDSFIG_PASSWORD") or not sys.stdin or not sys.stdin.isatty():
        return
    print("学生が使うときのパスワードを決めてください（空欄のまま Enter ならパスワードなし）。")
    print("password.txt に保存します。変えるときは password.txt を書き換えて、起動し直します。")
    try:
        pw = input("パスワード: ").strip()
    except EOFError:
        return
    if pw:
        PASSWORD_FILE.write_text(pw + "\n", encoding="utf-8")
        try:
            os.chmod(PASSWORD_FILE, 0o600)
        except OSError:
            pass


def answering(port: int) -> bool:
    """A Streamlit app already answers on this port of this PC (asked directly, never through a proxy)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
            return r.read().strip() == b"ok"
    except Exception:
        return False


def free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name != "nt":  # as Streamlit's own server does (on Windows this would allow sharing a port)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def keep_awake() -> None:
    """While this window is open the PC does not go to sleep by itself (the screen may still turn off)."""
    try:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
        elif sys.platform == "darwin" and shutil.which("caffeinate"):
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def no_quick_edit() -> None:
    """Windows' console pauses a program that writes while text is being selected in its window: a click
    there would stop the app for everyone until Enter.  Selecting is switched off for this window."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)  # the console's input
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, (mode.value | 0x0080) & ~0x0040)  # EXTENDED_FLAGS on, QUICK_EDIT off
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8510, help="ポート番号（既定 8510）")
    ap.add_argument("--check", action="store_true", help="この Python で動かせるか調べるだけ（起動スクリプト用）")
    a = ap.parse_args(argv)
    if a.check:
        return 0 if ready_to_run() else 1
    if not ready_to_run():
        print(f"この Python（{sys.executable}）には必要なパッケージ（streamlit {'.'.join(map(str, MIN_STREAMLIT))} 以降など）が"
              "入っていません。start_server_mac.command（Mac）か start_server_windows.bat（Windows）から起動してください。")
        return 1
    ip = lan_ip()
    port = a.port
    if answering(port):
        print(f"すでに起動しています。学生に伝えるアドレス： http://{ip or 'localhost'}:{port}")
        return 0
    if not free(port):
        port = next((p for p in range(a.port + 1, a.port + 40) if free(p) and not answering(p)), 0)
        if not port:
            print(f"ポート {a.port} の近くに空きがありません。--port で別の番号を指定してください。")
            return 1
        print(f"ポート {a.port} はほかのソフトが使っているため、{port} で起動します。")
    ask_password()
    pw_from = ("" if not password() else "EDSFIG_PASSWORD" if os.environ.get("EDSFIG_PASSWORD", "").strip()
               else "password.txt")
    url = f"http://{ip}:{port}" if ip else f"http://（この PC の IP アドレス）:{port}"
    line = "=" * 64
    print(line)
    print(f" {APP_NAME} を研究室の共有アプリとして起動します")
    print(f" 学生に伝えるアドレス： {url}")
    print(f" パスワード：{f'あり（{pw_from}）' if pw_from else 'なし（同じネットワークの誰でも開けます）'}")
    print(f" この PC で開くとき： http://localhost:{port}")
    print(" 起動中はこの PC を自動でスリープさせません。終了するときはこのウィンドウを閉じます")
    print(line, flush=True)
    cmd = [sys.executable, "-m", "streamlit", "run", str(HERE / "app.py"),
           "--server.address", "0.0.0.0", "--server.port", str(port), "--server.headless", "true",
           "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false",
           "--client.toolbarMode", "viewer"]
    if ip:
        cmd += ["--browser.serverAddress", ip]  # the address Streamlit prints (and never asks the internet for one)
    keep_awake()
    no_quick_edit()
    proc = subprocess.Popen(cmd, cwd=HERE, env=dict(os.environ, EDSFIG_SERVER="1"))
    if hasattr(signal, "SIGTERM"):  # asked to stop: the app stops with this window
        signal.signal(signal.SIGTERM, lambda *_: proc.terminate())
    try:
        for _ in range(240):
            if proc.poll() is not None:
                break
            if answering(port):
                print(f"準備ができました。学生はブラウザで {url} を開きます。", flush=True)
                break
            time.sleep(0.5)
        return proc.wait()
    except KeyboardInterrupt:  # Ctrl+C reaches the app too: let it stop by itself
        try:
            proc.wait(10)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            proc.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
