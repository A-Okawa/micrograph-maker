"""The launcher of the shared mode (serve.py): the password as the app reads it, the port checks, the
address told to the students.

    python tests/test_serve.py
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve  # noqa: E402


def test_password_file_and_variable():
    keep_file, keep_env = serve.PASSWORD_FILE, os.environ.pop("EDSFIG_PASSWORD", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            serve.PASSWORD_FILE = Path(d) / "password.txt"
            assert serve.password() == ""  # no file: no password
            serve.PASSWORD_FILE.write_bytes("﻿パス123\r\n".encode("utf-8"))  # as Notepad saves it
            assert serve.password() == "パス123"
            serve.PASSWORD_FILE.write_text("  \n", encoding="utf-8")
            assert serve.password() == ""  # an empty file: no password
            os.environ["EDSFIG_PASSWORD"] = " lab "
            assert serve.password() == "lab"  # the variable comes first
    finally:
        serve.PASSWORD_FILE = keep_file
        os.environ.pop("EDSFIG_PASSWORD", None)
        if keep_env is not None:
            os.environ["EDSFIG_PASSWORD"] = keep_env


def test_ports():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        assert not serve.free(port)  # taken by another program
        assert not serve.answering(port)  # ... which is not this app
    finally:
        s.close()
    assert serve.free(port)


def test_address_and_check():
    ip = serve.lan_ip()
    assert isinstance(ip, str) and not ip.startswith("127.")
    assert serve.main(["--check"]) in (0, 1)


if __name__ == "__main__":
    test_password_file_and_variable()
    test_ports()
    test_address_and_check()
    print("ok")
