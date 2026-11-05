"""Work files (.edsfig): the loaded files kept byte for byte without compression, and the settings.

    python tests/test_project.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import project as P  # noqa: E402

FILES = [("EDS.docx", b"PK\x03\x04 word file" * 1000), ("a.tif", bytes(range(256)) * 400),
         ("sub/a.tif", b"another a.tif from another folder"), ("a.txt", "PixelSize=4.8\n".encode())]
STATE = {"sidebar": {"width_mm": 85.0, "arrange": "自由配置（番号で指定）"},
         "pictures": {"picrow": {"abc/word/media/image1.png": {"use": False, "label": "Si"}}},
         "figures": [{"key": "d123", "order": {"abc/word/media/image1.png": 2}, "spec": "1 (2/3)"}]}


def test_round_trip_keeps_every_byte():
    data = P.pack(FILES, STATE, app_version="0.7.0")
    doc, files = P.unpack(data)
    assert [(n, b) for n, b, _ in files] == FILES
    assert all(h == P.digest(b) for _, b, h in files)
    assert doc["state"] == STATE and doc["with_files"] and doc["app"] == "0.7.0"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = z.infolist()
        assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)  # no compression at all
        names = [i.filename for i in infos]
    assert names == ["edsfig.json", "files/EDS.docx", "files/a.tif", "files/003_a.tif", "files/a.txt"], names
    assert len(data) < sum(len(b) for _, b in FILES) + 4000  # the files plus a small settings file


def test_settings_only():
    data = P.pack(FILES, STATE, with_files=False)
    doc, files = P.unpack(data)
    assert files == [] and not doc["with_files"] and len(data) < 3000
    present = {P.digest(FILES[0][1]), P.digest(FILES[1][1])}
    assert P.missing(doc, present) == ["sub/a.tif", "a.txt"]
    assert P.missing(doc, {P.digest(b) for _, b in FILES}) == []


def _expect_error(data: bytes, word: str):
    try:
        P.unpack(data)
    except P.ProjectError as e:
        assert word in str(e), str(e)
    else:
        raise AssertionError("no error")


def test_damaged_or_foreign_files():
    _expect_error(b"not a zip at all", "開けません")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:  # an ordinary ZIP (say, the per-picture export)
        z.writestr("calibration.json", "{}")
    _expect_error(buf.getvalue(), "作業ファイルではありません")
    good = P.pack(FILES, STATE)
    doc = json.loads(zipfile.ZipFile(io.BytesIO(good)).read("edsfig.json"))
    doc["files"][1]["md5"] = "0" * 32  # a file that changed after saving
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(good)) as src, zipfile.ZipFile(buf, "w") as z:
        for i in src.infolist():
            z.writestr(i, json.dumps(doc) if i.filename == "edsfig.json" else src.read(i))
    _expect_error(buf.getvalue(), "a.tif が壊れています")
    doc = json.loads(zipfile.ZipFile(io.BytesIO(good)).read("edsfig.json"))
    doc["version"] = P.FORMAT + 1
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("edsfig.json", json.dumps(doc))
    _expect_error(buf.getvalue(), "新しい版")


def test_numpy_values_are_saved_as_numbers():
    import numpy as np
    doc, _ = P.unpack(P.pack([], {"n": np.int64(3), "x": np.float32(0.5), "b": np.bool_(True)}))
    assert doc["state"] == {"n": 3, "x": 0.5, "b": True}


if __name__ == "__main__":
    test_round_trip_keeps_every_byte()
    test_settings_only()
    test_damaged_or_foreign_files()
    test_numpy_values_are_saved_as_numbers()
    print("ok")
