"""Work in progress saved to one file and opened again later.

A work file (.edsfig) is a ZIP archive:

    edsfig.json     what was loaded and every setting (order, labels, adjustments, trimming, layout ...)
    files/<name>    the loaded files themselves, stored as they are (no compression, byte for byte)

A "settings only" work file has edsfig.json alone; it names the files by content (MD5), so the same
files loaded again next to it, from anywhere and in any order, get their settings back.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import PurePosixPath

EXT = ".edsfig"
MAGIC = "edsfig-work"
FORMAT = 1
STATE = "edsfig.json"
MAX_TOTAL = 16 * 1024 ** 3  # an archive claiming more than this is not one of ours


class ProjectError(ValueError):
    """The file cannot be opened as a work file (the message says why, for the screen)."""


def digest(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _plain(o):
    """JSON for numpy numbers and the like (settings come from widgets and tables)."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"cannot save {type(o).__name__}")


def pack(files: list[tuple[str, bytes]], state: dict, with_files: bool = True, app_version: str = "",
         sums: list[str] | None = None) -> bytes:
    """The work file: `files` = [(name, data)] in the order they were loaded; `state` = the settings.
    The files are stored without compression, so every byte is kept as it was.  `sums` = their MD5,
    when already known."""
    entries, used = [], set()
    for k, (name, data) in enumerate(files):
        base = PurePosixPath(str(name).replace("\\", "/")).name or f"file{k + 1}"
        path = f"files/{base}"
        if path in used:  # two files of the same name (from different folders)
            path = f"files/{k + 1:03d}_{base}"
        used.add(path)
        entries.append({"name": str(name), "path": path if with_files else None,
                        "md5": sums[k] if sums else digest(data), "size": len(data)})
    doc = {"format": MAGIC, "version": FORMAT, "app": app_version,
           "saved": datetime.now().strftime("%Y-%m-%d %H:%M"), "with_files": bool(with_files),
           "files": entries, "state": state}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr(STATE, json.dumps(doc, ensure_ascii=False, indent=1, default=_plain))
        if with_files:
            for e, (_, data) in zip(entries, files):
                z.writestr(e["path"], data)
    return buf.getvalue()


def unpack(data: bytes) -> tuple[dict, list[tuple[str, bytes, str]]]:
    """(the saved document, [(name, data, MD5)] of the files in it).  Every file is checked against
    the MD5 written when it was saved."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ProjectError("作業ファイルとして開けません（壊れているか、別の種類のファイルです）") from e
    with z:
        try:
            doc = json.loads(z.read(STATE).decode("utf-8"))
        except KeyError as e:
            raise ProjectError("作業ファイルではありません（設定 edsfig.json が入っていません）") from e
        except (ValueError, UnicodeDecodeError, zipfile.BadZipFile) as e:
            raise ProjectError("作業ファイルの設定が壊れています") from e
        if not isinstance(doc, dict) or doc.get("format") != MAGIC:
            raise ProjectError("作業ファイルではありません")
        try:
            version = int(doc.get("version", 0))
        except (TypeError, ValueError):
            version = 0
        if version > FORMAT:
            raise ProjectError("新しい版のアプリで保存された作業ファイルです。アプリを更新してから開いてください")
        entries = doc.get("files") or []
        if not isinstance(entries, list) or not isinstance(doc.get("state", {}), dict):
            raise ProjectError("作業ファイルの設定が壊れています")
        files, total = [], 0
        for e in entries:
            if not isinstance(e, dict) or not e.get("path"):
                continue  # settings only: the file itself was not saved
            name = str(e.get("name") or PurePosixPath(e["path"]).name)
            try:
                info = z.getinfo(e["path"])
            except KeyError as err:
                raise ProjectError(f"作業ファイルの中に {name} がありません（壊れています）") from err
            total += info.file_size
            if total > MAX_TOTAL:
                raise ProjectError("作業ファイルが大きすぎます")
            try:
                b = z.read(info)
            except (zipfile.BadZipFile, OSError) as err:
                raise ProjectError(f"作業ファイルの中の {name} が壊れています") from err
            h = digest(b)
            if e.get("md5") and h != e["md5"]:
                raise ProjectError(f"作業ファイルの中の {name} が壊れています")
            files.append((name, b, h))
    return doc, files


def missing(doc: dict, present: set[str]) -> list[str]:
    """Names of the files the work was made from that are not among `present` (MD5)."""
    return [str(e.get("name")) for e in doc.get("files") or []
            if isinstance(e, dict) and e.get("md5") not in present]
