# -*- coding: utf-8 -*-
"""Checks of the app window on the build machines (GitHub Actions). Used only when MM_SELFTEST is set.

In the real window (WKWebView on the Mac, Edge WebView2 on Windows) this waits for the app, then uploads the sample
files listed in MM_SELFTEST_FILES (made by desktop/ci/samples.py: no real data) through the page's file input, waits
for the figure, clicks the PDF, PNG, TIFF and SVG buttons and 作業を保存, and records what happened in
<MM_SELFTEST>/selftest.json. The build machine takes screenshots in between (marker files *.ready).

MM_SELFTEST_SAVE=<folder>: downloads go straight into that folder (no save dialog), so the saved files can be checked.
Without it the normal save dialog opens; the build machine photographs it, this presses its Save button, checks that the
PDF arrived in the Downloads folder, and the build machine then ends the app.
"""
import base64
import json
import mimetypes
import os
import sys
import threading
import time

RENAMED = 'selftest-renamed'                               # 保存の窓で打つ名前（拡張子なし）

UPLOAD_JS = """(function (files) {
  var inputs = document.querySelectorAll('[data-testid="stSidebar"] input[type="file"]');
  if (!inputs.length) return 'no file input';
  var dt = new DataTransfer();
  files.forEach(function (f) {
    var bin = atob(f.data), bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    dt.items.add(new File([bytes], f.name, {type: f.type}));
  });
  inputs[0].files = dt.files;
  inputs[0].dispatchEvent(new Event('input', {bubbles: true}));
  inputs[0].dispatchEvent(new Event('change', {bubbles: true}));
  return 'ok';
})(%s)"""

CLICK_JS = """(function (label) {
  var b = Array.prototype.slice.call(document.querySelectorAll('[data-testid="stDownloadButton"] button'))
    .filter(function (e) { var t = e.innerText.trim(); return t === label || t.indexOf(label + '（') === 0; })[0];
  if (!b) return false;
  b.scrollIntoView({block: 'center'});
  b.click();
  return true;
})(%s)"""


OPEN_SAVE_JS = """(function () {
  var s = Array.prototype.slice.call(document.querySelectorAll('[data-testid="stSidebar"] [data-testid="stExpander"] summary'))
    .filter(function (e) { return e.innerText.indexOf('作業の保存') >= 0; })[0];
  if (!s) return false;
  if (!s.parentElement.open) s.click();
  return true;
})()"""


def _poll(fn, timeout, step=0.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            v = fn()
        except Exception:  # noqa: BLE001
            v = None
        if v:
            return v
        time.sleep(step)
    return None


def _saved(folder, suffix):
    """Size and first bytes of the downloaded file ending in `suffix`, once it stops growing."""
    names = [n for n in os.listdir(folder) if n.endswith(suffix)] if os.path.isdir(folder) else []
    if not names:
        return None
    path = os.path.join(folder, names[0])
    size = os.path.getsize(path)
    time.sleep(0.6)
    if size == 0 or size != os.path.getsize(path):
        return None
    with open(path, 'rb') as fh:
        return {'name': names[0], 'bytes': size, 'head': fh.read(16).decode('latin-1')}


def run(window, url, log):
    out = os.environ['MM_SELFTEST']
    os.makedirs(out, exist_ok=True)
    save = os.environ.get('MM_SELFTEST_SAVE')
    res = {'url': url, 'platform': sys.platform, 'save_dir': save, 'errors': []}

    def js(code):
        try:
            return window.evaluate_js(code)
        except Exception as e:  # noqa: BLE001
            res['errors'].append(repr(e)[:300])
            return None

    def mark(name):
        with open(os.path.join(out, name + '.ready'), 'w') as fh:
            fh.write(str(time.time()))

    def dump():
        with open(os.path.join(out, 'selftest.json'), 'w', encoding='utf-8') as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)

    res['sidebar'] = bool(_poll(lambda: js('!!document.querySelector(\'[data-testid="stSidebar"]\')'), 180))
    time.sleep(2)
    res['user_agent'] = js('navigator.userAgent')
    res['title'] = js('document.title')
    res['exceptions'] = js('document.querySelectorAll(\'[data-testid="stException"]\').length')
    res['sidebar_text'] = (js('document.querySelector(\'[data-testid="stSidebar"]\').innerText') or '')[:400]
    dump()
    mark('loaded')
    log('selftest: loaded', res['sidebar'])
    time.sleep(4)                                          # 写真を撮る間

    files = []
    for path in [p for p in os.environ.get('MM_SELFTEST_FILES', '').split(os.pathsep) if p]:
        with open(path, 'rb') as fh:
            files.append({'name': os.path.basename(path), 'data': base64.b64encode(fh.read()).decode('ascii'),
                          'type': mimetypes.guess_type(path)[0] or 'application/octet-stream'})
    res['upload'] = js(UPLOAD_JS % json.dumps(files))
    names = [f['name'] for f in files]
    res['uploaded'] = bool(_poll(lambda: js('(function (n) { var t = document.body.innerText; return n.every('
                                            'function (x) { return t.indexOf(x) >= 0; }); })(%s)' % json.dumps(names)), 90))
    res['figure'] = bool(_poll(lambda: js("document.body.innerText.indexOf('4. プレビュー') >= 0 && "
                                          "document.querySelectorAll('[data-testid=\"stImage\"] img').length > 0"), 300))
    time.sleep(3)
    res['captions'] = (js('Array.prototype.map.call(document.querySelectorAll(\'[data-testid="stCaptionContainer"]\'),'
                          ' function (e) { return e.innerText; }).join(" | ")') or '')[:2000]
    res['exceptions_after_upload'] = js('document.querySelectorAll(\'[data-testid="stException"]\').length')
    dump()
    mark('figure')
    log('selftest: figure', res['uploaded'], res['figure'])
    time.sleep(4)                                          # 写真を撮る間
    if not save:                                           # 本物の保存の窓：写真を撮ってもらい、「保存」を押す
        # Windows は拡張子なしの名前を打って .pdf になるのを確かめる。Mac の保存の窓は別のプロセスで動いていて、ここからは
        # 名前を打てない（.pdf が付いたまま表示されること・PDF の種類に限られていることを確かめ、そのまま保存する）
        folder = downloads_folder()
        res['download_folder'] = folder
        for old in os.listdir(folder) if os.path.isdir(folder) else []:
            if old.endswith('.pdf') and (old.startswith(RENAMED) or old.endswith('_figure.pdf')):
                os.remove(os.path.join(folder, old))
        res['clicked_pdf'] = js(CLICK_JS % "'PDF'")
        dump()
        mark('dialog')
        log('selftest: save dialog requested')
        time.sleep(9)
        res['dialog'] = press_save(RENAMED)
        res['saved_by_dialog'] = _poll(lambda: _saved(folder, '.pdf'), 40)
        res['files_in_downloads'] = sorted(os.listdir(folder))[:30] if os.path.isdir(folder) else []
        dump()
        mark('dialog_done')
        log('selftest: dialog', res['dialog'], res['saved_by_dialog'])
        return
    res['downloads'] = {}
    for label, suffix in (('PDF', '_figure.pdf'), ('PNG', '_figure.png'), ('TIFF', '_figure.tif'), ('SVG', '_figure.svg'),
                          ('作業を保存', '.edsfig')):
        if label == '作業を保存':                          # the work file's button is in the sidebar's 「作業の保存」 section
            js(OPEN_SAVE_JS)
            time.sleep(2)
        clicked = js(CLICK_JS % json.dumps(label))
        res['downloads'][label] = _poll(lambda: _saved(save, suffix), 120) if clicked else {'clicked': False}
        dump()
    res['files_in_save_dir'] = sorted(os.listdir(save))
    res['exceptions_end'] = js('document.querySelectorAll(\'[data-testid="stException"]\').length')
    dump()
    mark('done')
    log('selftest: done', res['downloads'])
    time.sleep(4)
    if os.environ.get('MM_SELFTEST_CLOSE') == '1':
        window.destroy()


def downloads_folder():
    """The folder the save dialog starts in (the user's Downloads)."""
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders') as key:
                return winreg.QueryValueEx(key, '{374DE290-123F-4565-9164-39C4925E467B}')[0]
        except OSError:
            pass
    return os.path.join(os.path.expanduser('~'), 'Downloads')


def press_save(new_name):
    """Press Save in the open save dialog of this app, as a user would; on Windows first type `new_name` (without an
    extension) as the file name."""
    out = {'pressed': False}
    if sys.platform == 'darwin':
        import AppKit
        from PyObjCTools import AppHelper
        done = threading.Event()

        def press():
            try:
                w = AppKit.NSApp.modalWindow() or AppKit.NSApp.keyWindow()
                out['window'] = str(w.className()) if w is not None else None
                if w is not None and w.isKindOfClass_(AppKit.NSSavePanel):
                    out['name'] = str(w.nameFieldStringValue())
                    out['extension_hidden'] = bool(w.isExtensionHidden())
                    out['types'] = [str(t.identifier()) for t in (w.allowedContentTypes() or [])]
                    out['url'] = str(w.URL().path()) if w.URL() is not None else None
                    # the panel runs in another process (its Save button cannot be pressed from here): end the
                    # modal session with OK, as the Save button does
                    AppKit.NSApp.stopModalWithCode_(1)
                    out['pressed'] = True
            except Exception as e:  # noqa: BLE001
                out['error'] = repr(e)
            done.set()

        AppHelper.callAfter(press)
        done.wait(20)
    elif os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        found = []
        proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def each(hwnd, _):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == os.getpid() and user32.IsWindowVisible(hwnd):
                cls = ctypes.create_unicode_buffer(64)
                user32.GetClassNameW(hwnd, cls, 64)
                if cls.value == '#32770':                  # the save dialog
                    found.append(hwnd)
            return True

        user32.EnumWindows(proc(each), 0)
        if found:
            title = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(found[0], title, 256)
            out['window'] = title.value
            boxes = []

            def child(hwnd, _):                            # the file name box: the Edit inside a ComboBox
                cls = ctypes.create_unicode_buffer(64)
                user32.GetClassNameW(hwnd, cls, 64)
                if cls.value == 'Edit':
                    up = user32.GetParent(hwnd)
                    c1 = ctypes.create_unicode_buffer(64)
                    if up:
                        user32.GetClassNameW(up, c1, 64)
                    out.setdefault('edits', []).append(c1.value)
                    if c1.value == 'ComboBox':
                        boxes.append(hwnd)
                return True

            user32.EnumChildWindows(found[0], proc(child), 0)
            if boxes:
                text = ctypes.create_unicode_buffer(260)
                user32.GetWindowTextW(boxes[0], text, 260)
                out['name'] = text.value
                user32.SendMessageW(boxes[0], 0x00B1, 0, -1)                   # EM_SETSEL: select the name
                for ch in new_name:                                            # and type over it
                    user32.SendMessageW(boxes[0], 0x0102, ord(ch), 0)          # WM_CHAR
                user32.GetWindowTextW(boxes[0], text, 260)
                out['typed'] = text.value
            time.sleep(0.5)
            user32.PostMessageW(found[0], 0x0111, 1, 0)    # WM_COMMAND IDOK: the Save button
            out['pressed'] = True
    return out


def save_without_dialog(folder):
    """Downloads of the window go into `folder` without the save dialog (only for the checks)."""
    os.makedirs(folder, exist_ok=True)
    if sys.platform == 'darwin':
        import Foundation
        from webview.platforms import cocoa

        class MicrographMakerSelftestDownloadDelegate(cocoa.BrowserView.DownloadDelegate):
            def download_decideDestinationUsingResponse_suggestedFilename_completionHandler_(
                    self, download, response, suggested, handler):
                handler(Foundation.NSURL.fileURLWithPath_(os.path.join(folder, str(suggested))))

        cocoa.BrowserView.DownloadDelegate = MicrographMakerSelftestDownloadDelegate
    elif os.name == 'nt':
        from webview.platforms import edgechromium

        def on_download_starting(self, sender, args):
            args.ResultFilePath = os.path.join(folder, os.path.basename(str(args.ResultFilePath)))
            args.Handled = True

        edgechromium.EdgeChrome.on_download_starting = on_download_starting
