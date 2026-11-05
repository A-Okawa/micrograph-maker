# -*- coding: utf-8 -*-
"""Micrograph Maker の PC 版（Mac・Windows のアプリ）を起動する（figapp の PC 版と同じ仕組み）。

ダブルクリックで開くと、アプリの窓の中にいつもの画面が出る。画面を作る Streamlit のサーバーはこの PC の中
（127.0.0.1）だけで動き、ほかの PC からはつながらない。窓を閉じるとサーバーも止まる。

同じ実行ファイルが 3 通りに動く:
  MicrographMaker                   アプリの窓（pywebview）。窓を作れない PC（Windows で Edge WebView2 が無いなど）ではブラウザで開く
  MicrographMaker --browser         いつものブラウザで開く（タブを閉じてしばらくするとサーバーが止まる。画面の「アプリを終了」でも止まる）
  MicrographMaker --server PORT     Streamlit のサーバー（窓の側が起動する。ユーザーが直接使うことはない）

ログは Mac は ~/Library/Application Support/Micrograph Maker、Windows は %APPDATA%\\Micrograph Maker の logs
（MM_DATA_DIR で変更可）。読み取りの記憶はソース版と同じ場所（Mac: ~/.edsfig、Windows: %APPDATA%\\edsfig）。
開発中は `python desktop/launcher.py` でも動く（そのときはリポジトリの app.py を使う）。
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

APP = 'Micrograph Maker'                      # 窓・メニュー・メッセージに出る名前（appinfo.py の APP_NAME と同じ）
FROZEN = getattr(sys, 'frozen', False)
HERE = os.path.dirname(os.path.abspath(__file__))

TEXT = {
    'loading': '起動しています…',
    'first': '初めて開くときは 1 分ほどかかることがあります。',
    'failed': f'{APP} を起動できませんでした。もう一度開いてみてください。続くときは、次のフォルダの logs の中身を管理者に送ってください。',
    'running': f'{APP} はすでに起動しています。タスクバーの {APP} の窓を使ってください。',
    'all_files': 'すべてのファイル',
    'quit': f'{APP} を終了しますか？保存していない作業は消えます。',
}
# Mac のメニューと保存の窓（pywebview の表示）
MENU_JA = {
    'global.ok': 'OK', 'global.quit': '終了', 'global.cancel': 'キャンセル', 'global.saveFile': 'ファイルを保存',
    'cocoa.menu.about': f'{APP} について', 'cocoa.menu.services': 'サービス', 'cocoa.menu.view': '表示',
    'cocoa.menu.edit': '編集', 'cocoa.menu.hide': f'{APP} を隠す', 'cocoa.menu.hideOthers': 'ほかを隠す',
    'cocoa.menu.showAll': 'すべてを表示', 'cocoa.menu.quit': f'{APP} を終了', 'cocoa.menu.fullscreen': 'フルスクリーンにする',
    'cocoa.menu.cut': 'カット', 'cocoa.menu.copy': 'コピー', 'cocoa.menu.paste': 'ペースト', 'cocoa.menu.selectAll': 'すべてを選択',
    'windows.fileFilter.allFiles': 'すべてのファイル', 'windows.fileFilter.otherFiles': 'その他のファイル',
}


# ---------------------------------------------------------------------------------------------- places
def bundle_dir():
    """The folder with app.py: inside the app when frozen, else the repository."""
    return sys._MEIPASS if FROZEN else os.path.dirname(HERE)


def data_dir():
    d = os.environ.get('MM_DATA_DIR')
    if not d:
        if sys.platform == 'darwin':
            d = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', APP)
        elif os.name == 'nt':
            d = os.path.join(os.environ.get('APPDATA') or os.path.expanduser('~'), APP)
        else:
            d = os.path.join(os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share'),
                             'micrograph-maker')
    os.makedirs(os.path.join(d, 'logs'), exist_ok=True)
    return d


def open_log(name):
    """A fresh log for this run; the previous run's log is kept as <name>.1."""
    path = os.path.join(data_dir(), 'logs', name)
    try:
        if os.path.exists(path):
            os.replace(path, path + '.1')
    except OSError:
        pass
    return open(path, 'a', encoding='utf-8', errors='replace', buffering=1)


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


# ---------------------------------------------------------------------------------------------- server
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 127.0.0.1 にはプロキシを通さない


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def is_ready(port):
    try:
        with _NO_PROXY.open(f'http://127.0.0.1:{port}/_stcore/health', timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_ready(proc, port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log('server stopped with code', proc.returncode)
            return False
        if is_ready(port):
            log(f'server ready after {time.time() - t0:.1f} s')
            return True
        time.sleep(0.3)
    log('server did not answer in', timeout, 's')
    return False


def start_server(port, mode):
    cmd = [sys.executable] if FROZEN else [sys.executable, os.path.abspath(__file__)]
    cmd += ['--server', str(port), '--mode', mode, '--parent', str(os.getpid())]
    # PYINSTALLER_RESET_ENVIRONMENT: 自分自身を別のプロセスとして起動するときの PyInstaller の決まり
    env = dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT='1', MM_DATA_DIR=data_dir(), PYTHONDONTWRITEBYTECODE='1')
    kw = {'creationflags': 0x08000000} if os.name == 'nt' else {}        # CREATE_NO_WINDOW: 黒い窓を出さない
    log('starting the server:', cmd)
    return subprocess.Popen(cmd, env=env, cwd=bundle_dir(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **kw)


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(8)
        except subprocess.TimeoutExpired:
            proc.kill()
    log('server stopped')


def watch_parent(pid):
    """Stop this server when the window process is gone (also when it crashed or was killed)."""
    if not pid:
        return
    if os.name == 'nt':
        import ctypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        k32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        handle = k32.OpenProcess(0x00100000, False, pid)                      # SYNCHRONIZE
        if not handle:
            return

        def wait():
            k32.WaitForSingleObject(handle, 0xFFFFFFFF)
            log('the window process is gone: quitting')
            os._exit(0)
    else:
        def wait():
            while os.getppid() == pid:
                time.sleep(1.5)
            log('the window process is gone: quitting')
            os._exit(0)
    threading.Thread(target=wait, name='mm-parent', daemon=True).start()


def watch_tabs(grace=None, first=900):
    """Browser mode: stop when no tab has been open for `grace` s (or none opened within `first` s)."""
    grace = grace or int(os.environ.get('MM_TAB_GRACE', '120'))

    def sessions():
        try:
            from streamlit.runtime import Runtime
            if not Runtime.exists():
                return None
            return Runtime.instance()._session_mgr.num_active_sessions()
        except Exception:  # noqa: BLE001
            return None

    def loop():
        t0 = last = time.time()
        seen = False
        while True:
            time.sleep(5)
            n = sessions()
            if n:
                seen, last = True, time.time()
            elif n == 0 and ((seen and time.time() - last > grace) or (not seen and time.time() - t0 > first)):
                log('no browser tab for a while: quitting')
                os._exit(0)
    threading.Thread(target=loop, name='mm-tabs', daemon=True).start()


def run_server(argv):
    port = int(arg(argv, '--server'))
    mode = arg(argv, '--mode', 'window')
    out = open_log('server.log')
    sys.stdout = sys.stderr = out                    # 窓のアプリには標準出力が無いので、ログに書く（streamlit を読み込む前に）
    os.environ['EDSFIG_DESKTOP'] = mode              # app.py: window（アプリの窓）か browser（ブラウザで開いた）
    os.environ.pop('EDSFIG_SERVER', None)
    os.environ.setdefault('MM_DATA_DIR', data_dir())
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(data_dir(), 'matplotlib'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.dont_write_bytecode = True                   # アプリの中にファイルを書かない（Mac では署名が壊れる）
    watch_parent(int(arg(argv, '--parent', '0')))
    if mode == 'browser':
        watch_tabs()
    base = bundle_dir()
    os.chdir(base)
    flags = [f'--server.port={port}', '--server.address=127.0.0.1', '--server.headless=true',
             '--global.developmentMode=false', '--browser.gatherUsageStats=false', '--server.fileWatcherType=none',
             '--server.runOnSave=false', '--server.maxUploadSize=2000', '--client.toolbarMode=minimal']
    log('Micrograph Maker server', sys.version.split()[0], base, flags)
    from streamlit.web import cli
    cli.main(args=['run', os.path.join(base, 'app.py')] + flags, prog_name='streamlit', windows_expand_args=False)


# ---------------------------------------------------------------------------------------------- window
def load_webview():
    """pywebview, or None when this PC cannot show the app in its own window."""
    if sys.platform not in ('darwin', 'win32'):
        return None
    try:
        import webview
        if os.name == 'nt':
            from webview.platforms import winforms
            if not getattr(winforms, 'is_chromium', False):
                log('Microsoft Edge WebView2 is not installed: opening the browser instead')
                return None
        else:
            from webview.platforms import cocoa  # noqa: F401
        return webview
    except Exception as e:  # noqa: BLE001
        log('no app window:', repr(e))
        return None


def window_size(webview):
    try:
        s = webview.screens[0]
        return max(960, min(1600, int(s.width * 0.9))), max(640, min(1000, int(s.height * 0.88)))
    except Exception:  # noqa: BLE001
        return 1400, 900


def page(body):
    return ('<!doctype html><html><head><meta charset="utf-8"><style>'
            'body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#fff;color:#31333F;'
            'font-family:-apple-system,"Segoe UI","Hiragino Sans","Yu Gothic UI","Noto Sans",sans-serif}'
            '.box{max-width:560px;padding:24px;text-align:center;line-height:1.6}'
            '.spin{width:36px;height:36px;margin:0 auto 18px;border:4px solid #f3d4d4;border-top-color:#ff4b4b;'
            'border-radius:50%;animation:s 0.9s linear infinite}@keyframes s{to{transform:rotate(360deg)}}'
            'h1{font-size:20px;font-weight:600;margin:0 0 6px}p{margin:6px 0;color:#555;font-size:14px}'
            'code{font-size:12px;word-break:break-all;color:#31333F}</style></head>'
            f'<body><div class="box" lang="ja">{body}</div></body></html>')


def loading_page():
    return page(f'<div class="spin"></div><h1>{TEXT["loading"]}</h1><p>{TEXT["first"]}</p>')


def error_page():
    import html
    return page(f'<h1>{APP}</h1><p>{TEXT["failed"]}</p><p><code>{html.escape(data_dir())}</code></p>')


def message(text):
    """A message box without the app window (Windows); elsewhere only the log."""
    log(text)
    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, APP, 0x40)
        except Exception:  # noqa: BLE001
            pass


_MUTEX = None


def first_instance():
    """Windows starts a second copy on a second double-click; keep one (the Mac does this by itself)."""
    global _MUTEX
    if os.name != 'nt' or os.environ.get('MM_SELFTEST'):
        return True
    import ctypes
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    _MUTEX = k32.CreateMutexW(None, False, 'Local\\micrograph-maker-desktop')
    return ctypes.get_last_error() != 183                                     # ERROR_ALREADY_EXISTS


def tune_webview2():
    """Windows: let the page save several files one after another. Without this, Edge WebView2 asks
    "127.0.0.1 wants to download multiple files" from the second file on (the page is this app's own server)."""
    try:
        from webview.platforms import edgechromium
    except Exception as e:  # noqa: BLE001
        log('WebView2 not tuned:', repr(e))
        return
    original = edgechromium.EdgeChrome.on_webview_ready

    def allow(sender, args):
        try:
            if str(args.PermissionKind) == 'MultipleAutomaticDownloads':
                from Microsoft.Web.WebView2.Core import CoreWebView2PermissionState
                args.State = CoreWebView2PermissionState.Allow
        except Exception as e:  # noqa: BLE001
            log('permission:', repr(e))

    def on_webview_ready(self, sender, args):
        original(self, sender, args)
        if args.IsSuccess:
            sender.CoreWebView2.PermissionRequested += allow

    edgechromium.EdgeChrome.on_webview_ready = on_webview_ready


def keep_file_type_in_save_dialog():
    """The save dialog keeps the type of the file, so a name typed in the dialog still ends in .pdf, .tif, .edsfig …
    Windows: "PDF (*.pdf)" etc. as the file type (pywebview offers only "All files", and a typed name then has no
    extension). Mac: the extension is shown and kept."""
    if os.name == 'nt':
        try:
            from webview.platforms import edgechromium
        except Exception as e:  # noqa: BLE001
            log('save dialog not changed:', repr(e))
            return
        WinForms = edgechromium.WinForms

        def on_download_starting(self, sender, args):
            name = os.path.basename(str(args.ResultFilePath))
            ext = os.path.splitext(name)[1].lstrip('.').lower()
            dialog = WinForms.SaveFileDialog()
            folder = os.path.dirname(str(args.ResultFilePath))            # Edge の既定（ダウンロード）
            if folder:
                dialog.InitialDirectory = folder
            every = TEXT['all_files'] + ' (*.*)|*.*'
            if ext:
                dialog.Filter = f'{ext.upper()} (*.{ext})|*.{ext}|{every}'
                dialog.DefaultExt = ext
                dialog.AddExtension = True
            else:
                dialog.Filter = every
            dialog.FileName = name
            dialog.RestoreDirectory = True
            if dialog.ShowDialog(self.form) == WinForms.DialogResult.OK:
                args.ResultFilePath = dialog.FileName
            else:
                args.Cancel = True

        edgechromium.EdgeChrome.on_download_starting = on_download_starting
    elif sys.platform == 'darwin':
        try:
            import AppKit
            import Foundation
            from webview.platforms import cocoa
        except Exception as e:  # noqa: BLE001
            log('save dialog not changed:', repr(e))
            return
        try:
            from UniformTypeIdentifiers import UTType
        except Exception:  # noqa: BLE001
            UTType = None

        class MicrographMakerDownloadDelegate(cocoa.BrowserView.DownloadDelegate):
            def download_decideDestinationUsingResponse_suggestedFilename_completionHandler_(
                    self, download, response, suggested, handler):
                name = str(suggested)
                panel = AppKit.NSSavePanel.savePanel()
                downloads = Foundation.NSSearchPathForDirectoriesInDomains(
                    Foundation.NSDownloadsDirectory, Foundation.NSUserDomainMask, True)[0]
                panel.setDirectoryURL_(Foundation.NSURL.fileURLWithPath_(downloads))
                panel.setNameFieldStringValue_(name)
                ext = os.path.splitext(name)[1].lstrip('.')
                kind = UTType.typeWithFilenameExtension_(ext) if (ext and UTType is not None) else None
                if kind is not None:
                    panel.setAllowedContentTypes_([kind])
                panel.setExtensionHidden_(False)
                panel.setCanSelectHiddenExtension_(True)
                if panel.runModal() == 1:                                  # NSModalResponseOK
                    handler(panel.URL())
                else:
                    handler(None)

        cocoa.BrowserView.DownloadDelegate = MicrographMakerDownloadDelegate


def run_window(webview, proc, port, url):
    w, h = window_size(webview)
    loc = dict(MENU_JA)
    loc['global.quitConfirmation'] = TEXT['quit']
    window = webview.create_window(APP, html=loading_page(), width=w, height=h, min_size=(960, 640),
                                   confirm_close=not os.environ.get('MM_SELFTEST'), zoomable=True, text_select=True,
                                   background_color='#FFFFFF')

    def boot():
        if wait_ready(proc, port, 300):
            window.load_url(url)
            log('window shows', url)
            if os.environ.get('MM_SELFTEST'):
                import selftest
                selftest.run(window, url, log)
        else:
            window.load_html(error_page())

    webview.settings['ALLOW_DOWNLOADS'] = True             # ダウンロード（PDF などの保存）は保存先を聞く窓を出す
    keep_file_type_in_save_dialog()
    if os.name == 'nt':
        tune_webview2()
    if os.environ.get('MM_SELFTEST_SAVE'):
        import selftest
        selftest.save_without_dialog(os.environ['MM_SELFTEST_SAVE'])
    webview.start(boot, gui='edgechromium' if os.name == 'nt' else None, localization=loc, private_mode=True)
    log('window closed')


def run_browser(proc, port, url):
    if wait_ready(proc, port, 300):
        webbrowser.open(url)
        log('browser opened', url)
        proc.wait()                                         # 画面の「アプリを終了」か、タブを閉じてしばらくすると止まる
    else:
        message(TEXT['failed'] + '\n\n' + data_dir())


def run_app(argv):
    if FROZEN or not sys.stdout:
        sys.stdout = sys.stderr = open_log('launcher.log')
    log(f'{APP} desktop', sys.platform, sys.version.split()[0], 'frozen' if FROZEN else 'source')
    if not first_instance():
        message(TEXT['running'])
        return
    webview = None if ('--browser' in argv or os.environ.get('MM_BROWSER') == '1') else load_webview()
    mode = 'window' if webview else 'browser'
    port = free_port()
    proc = start_server(port, mode)
    url = f'http://127.0.0.1:{port}/'
    try:
        with open(os.path.join(data_dir(), 'running.json'), 'w', encoding='utf-8') as fh:
            json.dump({'pid': os.getpid(), 'server_pid': proc.pid, 'port': port, 'mode': mode, 'url': url}, fh)
    except OSError:
        pass
    try:
        if webview:
            run_window(webview, proc, port, url)
        else:
            run_browser(proc, port, url)
    finally:
        stop_server(proc)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--server' in argv:
        run_server(argv)
    else:
        run_app(argv)


if __name__ == '__main__':
    main()
