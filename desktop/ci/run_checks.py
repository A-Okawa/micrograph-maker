# -*- coding: utf-8 -*-
"""Checks of the built PC app on the build machines (GitHub Actions, Mac and Windows). Everything goes to $REPORT.

  python run_checks.py server              the app's server in Chromium (smoke_server.py), in the window and browser modes
  python run_checks.py window              the app window: page, upload of the sample report, downloads (saved without the dialog)
  python run_checks.py dialog              the app window: the save dialog opens (photo); killing the app stops the server
  python run_checks.py installer SETUP     Windows: install without questions, start, uninstall
  python run_checks.py minos APP           Mac: the oldest macOS that every program in the app supports
Exit code 1 when a check fails.  The sample files come from samples.py (made from the tests' generators, no real data).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.abspath(os.environ.get('REPORT') or 'report')
TMP = os.environ.get('RUNNER_TEMP') or os.path.abspath('tmp')
MAC = sys.platform == 'darwin'
EXE = os.path.abspath('dist/Micrograph Maker.app/Contents/MacOS/MicrographMaker' if MAC else 'dist/MicrographMaker/MicrographMaker.exe')
os.makedirs(REPORT, exist_ok=True)
fails, lines = [], []
MAGIC = {'PDF': ('%PDF',), 'PNG': ('\x89PNG',), 'TIFF': ('II*\x00', 'MM\x00*'), 'SVG': ('<?xml', '<svg'), '作業を保存': ('PK\x03\x04',)}


def ok(cond, what):
    line = ('OK   ' if cond else 'FAIL ') + what
    print(line, flush=True)
    lines.append(line)
    if not cond:
        fails.append(what)


def finish(name):
    with open(os.path.join(REPORT, f'checks-{name}.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n\nFAILED: ' + (', '.join(fails) if fails else 'none') + '\n')
    print('\nFAILED:', fails if fails else 'none')
    sys.exit(1 if fails else 0)


def health(port):
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                f'http://127.0.0.1:{port}/_stcore/health', timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def alive(pid):
    if os.name == 'nt':
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        k32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        k32.CloseHandle.argtypes = (ctypes.c_void_p,)
        h = k32.OpenProcess(0x1000, False, pid)                    # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259                                   # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for(fn, timeout, step=0.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return time.time() - t0
        time.sleep(step)
    return None


def photo(name):
    try:
        from PIL import ImageGrab
        im = ImageGrab.grab()
        im.save(os.path.join(REPORT, name))
        print('photo', name, im.size, flush=True)
    except Exception as e:  # noqa: BLE001
        print('photo failed', name, repr(e), flush=True)


def copy_logs(data, tag):
    d = os.path.join(data, 'logs')
    if os.path.isdir(d):
        for f in os.listdir(d):
            shutil.copy(os.path.join(d, f), os.path.join(REPORT, f'{tag}-{f}'))


def samples(tag):
    sys.path.insert(0, HERE)
    import samples as S
    return S.write(os.path.join(TMP, 'samples-' + tag))


def bundle_files():
    """Every file inside the app (to see that the app writes nothing into itself)."""
    top = os.path.dirname(os.path.dirname(EXE)) if MAC else os.path.dirname(EXE)
    return {os.path.join(r, f) for r, _, fs in os.walk(top) for f in fs}


# ---------------------------------------------------------------------------------------------- checks
def server():
    before = bundle_files()
    files = samples('server')
    for port, mode in ((8765, 'window'), (8766, 'browser')):
        data = os.path.join(TMP, 'data-server-' + mode)
        home = os.path.join(TMP, 'home-server-' + mode)
        os.makedirs(data, exist_ok=True)
        p = subprocess.Popen([EXE, '--server', str(port), '--mode', mode, '--parent', '0'],
                             env=dict(os.environ, MM_DATA_DIR=data, EDSFIG_HOME=home))
        t = wait_for(lambda: health(port), 300)
        ok(t is not None, f'{mode}: server answers ({t and round(t, 1)} s after start)')
        if t is not None:
            r = subprocess.run([sys.executable, os.path.join(HERE, 'smoke_server.py'), f'http://127.0.0.1:{port}',
                                os.path.join(REPORT, 'server'), os.pathsep.join(files), mode, home], timeout=1800)
            ok(r.returncode == 0, f'{mode}: browser checks of the server (see server/smoke_server_{mode}.txt)')
        if mode == 'browser':
            t = wait_for(lambda: p.poll() is not None, 30)
            ok(t is not None, f'browser: 「アプリを終了」 ended the server process (code {p.poll()})')
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
        copy_logs(data, 'server-' + mode)
    added = sorted(bundle_files() - before)
    ok(not added, f'nothing written into the app ({added[:3]})')
    finish('server')


def window(save):
    tag = 'window' if save else 'dialog'
    st = os.path.join(REPORT, tag)
    data = os.path.join(TMP, 'data-' + tag)
    home = os.path.join(TMP, 'home-' + tag)
    shutil.rmtree(st, ignore_errors=True)
    os.makedirs(st)
    report_file = samples(tag)[0]                                  # the Word report
    env = dict(os.environ, MM_DATA_DIR=data, EDSFIG_HOME=home, MM_SELFTEST=st, MM_SELFTEST_CLOSE='1',
               MM_SELFTEST_FILES=report_file)
    if save:
        env['MM_SELFTEST_SAVE'] = os.path.join(st, 'saved')
    t0 = time.time()
    p = subprocess.Popen([EXE], env=env)
    t = wait_for(lambda: os.path.exists(os.path.join(st, 'loaded.ready')), 420)
    ok(t is not None, f'{tag}: page shown in the app window ({t and round(t, 1)} s after start)')
    time.sleep(1.5)
    photo(f'{tag}-1-loaded.png')
    running = {}
    try:
        running = json.load(open(os.path.join(data, 'running.json'), encoding='utf-8'))
    except (OSError, ValueError):
        pass
    ok(running.get('mode') == 'window', f'{tag}: runs in its own window (mode {running.get("mode")})')
    spid = running.get('server_pid')
    t = wait_for(lambda: os.path.exists(os.path.join(st, 'figure.ready')), 420)
    ok(t is not None, f'{tag}: the sample report uploaded and drawn')
    time.sleep(1.5)
    photo(f'{tag}-2-figure.png')
    if save:
        t = wait_for(lambda: os.path.exists(os.path.join(st, 'done.ready')), 600)
        ok(t is not None, f'{tag}: downloads finished')
        time.sleep(1)
        photo(f'{tag}-3-done.png')
        t = wait_for(lambda: p.poll() is not None, 90)
        ok(t is not None, f'{tag}: the app ends when its window closes (code {p.poll()})')
    else:
        t = wait_for(lambda: os.path.exists(os.path.join(st, 'dialog.ready')), 420)
        ok(t is not None, f'{tag}: PDF button pressed')
        time.sleep(5)
        photo(f'{tag}-3-save-dialog.png')
        t = wait_for(lambda: os.path.exists(os.path.join(st, 'dialog_done.ready')), 120)
        ok(t is not None, f'{tag}: Save pressed in the dialog')
        time.sleep(1)
        photo(f'{tag}-4-after-save.png')
        p.kill()                                                    # like a crash: the server must stop by itself
        p.wait(30)
    if spid:
        t = wait_for(lambda: not alive(spid), 30)
        ok(t is not None, f'{tag}: the server stopped too ({t and round(t, 1)} s)')
    if p.poll() is None:
        p.kill()
    res = {}
    try:
        res = json.load(open(os.path.join(st, 'selftest.json'), encoding='utf-8'))
    except (OSError, ValueError):
        pass
    ok(res.get('sidebar') and res.get('title') == 'Micrograph Maker', f'{tag}: the app is there ({res.get("title")})')
    ok(res.get('exceptions') == 0, f'{tag}: no error on the page ({res.get("exceptions")})')
    ok(res.get('upload') == 'ok' and res.get('uploaded'), f'{tag}: the report uploaded in the window ({res.get("upload")})')
    ok(res.get('figure') and '1画素 = 65.79 nm' in (res.get('captions') or ''),
       f'{tag}: figure drawn, the scale read (65.79 nm per pixel)')
    ok(res.get('exceptions_after_upload') == 0, f'{tag}: no error after the upload')
    if save:
        dl = res.get('downloads') or {}
        for label, heads in MAGIC.items():
            got = dl.get(label) or {}
            ok(str(got.get('head', '')).startswith(heads) and got.get('bytes', 0) > 100,
               f'{tag}: {label} saved from the window ({got.get("name")}, {got.get("bytes")} bytes)')
        ok(res.get('exceptions_end') == 0, f'{tag}: no error after the downloads')
        ok(os.path.exists(os.path.join(home, 'exemplars.json')), f'{tag}: the confirmed readings learned (outside the app)')
    else:
        ok(res.get('clicked_pdf') is True, f'{tag}: PDF button found')
        ok((res.get('dialog') or {}).get('pressed'), f'{tag}: the save dialog was there ({res.get("dialog")})')
        got = res.get('saved_by_dialog') or {}
        ok(str(got.get('head', '')).startswith('%PDF') and got.get('bytes', 0) > 100,
           f'{tag}: saved as {got.get("name")} through the dialog ({got.get("bytes")} bytes)')
        d = res.get('dialog') or {}
        if MAC:
            ok(str(d.get('name', '')).endswith('_figure.pdf') and not d.get('extension_hidden') and d.get('types') == ['com.adobe.pdf'],
               f'{tag}: the dialog shows the PDF name and keeps the PDF type ({d.get("name")}, {d.get("types")})')
        else:
            ok(d.get('typed') == 'selftest-renamed' and got.get('name') == 'selftest-renamed.pdf',
               f'{tag}: a name without .pdf was typed ({d.get("typed")}) and .pdf was added ({got.get("name")})')
    print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])
    print('took', round(time.time() - t0, 1), 's')
    copy_logs(data, tag)
    finish(tag)


def installer(setup):
    log = os.path.join(REPORT, 'installer-setup.log')
    r = subprocess.run([setup, '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', f'/LOG={log}'], timeout=900)
    ok(r.returncode == 0, f'installer ran without questions (code {r.returncode})')
    home = os.path.join(os.environ['LOCALAPPDATA'], 'Programs', 'Micrograph Maker')
    exe = os.path.join(home, 'MicrographMaker.exe')
    ok(os.path.exists(exe), f'installed to {home}')
    menu = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Micrograph Maker.lnk')
    desk = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'Micrograph Maker.lnk')
    ok(os.path.exists(menu), 'Start menu shortcut')
    ok(os.path.exists(desk), 'desktop shortcut')
    data = os.path.join(TMP, 'data-installed')
    p = subprocess.Popen([exe, '--server', '8790', '--mode', 'window', '--parent', '0'], env=dict(os.environ, MM_DATA_DIR=data))
    t = wait_for(lambda: health(8790), 300)
    ok(t is not None, f'installed app starts ({t and round(t, 1)} s)')
    p.kill()
    p.wait(30)
    time.sleep(2)
    r = subprocess.run([os.path.join(home, 'unins000.exe'), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'], timeout=600)
    t = wait_for(lambda: not os.path.exists(exe), 120)
    ok(t is not None, 'uninstalled (program removed)')
    ok(not os.path.exists(menu), 'uninstalled (Start menu shortcut removed)')
    finish('installer')


def minos(app):
    """The newest 'minimum macOS' among all Mach-O files (otool -l), i.e. the oldest macOS the app runs on."""
    worst, rows = (0, 0), []
    for root, _, files in os.walk(app):
        for f in files:
            path = os.path.join(root, f)
            if os.path.islink(path):
                continue
            with open(path, 'rb') as fh:
                magic = fh.read(4)
            if magic not in (b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xce\xfa\xed\xfe'):
                continue
            out = subprocess.run(['otool', '-l', path], capture_output=True, text=True).stdout.split('\n')
            for i, line in enumerate(out):
                s = line.strip()
                if s.startswith('minos ') or (s.startswith('version ') and 'LC_VERSION_MIN_MACOSX' in ''.join(out[max(0, i - 3):i])):
                    v = tuple(int(x) for x in s.split()[1].split('.')[:2])
                    rows.append((v, os.path.relpath(path, app)))
                    worst = max(worst, v)
                    break
    rows.sort(reverse=True)
    with open(os.path.join(REPORT, 'min-macos.txt'), 'w') as fh:
        fh.write(f'oldest macOS the app supports: {worst[0]}.{worst[1]}\n\nnewest requirements:\n')
        fh.writelines(f'{v[0]}.{v[1]}  {p}\n' for v, p in rows[:25])
    print(f'oldest macOS the app supports: {worst[0]}.{worst[1]} ({len(rows)} programs)')
    ok(worst <= (12, 0), f'runs on macOS 12 or later (needs {worst[0]}.{worst[1]})')
    finish('minos')


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'server':
        server()
    elif what == 'window':
        window(True)
    elif what == 'dialog':
        window(False)
    elif what == 'installer':
        installer(sys.argv[2])
    elif what == 'minos':
        minos(sys.argv[2])
