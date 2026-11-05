# -*- coding: utf-8 -*-
"""Check the built PC version's server in a real browser (Chromium through Playwright).

The page opens without an error; the sample Word report and SEM picture (desktop/ci/samples.py, no real data) upload
and are read (element labels, scales); PDF, PNG, TIFF, SVG, the per-picture ZIP and the work file (.edsfig) download
with the right contents; the learned readings go to EDSFIG_HOME (not into the app); 「アプリを終了」 is offered only in
the browser mode and stops the server there.

usage: python smoke_server.py BASE_URL OUT_DIR SAMPLES MODE(window|browser) [EDSFIG_HOME]
       SAMPLES: the sample files, os.pathsep between
"""
import os
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

BASE, OUT, SAMPLES, MODE = sys.argv[1].rstrip('/'), sys.argv[2], sys.argv[3].split(os.pathsep), sys.argv[4]
HOME = sys.argv[5] if len(sys.argv) > 5 else ''
os.makedirs(OUT, exist_ok=True)
EXC = '[data-testid="stException"]'
MAGIC = {'pdf': (b'%PDF',), 'png': (b'\x89PNG',), 'tif': (b'II*\x00', b'MM\x00*'), 'svg': (b'<?xml', b'<svg'),
         'zip': (b'PK\x03\x04',), 'edsfig': (b'PK\x03\x04',)}
REPORT_FILE = os.path.join(OUT, f'smoke_server_{MODE}.txt')
open(REPORT_FILE, 'w', encoding='utf-8').close()
fails = []


def ok(cond, what):
    line = ('OK   ' if cond else 'FAIL ') + what
    print(line, flush=True)
    with open(REPORT_FILE, 'a', encoding='utf-8') as fh:            # 途中で止まっても、そこまでの結果が残る
        fh.write(line + '\n')
    if not cond:
        fails.append(what)


def settle(pg, ms=1500):
    pg.wait_for_timeout(ms)
    for _ in range(240):
        st = pg.locator('[data-testid="stStatusWidget"]')
        try:
            busy = st.count() and 'running' in (st.first.inner_text(timeout=1000) or '').lower()
        except Exception:  # noqa: BLE001
            busy = False
        if not busy:
            break
        pg.wait_for_timeout(500)
    pg.wait_for_timeout(500)


def healthy():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(BASE + '/_stcore/health', timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def captions(pg):
    return ' | '.join(pg.locator('[data-testid="stCaptionContainer"]').all_inner_texts())


def download(pg, button, ext, tries=3):
    """Click a download button (found again each time) and check the file. A click that lands while the page is
    still redrawing after the previous download is lost, so it is tried again."""
    err = None
    for attempt in range(1, tries + 1):
        settle(pg)
        try:
            with pg.expect_download(timeout=90000) as d:
                button().click(timeout=90000)                       # waits while the page is still busy
            path = os.path.join(OUT, d.value.suggested_filename)
            d.value.save_as(path)
            with open(path, 'rb') as fh:
                data = fh.read()
            good = any(data.startswith(m) for m in MAGIC[ext]) and len(data) > 100
            ok(good, f'{d.value.suggested_filename}: {len(data)} bytes, {data[:6]!r}'
               + (f' (click {attempt})' if attempt > 1 else ''))
            return path
        except Exception as e:  # noqa: BLE001
            err = e
            try:
                state = [(b.inner_text()[:16], b.is_disabled()) for b in pg.locator('[data-testid="stDownloadButton"] button').all()]
                pg.screenshot(path=os.path.join(OUT, f'{MODE}-download-{ext}-{attempt}.png'), full_page=True)
            except Exception:  # noqa: BLE001
                state = '?'
            print(f'download {ext}: no download after click {attempt} ({repr(e)[:120]}); buttons {state}', flush=True)
    ok(False, f'download {ext}: {err!r}'[:300])
    return None


def pick_figure(pg, text):
    for _ in range(3):
        pg.get_by_text('4. プレビュー', exact=True).first.evaluate('e => window.scrollTo(0, 0)')
        pg.locator('[data-testid="stSelectbox"]').filter(has_text='図').first.click()
        opt = pg.locator('[role="option"]').filter(has_text=text).first
        try:
            opt.wait_for(state='visible', timeout=5000)
            opt.click()
            settle(pg, 2500)
            return True
        except Exception:  # noqa: BLE001
            pg.keyboard.press('Escape')
            settle(pg)
    return False


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1500, 'height': 2600}, accept_downloads=True)
    t0 = time.time()
    pg.goto(BASE)
    pg.wait_for_selector('input[type="file"]', state='attached', timeout=180000)
    settle(pg)
    ok(pg.title() == 'Micrograph Maker', f'page opens: "{pg.title()}" ({time.time() - t0:.1f} s)')
    ok(pg.locator(EXC).count() == 0, 'no error on the first page')
    quit_button = pg.get_by_role('button', name='アプリを終了')
    ok((quit_button.count() > 0) == (MODE == 'browser'), f'「アプリを終了」 only in the browser mode ({MODE}: {quit_button.count()})')
    pg.screenshot(path=os.path.join(OUT, f'{MODE}-1-start.png'))

    pg.set_input_files('input[type="file"]', SAMPLES)
    pg.wait_for_selector('text=4. プレビュー', timeout=300000)
    settle(pg, 3000)
    cap = captions(pg)
    cards = ' '.join(pg.locator('[data-testid="caption"], [data-testid="stImageCaption"]').all_inner_texts())
    ok('1画素 = 65.79 nm' in cap, 'the Word report: 5 µm over 76 px read (65.79 nm per pixel)')
    ok('2. Ti' in cards and '3. O' in cards, f'the Word report: element labels read ({cards[:80]})')
    ok(pg.locator(EXC).count() == 0, 'no error after loading the samples')
    pg.screenshot(path=os.path.join(OUT, f'{MODE}-2-report.png'), full_page=True)

    for label, ext in (('PDF', 'pdf'), ('PNG', 'png'), ('TIFF', 'tif'), ('SVG', 'svg'), ('画像ごと（ZIP）', 'zip')):
        download(pg, lambda label=label: pg.locator('[data-testid="stDownloadButton"] button', has_text=label).first, ext)
    settle(pg)
    pg.locator('[data-testid="stSidebar"] [data-testid="stExpander"]').filter(has_text='作業の保存').first.click()
    download(pg, lambda: pg.locator('[data-testid="stSidebar"] [data-testid="stDownloadButton"] button',
                                    has_text='作業を保存').first, 'edsfig')
    settle(pg)
    if HOME:
        ok(os.path.exists(os.path.join(HOME, 'exemplars.json')), 'the confirmed readings are learned into EDSFIG_HOME')

    ok(pick_figure(pg, 'SEM像'), 'the SEM picture is its own figure')
    ok('1画素 = 166.67 nm' in captions(pg), 'the SEM picture: 50 µm over 300 px read in its band (166.67 nm per pixel)')
    ok(pg.locator(EXC).count() == 0, 'no error at the end')
    pg.screenshot(path=os.path.join(OUT, f'{MODE}-3-sem.png'), full_page=True)

    if MODE == 'browser':
        pg.get_by_role('button', name='アプリを終了').click()
        pg.get_by_role('button', name='終了する').click()
        t1 = time.time()
        while time.time() - t1 < 30 and healthy():
            time.sleep(0.5)
        ok(not healthy(), f'「アプリを終了」 stops the server ({time.time() - t1:.1f} s)')
    b.close()

print('\nFAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
