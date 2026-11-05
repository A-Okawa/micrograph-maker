# -*- mode: python ; coding: utf-8 -*-
# PyInstaller の設定：Micrograph Maker の PC 版（Mac は Micrograph Maker.app、Windows は MicrographMaker\MicrographMaker.exe）を作る。
# リポジトリのいちばん上で:  pyinstaller desktop/micrograph_maker.spec --noconfirm --clean
# app.py・appinfo.py・edsfig はそのままの .py でアプリの中に入れ、Streamlit がソース版と同じように実行する（figapp の PC 版と同じ作り）。
import ast
import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
DESK = os.path.join(ROOT, 'desktop')
sys.path.insert(0, ROOT)
VERSION = re.search(r'__version__ = "([^"]+)"', open(os.path.join(ROOT, 'edsfig', '__init__.py'), encoding='utf-8').read()).group(1)
NAME = 'Micrograph Maker'          # Finder・メニュー・ショートカットに出る名前
EXE_NAME = 'MicrographMaker'       # 実行ファイルとフォルダの名前（空白なし）


def imports_of(path):
    """Modules imported by app.py: PyInstaller does not look inside it (Streamlit runs it as a file)."""
    names = set()
    for node in ast.walk(ast.parse(open(path, encoding='utf-8').read())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return sorted(names)


datas = [(os.path.join(ROOT, 'app.py'), '.'),
         (os.path.join(ROOT, '.streamlit', 'config.toml'), '.streamlit'),
         (os.path.join(ROOT, 'edsfig', 'assets'), os.path.join('edsfig', 'assets'))]   # 読み取り用フォント・初期の記憶・アイコン
datas += collect_data_files('streamlit', excludes=['**/tests/**', '**/.agents/**'])   # 画面（static）と部品のひな形
for dist in ('streamlit', 'altair', 'narwhals', 'pyarrow', 'pandas', 'numpy', 'matplotlib', 'pillow', 'starlette', 'uvicorn',
             'websockets', 'protobuf', 'pydeck', 'click', 'packaging', 'requests', 'jsonschema', 'typing_extensions',
             'anyio', 'python-multipart', 'itsdangerous'):
    datas += copy_metadata(dist)

hiddenimports = ['appinfo', 'selftest'] + collect_submodules('edsfig') + imports_of(os.path.join(ROOT, 'app.py'))
hiddenimports += collect_submodules('streamlit', filter=lambda n: not n.startswith(('streamlit.testing', 'streamlit.hello',
                                                                                     'streamlit.external')))
hiddenimports += collect_submodules('uvicorn') + collect_submodules('websockets') + collect_submodules('starlette')
hiddenimports += ['matplotlib.backends.backend_agg', 'matplotlib.backends.backend_pdf', 'matplotlib.backends.backend_svg',
                  'matplotlib.backends.backend_ps', 'PIL.TiffImagePlugin', 'PIL.PngImagePlugin', 'PIL.JpegImagePlugin',
                  'PIL.BmpImagePlugin']
if sys.platform == 'darwin':
    hiddenimports += ['webview.platforms.cocoa', 'Foundation', 'AppKit', 'WebKit']
elif sys.platform == 'win32':
    hiddenimports += ['webview.platforms.winforms', 'webview.platforms.edgechromium', 'clr']

# pyarrow は Streamlit と pandas が lib・compute・ipc だけを使う。どの部品からも使われない Flight・S3 などは入れない（約 50 MB 減る）。
# Parquet・Dataset・Acero・Substrait は libarrow_python が読み込むので残す（figapp の PC 版で確かめた分け方）
PYARROW_UNUSED = ['pyarrow.' + m for m in (
    'flight', '_flight', '_s3fs', '_gcsfs', '_azurefs', '_hdfs', 'orc', '_orc', '_dataset_orc', 'cuda', '_cuda',
    '_pyarrow_cpp_tests', 'tests', 'benchmark', '_parquet_encryption', '_dataset_parquet_encryption', 'parquet.encryption')]
_DROP = ('flight', 'arrow_s3', '_s3fs', 'gcsfs', 'azurefs', 'hdfs', 'orc', 'cuda', 'cpp_tests', 'encryption')


def _keep(dest):
    d = dest.replace('\\', '/')
    name = d.rsplit('/', 1)[-1].lower()
    # pyarrow の中のものと、PyInstaller がいちばん上に作る pyarrow のライブラリへのリンク（libarrow_flight.so.2600 など）
    if not (d.startswith('pyarrow/') or name.startswith(('libarrow', 'arrow', 'libparquet', 'parquet'))):
        return True
    if d.startswith('pyarrow/include/') or d.startswith('pyarrow/tests/') or name.endswith(('.pyx', '.pxd', '.pxi', '.h', '.cc')):
        return False
    return not any(k in name for k in _DROP)


a = Analysis(
    [os.path.join(DESK, 'launcher.py')],
    pathex=[ROOT, DESK],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['tkinter', 'IPython', 'pytest', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'gi', 'scipy', 'webview.platforms.qt',
              'webview.platforms.gtk', 'webview.platforms.cef', 'webview.platforms.android'] + PYARROW_UNUSED,
    module_collection_mode={'edsfig': 'py', 'appinfo': 'py'},             # .py のまま app.py の隣に置く
    noarchive=False,
)
a.binaries = [b for b in a.binaries if _keep(b[0])]
a.datas = [d for d in a.datas if _keep(d[0])]
pyz = PYZ(a.pure)

icon = os.path.join(DESK, 'icons', 'MicrographMaker.icns' if sys.platform == 'darwin' else 'MicrographMaker.ico')
version_file = None
if sys.platform == 'win32':
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                     VarFileInfo, VarStruct, VSVersionInfo)
    nums = tuple(int(x) for x in (VERSION.split('.') + ['0', '0', '0'])[:4])
    version_file = VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[StringFileInfo([StringTable('041104B0', [
            StringStruct('ProductName', NAME), StringStruct('FileDescription', f'{NAME}（顕微鏡像・元素マップの図）'),
            StringStruct('FileVersion', VERSION), StringStruct('ProductVersion', VERSION),
            StringStruct('InternalName', EXE_NAME), StringStruct('OriginalFilename', f'{EXE_NAME}.exe'),
            StringStruct('CompanyName', NAME), StringStruct('LegalCopyright', NAME)])]),
              VarFileInfo([VarStruct('Translation', [0x0411, 1200])])])

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXE_NAME,
    console=False,
    icon=icon,
    version=version_file,
    disable_windowed_traceback=False,
    argv_emulation=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=EXE_NAME)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name=f'{NAME}.app',
        icon=icon,
        bundle_identifier='io.github.a-okawa.micrograph-maker',
        version=VERSION,
        info_plist={
            'CFBundleName': NAME,
            'CFBundleDisplayName': NAME,
            'CFBundleShortVersionString': VERSION,
            'CFBundleVersion': VERSION,
            'LSMinimumSystemVersion': '12.0',
            'LSApplicationCategoryType': 'public.app-category.education',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'NSAppTransportSecurity': {'NSAllowsLocalNetworking': True},   # http://127.0.0.1 の画面を窓に出す
            'NSHumanReadableCopyright': NAME,
        },
    )
