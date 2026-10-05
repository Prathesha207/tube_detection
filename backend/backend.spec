# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [
    ('app/ml/model', 'app/ml/model'),
    ('app/ml/config', 'app/ml/config'),
    ('alembic', 'alembic'),
]
binaries = []
hiddenimports = [
    'pydantic',
    'pydantic_settings',
    'pydantic_core',
    'skimage',
    'skimage.morphology',
    'yaml',
    'multipart',
    'aiofiles',
]

datas += copy_metadata('torchvision')
datas += copy_metadata('ultralytics')
try:
    datas += copy_metadata('head_tail_analyzer')
except Exception:
    pass

for pkg in [
    'app', 'fastapi', 'starlette', 'uvicorn', 'sqlalchemy',
    'cv2', 'torch', 'torchvision', 'ultralytics', 'depthai',
    'av', 'scipy', 'imageio_ffmpeg', 'head_tail_analyzer'
]:
    try:
        tmp_ret = collect_all(pkg)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception:
        pass

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'PyQt5', 'PyQt6', 'wx'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend',
)
