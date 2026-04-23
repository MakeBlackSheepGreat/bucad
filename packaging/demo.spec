# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
project_root = Path(SPEC).resolve().parents[1]
package_datas = (
    collect_data_files('gradio', include_py_files=True)
    + collect_data_files('safehttpx')
    + collect_data_files('groovy')
)
project_datas = [(str(project_root / 'configs'), 'configs')]
release_dir = project_root / 'artifacts' / 'release_v1'
if release_dir.exists():
    project_datas.append((str(release_dir), 'artifacts/release_v1'))

a = Analysis(
    [str(project_root / 'app' / 'main.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=project_datas + package_datas,
    hiddenimports=['gradio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='bucad-demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='bucad-demo',
)
