# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

import yaml

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None
project_root = Path(SPEC).resolve().parents[1]
package_datas = (
    collect_data_files('gradio', include_py_files=True)
    + collect_data_files('safehttpx')
    + collect_data_files('groovy')
    + collect_data_files('webview')
)
package_binaries = collect_dynamic_libs('webview')


def _demo_checkpoint_datas():
    config_path = project_root / 'configs' / 'inference' / 'demo.yml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    runtime = config.get('runtime', {})
    checkpoint_paths = []
    for member in runtime.get('classifier_members', []) or []:
        checkpoint = member.get('checkpoint') if isinstance(member, dict) else None
        if checkpoint:
            checkpoint_paths.append(checkpoint)
    for key in ('classifier_checkpoint', 'segmenter_checkpoint'):
        if runtime.get(key):
            checkpoint_paths.append(runtime[key])
    for checkpoint in runtime.get('classifier_checkpoints', []) or []:
        if checkpoint:
            checkpoint_paths.append(checkpoint)

    datas = []
    seen = set()
    for checkpoint in checkpoint_paths:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if checkpoint_path.exists() and checkpoint_path not in seen:
            seen.add(checkpoint_path)
            datas.append((str(checkpoint_path), 'artifacts/checkpoints'))
    return datas


project_datas = [
    (str(project_root / 'configs'), 'configs'),
    *_demo_checkpoint_datas(),
]
hidden_imports = (
    ['gradio', 'cv2', 'webview', 'clr', 'pythonnet', 'clr_loader']
    + collect_submodules('webview')
    + collect_submodules('timm')
    + collect_submodules('segmentation_models_pytorch')
    + collect_submodules('pytorch_grad_cam')
)

a = Analysis(
    [str(project_root / 'app' / 'desktop_main.py')],
    pathex=[str(project_root)],
    binaries=package_binaries,
    datas=project_datas + package_datas,
    hiddenimports=hidden_imports,
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
    name='bucad-demo-desktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='bucad-demo-desktop',
)
