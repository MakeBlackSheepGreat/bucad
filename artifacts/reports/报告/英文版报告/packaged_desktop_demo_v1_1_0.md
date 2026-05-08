<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->
# Packaged Desktop Demo v1.1.0

## Summary

- Version: `v1.1.0`
- Build type: Windows desktop WebView package
- Source folder: `dist/bucad-demo-desktop/`
- Entrypoint: `dist/bucad-demo-desktop/bucad-demo-desktop.exe`
- Release folder: `Release/v1.1.0/`

## Assets

- Full archive: `Release/v1.1.0/BUCAD-v1.1.0-windows-desktop-demo.zip`
- SHA256: `15adeb889b91d19f3f5b257be401e0ad92a3924c05539de464a3b2bfde5c22fb`
- Split parts: `3`
- Merge script: `Release/v1.1.0/merge_bucad_v1_1_0_parts_windows.ps1`

## Notes

- The release assets are intentionally excluded from Git via `Release/` and `dist/`.
- The desktop package wraps the existing local Gradio UI inside a Windows WebView window.
- The source changes are tracked separately through `app/desktop_main.py` and `packaging/desktop_demo.spec`.
