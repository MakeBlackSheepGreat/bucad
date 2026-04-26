from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ensure_standard_streams() -> None:
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


_ensure_standard_streams()

from app.main import build_app


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
WINDOW_TITLE = "BUCAD 乳腺超声辅助诊断系统"


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _preferred_server_port(host: str, preferred_port: int) -> int | None:
    if preferred_port > 0 and _is_port_available(host, preferred_port):
        return preferred_port
    return None


def start_gradio_server(
    *,
    config_path: str | Path | None = None,
    host: str = DEFAULT_HOST,
    preferred_port: int = DEFAULT_PORT,
):
    app = build_app(config_path)
    server_port = _preferred_server_port(host, preferred_port)
    _, local_url, _ = app.launch(
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        server_name=host,
        server_port=server_port,
        quiet=True,
    )
    return app, local_url


def open_desktop_window(
    url: str,
    *,
    title: str = WINDOW_TITLE,
    width: int = 1440,
    height: int = 900,
    debug: bool = False,
) -> None:
    import webview

    webview.create_window(
        title,
        url,
        width=width,
        height=height,
        min_size=(1180, 760),
        resizable=True,
        text_select=True,
        confirm_close=False,
    )
    webview.start(gui="edgechromium", debug=debug, private_mode=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch BUCAD in a desktop WebView window.")
    parser.add_argument("--config", type=Path, default=None, help="Inference config path.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Local server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Preferred local server port.")
    parser.add_argument("--width", type=int, default=1440, help="Window width.")
    parser.add_argument("--height", type=int, default=900, help="Window height.")
    parser.add_argument("--debug", action="store_true", help="Enable WebView debug mode.")
    parser.add_argument(
        "--server-only-seconds",
        type=float,
        default=0,
        help="Start only the local Gradio server for this many seconds, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_standard_streams()
    args = parse_args()
    app, local_url = start_gradio_server(
        config_path=args.config,
        host=args.host,
        preferred_port=args.port,
    )
    try:
        if args.server_only_seconds > 0:
            print(local_url)
            time.sleep(args.server_only_seconds)
            return
        open_desktop_window(
            local_url,
            width=args.width,
            height=args.height,
            debug=args.debug,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
