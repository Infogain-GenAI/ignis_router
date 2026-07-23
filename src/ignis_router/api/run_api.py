"""Entrypoint to run the Ignis Router API with a stable default port."""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request

import uvicorn


def _resolve_port() -> int:
    raw_port = os.getenv("IGNIS_ROUTER_API_PORT") or os.getenv("API_PORT") or "8080"
    try:
        return int(raw_port)
    except ValueError:
        return 8013


def _resolve_reload() -> bool:
    raw_value = (os.getenv("IGNIS_ROUTER_API_RELOAD") or "false").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _is_api_healthy(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main() -> None:
    port = _resolve_port()

    if not _is_port_available(port):
        if _is_api_healthy(port):
            print(f"Ignis Router API is already running on http://127.0.0.1:{port}")
            return

        raise RuntimeError(
            f"Port {port} is in use by another process. "
            f"Free the port or set API_PORT/IGNIS_ROUTER_API_PORT to another value."
        )

    uvicorn.run(
        "ignis_router.api:create_app",
        factory=True,
        host="127.0.0.1",
        port=port,
        reload=_resolve_reload(),
    )


if __name__ == "__main__":
    main()
