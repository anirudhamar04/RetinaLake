"""
Lab server: makes ChaksuDB available to other machines on the same network.

Starts two HTTP servers and prints the env vars lab users need:
  - Port 8090: serves the built wheel   → pip install chaksudb --find-links http://<host>:8090/
  - Port 8091: serves image files       → set STORAGE_IMAGE_SERVER_URL=http://<host>:8091

Usage:
    uv run python scripts/serve.py            # build wheel + start both servers
    uv run python scripts/serve.py --no-build # skip wheel rebuild (use existing dist/)
    uv run python scripts/serve.py --wheel-port 8090 --image-port 8091
"""

import argparse
import http.server
import ipaddress
import logging
import shutil
import socket
import subprocess
import sys
import threading
import urllib.parse
from pathlib import Path

# ── project root is the directory containing this script's parent ────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("serve")


# ── helpers ──────────────────────────────────────────────────────────────────

def local_ip() -> str:
    """Best-effort LAN IP (not 127.0.0.1)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def build_wheel() -> Path:
    """Run `uv build` and return the path to the newest .whl in dist/."""
    log.info("Building wheel …")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(DIST_DIR)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    wheels = sorted(DIST_DIR.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        sys.exit("ERROR: No wheel found in dist/ after build.")
    latest = wheels[0]
    fixed = DIST_DIR / "chaksudb.whl"
    shutil.copy2(latest, fixed)
    log.info("Wheel ready: %s", fixed)
    return fixed


def make_image_handler(roots: list[Path]):
    """
    Return an HTTP request handler that serves files from multiple root directories.

    A GET /EYEPACS/img.jpg will be satisfied by the first root that contains it.
    Both data_root and storage_root are searched so images and processed masks
    are all accessible under the same server.
    """
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            rel = urllib.parse.unquote(self.path.lstrip("/"))
            for root in roots:
                candidate = root / rel
                if candidate.is_file():
                    data = candidate.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.send_error(404, f"Not found: {self.path}")

        def log_message(self, fmt, *args):  # suppress per-request noise
            pass

        def handle_error(self, request, client_address):
            pass  # suppress BrokenPipeError noise from client disconnects

    return _Handler


def start_server(handler, port: int, label: str) -> http.server.HTTPServer:
    server = http.server.HTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name=label)
    t.start()
    log.info("%s listening on port %d", label, port)
    return server


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChaksuDB lab server")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip wheel rebuild; use whatever is already in dist/")
    parser.add_argument("--wheel-port", type=int, default=8090)
    parser.add_argument("--image-port", type=int, default=8091)
    args = parser.parse_args()

    # ── load storage config to find image roots ───────────────────────────
    sys.path.insert(0, str(PROJECT_ROOT))
    from chaksudb.config.config import storage_config

    data_root = storage_config.data_root.resolve()
    storage_root = storage_config.local_root.resolve()
    log.info("Image roots: %s | %s", data_root, storage_root)

    # ── build wheel ───────────────────────────────────────────────────────
    if args.no_build:
        wheels = list(DIST_DIR.glob("chaksudb.whl"))
        if not wheels:
            wheels = sorted(DIST_DIR.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not wheels:
            sys.exit("ERROR: No wheel in dist/ — run without --no-build first.")
        shutil.copy2(wheels[0], DIST_DIR / "chaksudb.whl")
        log.info("Using existing wheel: %s", wheels[0])
    else:
        build_wheel()

    # ── start servers ─────────────────────────────────────────────────────
    wheel_handler = http.server.SimpleHTTPRequestHandler
    wheel_handler.directory = str(DIST_DIR)  # type: ignore[attr-defined]

    class _WheelHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(DIST_DIR), **kw)
        def log_message(self, fmt, *args):
            pass

    image_handler = make_image_handler([data_root, storage_root])

    start_server(_WheelHandler, args.wheel_port, "WheelServer")
    start_server(image_handler, args.image_port, "ImageServer")

    # ── print lab-user instructions ───────────────────────────────────────
    ip = local_ip()
    wheel_index_url = f"http://{ip}:{args.wheel_port}/"
    image_url = f"http://{ip}:{args.image_port}"

    from chaksudb.config.config import db_config
    print()
    print("=" * 62)
    print("  ChaksuDB lab server is running")
    print("=" * 62)
    print()
    print("  Share these instructions with lab users:")
    print()
    print("  1. Install the package (once, or after each update):")
    print(f"       pip install chaksudb --find-links {wheel_index_url}")
    print(f"     # or:")
    print(f"       uv add chaksudb --find-links {wheel_index_url}")
    print()
    print("  2. Set these environment variables (add to .env or shell):")
    print()
    print(f"       DB_HOST={ip}")
    print(f"       DB_PORT={db_config.port}")
    print(f"       DB_DATABASE={db_config.database}")
    print(f"       DB_USER={db_config.user}")
    print(f"       DB_PASSWORD={db_config.password}")
    print(f"       STORAGE_IMAGE_SERVER_URL={image_url}")
    print()
    print("  3. Use:")
    print()
    print("       from chaksudb.export.spec import ExportSpec")
    print("       from chaksudb.export.api import export")
    print()
    print("       spec = ExportSpec(")
    print('           dataset_names=["MESSIDOR"],')
    print('           annotation_tasks=["grading"],')
    print("       )")
    print('       export(spec, parquet_path="out.parquet")')
    print()
    print("     Images are downloaded on first access and cached in")
    print("     ~/.cache/chaksudb/  —  no re-downloads on subsequent runs.")
    print()
    print("=" * 62)
    print("  Press Ctrl-C to stop.")
    print("=" * 62)
    print()

    try:
        threading.Event().wait()  # block forever
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
