#!/usr/bin/env python3
"""Static file server with COOP/COEP headers, stdlib only.

Cross-origin isolation (`self.crossOriginIsolated === true`) is required for
`SharedArrayBuffer` and atomics, which a threaded wasm build (the
wasm-bindgen-rayon pool) needs to start its workers. GitHub Pages cannot set
these headers, so this server is for local testing only, the same approach
as t0-web's local `coi_server.py` (idle-intelligence/t0-web PR #8).

Usage:
    python3 crates/lean-runtime/scripts/serve_coi.py [--dir DIR] [--port PORT]
"""
import argparse
import http.server
import os
import socketserver
import sys


class CoiHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=".")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    handler = lambda *a, **kw: CoiHandler(*a, directory=root, **kw)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {root} on http://{args.bind}:{args.port} (COOP/COEP set, Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
