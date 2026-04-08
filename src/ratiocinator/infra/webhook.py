"""Webhook receiver for collecting metrics from remote instances."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

logger = logging.getLogger(__name__)


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler that receives metric pushes from Vast.ai instances."""

    callback: Callable[[dict[str, Any]], None] | None = None

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            logger.info("Webhook received from instance %s", data.get("instance_id"))
            if self.callback:
                self.callback(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        except Exception:
            logger.exception("Failed to process webhook")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)


class WebhookReceiver:
    """Runs a lightweight HTTP server to receive metrics from instances."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.callback = callback
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None
        self.received: list[dict[str, Any]] = []

    def _default_callback(self, data: dict[str, Any]) -> None:
        self.received.append(data)

    def start(self) -> None:
        """Start the webhook server in a background thread."""
        cb = self.callback or self._default_callback
        WebhookHandler.callback = cb

        self._server = HTTPServer((self.host, self.port), WebhookHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Webhook receiver started on %s:%s", self.host, self.port)

    def stop(self) -> None:
        """Stop the webhook server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Webhook receiver stopped")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"
