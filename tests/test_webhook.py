"""Tests for the webhook receiver."""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

from ratiocinator.infra.webhook import WebhookReceiver


@pytest.fixture
def receiver():
    r = WebhookReceiver(host="127.0.0.1", port=0)
    r.start()
    # Get the actual port assigned
    r.port = r._server.server_address[1]
    yield r
    r.stop()


def test_receives_webhook(receiver):
    data = json.dumps({"instance_id": "test-1", "exit_code": 0}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{receiver.port}/",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    assert resp.status == 200

    # Give the handler time to process
    time.sleep(0.1)
    assert len(receiver.received) == 1
    assert receiver.received[0]["instance_id"] == "test-1"


def test_custom_callback():
    results = []
    r = WebhookReceiver(host="127.0.0.1", port=0, callback=results.append)
    r.start()
    port = r._server.server_address[1]

    try:
        data = json.dumps({"msg": "hello"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req)
        time.sleep(0.1)
        assert len(results) == 1
        assert results[0]["msg"] == "hello"
    finally:
        r.stop()
