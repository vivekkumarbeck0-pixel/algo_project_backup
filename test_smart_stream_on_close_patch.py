from angel_one.smart_stream import AngelSmartWebSocketClient


def test_on_close_accepts_websocket_client_arguments():
    calls = []

    class FakeSocket:
        def _on_close(self, wsapp):
            calls.append(wsapp)

    AngelSmartWebSocketClient._patch_on_close(FakeSocket)
    AngelSmartWebSocketClient._patch_on_close(FakeSocket)  # idempotent

    FakeSocket()._on_close("ws", 1000, "bye")
    FakeSocket()._on_close("ws2")
    assert calls == ["ws", "ws2"]
