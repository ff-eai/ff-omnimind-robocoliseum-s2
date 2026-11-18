# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any

import msgpack
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from cap_harness.g2s2.wire import S2Request, decode_request, encode_chunk, hold_chunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BridgeEvent:
    kind: str

    request: S2Request | None = None
    connection: int = 0
    at_s: float = 0.0


class Superseded(RuntimeError):
    pass


DEFAULT_KEEPALIVE_BY_RUNTIME = {"rubik": 50.0, "newton": 600.0}


class _Pending:
    __slots__ = ("answered_by", "event", "request", "result")

    def __init__(self, request: S2Request) -> None:
        self.request = request
        self.event = threading.Event()
        self.result: dict[str, Any] | None = None
        self.answered_by = ""


def done_ack(request: S2Request) -> dict[str, Any]:
    chunk = hold_chunk(request, 1)
    return encode_chunk(chunk, request.embodiment, request.runtime, need_depth=False, contract=request.action_contract)


class S2Bridge:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        keepalive_s: float | None = None,
        keepalive_by_runtime: dict[str, float] | None = None,
        save_first_requests: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.keepalive_s = keepalive_s
        self.keepalive_by_runtime = dict(DEFAULT_KEEPALIVE_BY_RUNTIME if keepalive_by_runtime is None else keepalive_by_runtime)
        self.hold_provider: Any = None
        self._keepalive_answered: set[int] = set()
        self.save_first_requests = save_first_requests
        self.log_path = log_path
        self._events: queue.Queue[BridgeEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._pending: _Pending | None = None
        self._sequence = 0
        self._connections = 0
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.bound_port: int | None = None
        self.stats = {"requests": 0, "done_acks": 0, "keepalives": 0, "errors": 0, "bytes_in": 0}

    def start(self) -> S2Bridge:
        self._server = serve(
            self._handler, self.host, self.port, max_size=None, compression=None, ping_interval=20, ping_timeout=60
        )
        self.bound_port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="s2-bridge", daemon=True)
        self._thread.start()
        logger.info("S2 bridge listening on ws://%s:%d", self.host, self.bound_port)
        return self

    def close(self) -> None:
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is not None and not pending.event.is_set():
            pending.answered_by = "shutdown"
            pending.event.set()
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def next_event(self, timeout: float | None = None) -> BridgeEvent | None:
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def reply(self, request: S2Request, result: dict[str, Any]) -> int:
        payload = {"result": result}
        size = len(msgpack.packb(payload, use_bin_type=True))
        with self._lock:
            pending = self._pending
            if pending is None or pending.request.sequence != request.sequence:
                if request.sequence in self._keepalive_answered:
                    raise Superseded(f"request {request.sequence} was answered by a keepalive")
                raise RuntimeError(f"request {request.sequence} is not pending (client gone or already answered)")
            pending.result = result
            pending.answered_by = "executor"
            self._pending = None
        pending.event.set()
        return size

    @property
    def connected(self) -> bool:
        return self._connections > 0

    def _log(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.exception("bridge log write failed")

    def _handler(self, connection: Any) -> None:
        with self._lock:
            self._connections += 1
            connection_id = self._connections
        logger.info("GenieSim client connected (#%d)", connection_id)
        try:
            for message in connection:
                if isinstance(message, str):
                    connection.send(msgpack.packb({"error": "binary MessagePack expected"}, use_bin_type=True))
                    continue
                self._handle(connection, connection_id, message)
        except ConnectionClosed:
            pass
        finally:
            with self._lock:
                self._connections -= 1
                pending = self._pending
                if pending is not None:
                    self._pending = None
            if pending is not None and not pending.event.is_set():
                pending.answered_by = "disconnect"
                pending.event.set()
            self._events.put(BridgeEvent("disconnect", None, connection_id, time.time()))
            logger.info("GenieSim client #%d disconnected", connection_id)

    def _handle(self, connection: Any, connection_id: int, message: bytes) -> None:
        started = time.time()
        try:
            envelope = msgpack.unpackb(message, raw=False)
            if not isinstance(envelope, dict) or envelope.get("method") != "infer":
                raise ValueError("expected {'method': 'infer', 'params': ...}")
            with self._lock:
                sequence = self._sequence
                self._sequence += 1
            request = decode_request(envelope.get("params") or {}, sequence=sequence, nbytes=len(message))
        except Exception as exc:
            self.stats["errors"] += 1
            logger.warning("rejected request: %s", exc)
            connection.send(msgpack.packb({"error": f"worker could not decode the request: {exc}"}, use_bin_type=True))
            return
        self.stats["requests"] += 1
        self.stats["bytes_in"] += len(message)
        if request.episode_done:
            result = done_ack(request)
            connection.send(msgpack.packb({"result": result}, use_bin_type=True))
            self.stats["done_acks"] += 1
            self._events.put(BridgeEvent("episode_done", request, connection_id, started))
            self._log({"t": started, "seq": request.sequence, "kind": "episode_done", "answered": "done_ack", **request.summary})
            return
        if self.save_first_requests is not None and request.has_depth and request.episode_start is not False:
            self._maybe_save(envelope, request)
        pending = _Pending(request)
        with self._lock:
            self._pending = pending
        self._events.put(BridgeEvent("request", request, connection_id, started))
        timeout = self.keepalive_by_runtime.get(request.runtime, self.keepalive_s)
        waited = pending.event.wait(timeout) if timeout else pending.event.wait()
        if not waited:
            hold = None
            if self.hold_provider is not None:
                try:
                    hold = self.hold_provider(request)
                except Exception:
                    logger.exception("hold provider failed")
            keepalive = False
            with self._lock:
                if self._pending is pending:
                    self._pending = None
                    pending.result = hold or encode_chunk(
                        hold_chunk(request, 1), request.embodiment, request.runtime, contract=request.action_contract
                    )
                    pending.answered_by = "keepalive"
                    self.stats["keepalives"] += 1
                    self._keepalive_answered.add(request.sequence)
                    self._events.put(BridgeEvent("keepalive", request, connection_id, time.time()))
                    keepalive = True
            if not keepalive:
                pending.event.wait()
        if pending.result is None:
            return
        connection.send(msgpack.packb({"result": pending.result}, use_bin_type=True))
        self._log(
            {
                "t": started,
                "seq": request.sequence,
                "kind": "reply",
                "answered": pending.answered_by,
                "reply_s": round(time.time() - started, 3),
                **request.summary,
            }
        )

    def _maybe_save(self, envelope: dict[str, Any], request: S2Request) -> None:
        try:
            folder = Path(self.save_first_requests)
            folder.mkdir(parents=True, exist_ok=True)
            name = f"req_{request.sequence:06d}_{request.task_name}_{request.episode_idx}.msgpack"
            (folder / name).write_bytes(msgpack.packb(envelope, use_bin_type=True))
        except OSError:
            logger.exception("could not save the request")


def raw_request_from_samples(sample_dir: Path, depth_dir: Path) -> dict[str, Any]:
    import hashlib

    params = json.loads((sample_dir / "request.json").read_text(encoding="utf-8"))
    images = params.get("images") or {}
    for name, record in images.items():
        descriptor = record.get("image_data")
        if not isinstance(descriptor, dict):
            continue
        if record.get("encoding") == "RAW_UINT16":
            array = np.load(depth_dir / f"{name}.npy").astype("<u2")
            data = array.tobytes()
        else:
            data = (sample_dir / f"{name}.jpg").read_bytes()
        if len(data) != int(descriptor["_bytes"]) or hashlib.sha256(data).hexdigest()[:16] != descriptor["_sha256"]:
            raise ValueError(f"{sample_dir.name}/{name}: bytes do not match the saved descriptor")
        record["image_data"] = data
    return params


__all__ = ["DEFAULT_KEEPALIVE_BY_RUNTIME", "BridgeEvent", "S2Bridge", "Superseded", "done_ack", "raw_request_from_samples"]
