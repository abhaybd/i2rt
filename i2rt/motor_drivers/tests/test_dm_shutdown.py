"""CAN shutdown must leave the socket open until every in-flight transaction finishes."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from i2rt.motor_drivers import dm_driver as dm


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> dm.DMChainCanInterface:
    interface = SimpleNamespace(closed=False)

    def close() -> None:
        interface.closed = True

    def motor_on(self: dm.DMChainCanInterface) -> None:
        self.state = [SimpleNamespace(torque=0.0)]
        self.running = True

    interface.close = close
    monkeypatch.setattr(dm, "run_startup_checks", lambda *args, **kwargs: None)
    monkeypatch.setattr(dm, "DMSingleMotorCanInterface", lambda **kwargs: interface)
    monkeypatch.setattr(dm.DMChainCanInterface, "_motor_on", motor_on)
    monkeypatch.setattr(dm.DMChainCanInterface, "_update_absolute_positions", lambda *args: None)
    return dm.DMChainCanInterface([(1, "DM4310")], [0.0], [1], "can_test", start_thread=False)


@pytest.mark.parametrize("same_bus_read", [False, True], ids=["motor-command", "encoder-read"])
def test_close_waits_for_in_flight_transaction(
    chain: dm.DMChainCanInterface, monkeypatch: pytest.MonkeyPatch, same_bus_read: bool
) -> None:
    entered = threading.Event()
    finish = threading.Event()
    joining = threading.Event()
    errors = []

    def transaction() -> None:
        entered.set()
        if not finish.wait(timeout=5):
            errors.append("transaction was never released")
        if chain.motor_interface.closed:
            errors.append("CAN socket closed during transaction")

    def commands(_commands: Any) -> list[Any]:
        if not same_bus_read:
            transaction()
        return [SimpleNamespace(error_code="0x1")]

    monkeypatch.setattr(chain, "_set_commands", commands)
    if same_bus_read:
        chain.same_bus_device_driver = SimpleNamespace(read_states=transaction)
    chain.start_thread()
    assert entered.wait(timeout=2)
    worker = chain._control_thread
    assert worker is not None
    original_join = worker.join

    def join(timeout: float | None = None) -> None:
        joining.set()
        original_join(timeout)

    monkeypatch.setattr(worker, "join", join)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing = executor.submit(chain.close)
            try:
                assert joining.wait(timeout=2)
                assert not chain.motor_interface.closed
                assert not closing.done()
            finally:
                finish.set()
            closing.result(timeout=2)
    finally:
        finish.set()
        chain.close()
    assert not worker.is_alive()
    assert chain.motor_interface.closed
    assert not errors


def test_close_without_starting_thread(chain: dm.DMChainCanInterface) -> None:
    chain.close()
    assert not chain.running
    assert chain.motor_interface.closed


def test_close_is_safe_to_repeat_after_worker_exits(
    chain: dm.DMChainCanInterface, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop() -> None:
        chain.running = False

    monkeypatch.setattr(chain, "_set_torques_and_update_state", stop)
    chain.start_thread()
    chain.close()
    chain.close()
    assert chain._control_thread is not None
    assert not chain._control_thread.is_alive()
    assert chain.motor_interface.closed
