import asyncio
import time

from control.queue import WorkerQueue


async def never_gone():
    return False


def test_grants_up_to_capacity_then_queues():
    async def go():
        q = WorkerQueue("WorkerA", max_in_flight=32)
        tickets = [q.enqueue("interactive", time.time()) for _ in range(33)]
        assert q.in_flight == 32 and q.depth() == 1
        assert tickets[32].state == "waiting"
        q.release()                                   # one seat frees: request 33 goes
        assert await q.wait(tickets[32], never_gone) == "granted"
        assert q.in_flight == 32 and q.depth() == 0
    asyncio.run(go())


def test_interactive_before_batch_then_arrival():
    async def go():
        q = WorkerQueue("WorkerA", max_in_flight=1)
        now = time.time()
        q.enqueue("batch", now)                       # takes the only seat
        b1 = q.enqueue("batch", now + 0.1)
        i1 = q.enqueue("interactive", now + 0.2)
        i2 = q.enqueue("interactive", now + 0.3)
        q.release()
        assert (i1.state, i2.state, b1.state) == ("granted", "waiting", "waiting")
        q.release()
        assert (i2.state, b1.state) == ("granted", "waiting")
        q.release()
        assert b1.state == "granted"
    asyncio.run(go())


def test_deadline_in_queue_times_out():
    async def go():
        q = WorkerQueue("WorkerA", max_in_flight=1)
        q.enqueue("interactive", time.time())
        late = q.enqueue("interactive", time.time(), deadline_s=0.3)
        assert await q.wait(late, never_gone) == "timeout"
        assert late.state == "cancelled" and q.depth() == 0
        q.release()                                   # a cancelled ticket is never granted
        assert late.state == "cancelled" and q.in_flight == 0
    asyncio.run(go())


def test_client_gone_while_queued_is_dropped():
    async def gone():
        return True

    async def go():
        q = WorkerQueue("WorkerA", max_in_flight=1)
        q.enqueue("batch", time.time())
        t = q.enqueue("batch", time.time())
        assert await q.wait(t, gone) == "gone"
        assert t.state == "cancelled" and q.depth() == 0
    asyncio.run(go())


def test_deadlines_by_tenant():
    q = WorkerQueue("WorkerA", max_in_flight=0)
    now = 1_000.0
    assert q.enqueue("interactive", now).deadline == now + 6.0
    assert q.enqueue("batch", now).deadline == now + 60.0
