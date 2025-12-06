from contextlib import contextmanager
from threading import Event, Lock, RLock, Semaphore, Thread, Condition
from typing import Any
from collections import Counter, deque
import time

class ThreadSafeCounter:
    def __init__(self):
      self._counter = 0
      self._lock = Lock()

    def increment(self):
      #with self._lock:
      self._counter += 1

    def decrement(self):
      with self._lock:
        self._counter -= 1
    
    def value(self) -> int:
      return self._counter

class BankAccount:
    def __init__(self, balance=0):
      self._balance = balance
      self._lock = RLock()

    def deposit(self, amount: int):
      assert amount > 0
      with self._lock:
        self._balance += amount

    def withdraw(self, amount: int) -> bool:
      if amount > self._balance:
        return False
      else:
        with self._lock:
          self._balance -= amount
        return True

    def transfer_to(self, other: "BankAccount", amount: int) -> bool:
      first, second = (self, other) if id(self) < id(other) else (other, self)
      with first._lock:
        with second._lock:
          if self.withdraw(amount):
            other.deposit(amount)
            return True
          else:
            return False

class BoundedBlockingQueue:
    def __init__(self, capacity: int):
      self._capacity = capacity
      self._q = deque()
      self._condition = Condition()

    def put(self, item: Any):
      with self._condition:
        if len(self._q) == self._capacity:
          self._condition.wait()
        self._q.append(item)
        self._condition.notify()

    def get(self) -> Any:
      with self._condition:
        if len(self._q) == 0:
          self._condition.wait()
        v = self._q.popleft()
        self._condition.notify()
        return v

class Future:
    def __init__(self):
      self._condition = Condition()
      self._result = None

    def set_result(self, value):
      with self._condition:
        self._result = value
        self._condition.notify_all()

    def result(self, timeout=None):
      with self._condition:
        if self._result is None:
          if not self._condition.wait(timeout=timeout):
            raise TimeoutError
        return self._result

class ConnectionPool:
    def __init__(self, max_connections: int):
      self._sem = Semaphore(max_connections)
      self._connections = [i for i in range(max_connections)]
      self._connections_lock = Lock()
    
    @contextmanager
    def acquire(self):
      self._sem.acquire()
      with self._connections_lock:
        conn = self._connections.pop(0)
      yield conn
      with self._connections_lock:
        self._connections.append(conn)
      self._sem.release()

class CancellationToken:
    def __init__(self):
      self._cancelled_event = Event()

    def cancel(self):
      self._cancelled_event.set()

    def is_cancelled(self) -> bool:
      return self._cancelled_event.is_set()

    def wait_cancelled(self, timeout=None) -> bool:
      self._cancelled_event.wait(timeout=timeout)

class ReadWriteLock:
  def __init__(self):
    self._condition = Condition()
    self.writer_active = False
    self.active_writers = 0
    self.readers = 0

  def acquire_read(self):
    with self._condition:
      while self.active_writers > 0 or self.writer_active:
        self._condition.wait()
      self.readers += 1

  def release_read(self):
    with self._condition:
      self.readers -= 1
      if self.readers == 0:
        self._condition.notify_all()

  def acquire_write(self):
    with self._condition:
      self.active_writers += 1
      while self.writer_active:
        self._condition.wait()
      self.writer_active = True

  def release_write(self):
    with self._condition:
      self.active_writers -= 1
      self.writer_active = False
      self._condition.notify_all()

def test_thread_safe_counter():
    counter = ThreadSafeCounter()
    NUM_THREADS = 1000
    INCREMENTS_PER_THREAD = 100

    def worker():
        for _ in range(INCREMENTS_PER_THREAD):
            counter.increment()

    threads = [Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads: t.start()
    for t in threads: t.join()

    expected = NUM_THREADS * INCREMENTS_PER_THREAD
    actual = counter.value()
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("ThreadSafeCounter test passed.")

def test_bank_account_basic():
    a = BankAccount(100)
    b = BankAccount(50)

    assert a.transfer_to(b, 30) is True
    assert a._balance == 70
    assert b._balance == 80

    # insufficient funds
    assert a.transfer_to(b, 1000) is False
    assert a._balance == 70
    assert b._balance == 80

    print("BankAccount basic test passed.")

def test_bank_account_concurrent_transfers():
    a = BankAccount(1000)
    b = BankAccount(1000)
    c = BankAccount(1000)
    accounts = [a, b, c]

    total_before = sum(acc._balance for acc in accounts)

    def worker():
        # do lots of random transfers between accounts
        import random
        for _ in range(1000):
            src, dst = random.sample(accounts, 2)
            amount = random.randint(1, 10)
            src.transfer_to(dst, amount)

    threads = [Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    total_after = sum(acc._balance for acc in accounts)
    assert total_before == total_after, "Total balance should be conserved"
    assert all(acc._balance >= 0 for acc in accounts), "No negative balances allowed"

    print("BankAccount concurrent test passed.")

def test_bounded_blocking_queue_basic():
    q = BoundedBlockingQueue(2)
    q.put(1)
    q.put(2)
    # Should block if we try to overfill; we test non-blocking part first
    assert q.get() == 1
    assert q.get() == 2
    print("BoundedBlockingQueue basic FIFO test passed.")

def test_bounded_blocking_queue_producer_consumer():
    q = BoundedBlockingQueue(10)

    NUM_ITEMS = 10000
    NUM_PRODUCERS = 50
    NUM_CONSUMERS = 50

    produced_items = list(range(NUM_ITEMS))
    consumed = []

    producers = []
    def producer(items):
        for x in items:
            q.put(x)

    # split produced_items into chunks
    chunk_size = NUM_ITEMS // NUM_PRODUCERS
    for i in range(NUM_PRODUCERS):
        chunk = produced_items[i*chunk_size:(i+1)*chunk_size]
        t = Thread(target=producer, args=(chunk,))
        producers.append(t)

    consumers = []
    def consumer():
        for _ in range(NUM_ITEMS // NUM_CONSUMERS):
            x = q.get()
            consumed.append(x)

    for _ in range(NUM_CONSUMERS):
        consumers.append(Thread(target=consumer))

    for t in producers + consumers: t.start()
    for t in producers + consumers: t.join()

    assert Counter(consumed) == Counter(produced_items)
    print("BoundedBlockingQueue producer/consumer test passed.")

def test_future_result():
    fut = Future()

    def producer():
        time.sleep(0.2)
        fut.set_result("done")

    t = Thread(target=producer)
    t.start()

    val = fut.result(timeout=1.0)
    t.join()

    assert val == "done"
    print("Future result test passed.")

def test_future_timeout():
    fut = Future()
    start = time.time()
    try:
        fut.result(timeout=0.2)
        assert False, "Expected timeout behavior (e.g., raise or custom handling)"
    except TimeoutError:
        elapsed = time.time() - start
        assert elapsed >= 0.2
        print("Future timeout test passed.")

def test_connection_pool_limits_concurrency():
    max_conn = 3
    pool = ConnectionPool(max_conn)

    current_active = 0
    max_seen = 0
    lock = Lock()

    def worker():
        nonlocal current_active, max_seen
        for _ in range(20):
            with pool.acquire() as conn_id:
                with lock:
                    current_active += 1
                    max_seen = max(max_seen, current_active)
                # simulate work
                time.sleep(0.01)
                with lock:
                    current_active -= 1

    threads = [Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert max_seen <= max_conn, f"max_seen={max_seen} > {max_conn}"
    print("ConnectionPool concurrency limit test passed.")

def test_cancellation_token():
    token = CancellationToken()
    stopped = []

    def worker():
        while not token.is_cancelled():
            # do some fake work
            time.sleep(0.01)
        stopped.append(True)

    t = Thread(target=worker)
    t.start()

    time.sleep(0.05)
    token.cancel()

    t.join(timeout=1.0)
    assert stopped == [True], "Worker should have stopped after cancellation"
    print("CancellationToken test passed.")

def test_read_write_lock():
    rwlock = ReadWriteLock()
    shared_data = [0]
    log = []
    log_lock = Lock()
    
    def reader(id):
        rwlock.acquire_read()
        with log_lock:
            log.append(f"R{id}_start")
        time.sleep(0.02)
        value = shared_data[0]
        with log_lock:
            log.append(f"R{id}_end")
        rwlock.release_read()
        return value
    
    def writer(id, value):
        rwlock.acquire_write()
        with log_lock:
            log.append(f"W{id}_start")
        time.sleep(0.02)
        shared_data[0] = value
        with log_lock:
            log.append(f"W{id}_end")
        rwlock.release_write()
    
    # Test 1: Multiple readers can proceed concurrently
    threads = [Thread(target=reader, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Check that readers overlapped (at least partially)
    # If they didn't overlap at all, it would take 0.06s minimum
    
    # Test 2: Writer excludes everyone
    log.clear()
    writer_thread = Thread(target=writer, args=(0, 42))
    reader_threads = [Thread(target=reader, args=(i,)) for i in range(2)]
    
    writer_thread.start()
    time.sleep(0.005)  # Let writer start
    for r in reader_threads:
        r.start()
    
    writer_thread.join()
    for r in reader_threads:
        r.join()
    
    # Writer should complete before readers start
    w_end_idx = log.index("W0_end")
    r_start_indices = [log.index(f"R{i}_start") for i in range(2)]
    for r_idx in r_start_indices:
        assert r_idx > w_end_idx, "Reader started before writer finished"
    
    print("ReadWriteLock test passed.")

if __name__ == "__main__":
    test_thread_safe_counter()
    test_bank_account_basic()
    test_bank_account_concurrent_transfers()
    test_bounded_blocking_queue_basic()
    test_bounded_blocking_queue_producer_consumer()
    test_future_result()
    test_future_timeout()
    test_connection_pool_limits_concurrency()
    test_cancellation_token()
    test_read_write_lock()
