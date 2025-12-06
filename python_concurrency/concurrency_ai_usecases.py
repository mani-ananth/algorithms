import threading
import time
from collections import OrderedDict, Counter
from contextlib import contextmanager


# =========================================
# 1. BoundedBatchQueue - inference requests
# =========================================

class BoundedBatchQueue:
    """
    Bounded queue representing an inference request queue.
    Producers: network threads adding requests.
    Consumers: batching / worker threads.
    """
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._queue = []
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            while len(self._queue) >= self._capacity:
                self._cond.wait()
            self._queue.append(item)
            self._cond.notify()

    def get(self):
        with self._cond:
            while not self._queue:
                self._cond.wait()
            item = self._queue.pop(0)
            self._cond.notify()
            return item


# =========================================
# 2. GpuInferenceLimiter - Semaphore
# =========================================

class GpuInferenceLimiter:
    """
    Limits the number of concurrent GPU inference slots.
    Think: no more than K requests simultaneously running on a GPU.
    """
    def __init__(self, max_concurrent: int):
        self._sem = threading.BoundedSemaphore(max_concurrent)

    @contextmanager
    def acquire_slot(self):
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


# =========================================
# 3. CancellationToken - model step cancellation
# =========================================

class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait_cancelled(self, timeout=None) -> bool:
        return self._event.wait(timeout)


# =========================================
# 4. TokenizerLRUCache - LRU for tokenization
# =========================================

class TokenizerLRUCache:
    """
    Thread-safe LRU cache for tokenizer results.
    Example: cache mapping text -> token_ids to avoid recomputing.
    """
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._cache = OrderedDict()

    def get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            # move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

    def put(self, key, value):
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self._capacity:
                # pop least recently used
                self._cache.popitem(last=False)
            self._cache[key] = value


# ===========
# TESTS
# ===========

def test_bounded_batch_queue_inference_pattern():
    q = BoundedBatchQueue(5)

    NUM_REQUESTS = 50
    NUM_PRODUCERS = 3
    NUM_CONSUMERS = 2

    produced = list(range(NUM_REQUESTS))
    consumed = []
    stop_sentinel = object()

    def producer(items):
        for req in items:
            q.put(req)

    # split produced across producers (last producer takes remainder)
    chunk_size = NUM_REQUESTS // NUM_PRODUCERS
    producers = []
    for i in range(NUM_PRODUCERS):
        start = i * chunk_size
        end = NUM_REQUESTS if i == NUM_PRODUCERS - 1 else (i + 1) * chunk_size
        chunk = produced[start:end]
        t = threading.Thread(target=producer, args=(chunk,))
        producers.append(t)

    def consumer():
        while True:
            item = q.get()
            if item is stop_sentinel:
                break
            consumed.append(item)
            # simulate forming batches, preprocessing, etc.
            time.sleep(0.001)

    consumers = [threading.Thread(target=consumer) for _ in range(NUM_CONSUMERS)]

    for t in consumers:
        t.start()
    for t in producers:
        t.start()
    for t in producers:
        t.join()

    # signal shutdown to consumers
    for _ in range(NUM_CONSUMERS):
        q.put(stop_sentinel)

    for t in consumers:
        t.join()

    assert Counter(consumed) == Counter(produced)
    print("BoundedBatchQueue inference pattern test passed.")


def test_gpu_inference_limiter():
    max_concurrent = 4
    limiter = GpuInferenceLimiter(max_concurrent)

    current_active = 0
    max_seen = 0
    lock = threading.Lock()

    def worker():
        nonlocal current_active, max_seen
        for _ in range(20):
            with limiter.acquire_slot():
                with lock:
                    current_active += 1
                    max_seen = max(max_seen, current_active)
                # simulate GPU work
                time.sleep(0.005)
                with lock:
                    current_active -= 1

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_seen <= max_concurrent, f"max_seen={max_seen} > {max_concurrent}"
    print("GpuInferenceLimiter concurrency limit test passed.")


def test_cancellation_token_for_model_step():
    token = CancellationToken()
    steps_executed = 0

    def model_worker():
        nonlocal steps_executed
        while not token.is_cancelled():
            # pretend to do a generation / decoding step
            time.sleep(0.01)
            steps_executed += 1

    t = threading.Thread(target=model_worker)
    t.start()

    time.sleep(0.05)
    token.cancel()
    t.join(timeout=1.0)

    assert steps_executed > 0, "Worker should have done some steps"
    print("CancellationToken model step test passed.")


def test_data_parallel_barrier_sync():
    """
    N workers:
      - Step 1: preprocess batch
      - Barrier
      - Step 2: tokenize batch
      - Barrier

    Verify all preprocess events happen before any tokenize events.
    """
    n = 6
    barrier = threading.Barrier(n)
    log = []
    log_lock = threading.Lock()

    def preprocess(worker_id, batch_id):
        with log_lock:
            log.append(("preprocess", worker_id, batch_id))

    def tokenize(worker_id, batch_id):
        with log_lock:
            log.append(("tokenize", worker_id, batch_id))

    def worker(worker_id):
        # pretend each worker handles one batch id == worker_id
        preprocess(worker_id, worker_id)
        barrier.wait()
        tokenize(worker_id, worker_id)
        barrier.wait()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Ensure all preprocess events appear before any tokenize
    first_tokenize_idx = next(
        idx for idx, (kind, _, _) in enumerate(log) if kind == "tokenize"
    )
    assert all(kind == "preprocess" for kind, _, _ in log[:first_tokenize_idx])
    print("Data-parallel Barrier sync test passed.")


def test_tokenizer_lru_cache():
    cache = TokenizerLRUCache(capacity=2)
    cache.put("hello", [1, 2, 3])
    cache.put("world", [4, 5])

    assert cache.get("hello") == [1, 2, 3]
    # Access "hello" to make it most recently used
    cache.put("openai", [6])

    # capacity is 2, so "world" should be evicted
    assert cache.get("world") is None
    assert cache.get("hello") == [1, 2, 3]
    assert cache.get("openai") == [6]

    # Concurrency smoke test
    def worker():
        for i in range(100):
            key = f"k{i % 3}"
            cache.put(key, [i])
            cache.get(key)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("TokenizerLRUCache tests passed.")


if __name__ == "__main__":
    test_bounded_batch_queue_inference_pattern()
    test_gpu_inference_limiter()
    test_cancellation_token_for_model_step()
    test_data_parallel_barrier_sync()
    test_tokenizer_lru_cache()
