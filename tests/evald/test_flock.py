"""模块: kernel_agent_loop.evald.flockutil
功能: flock 保证同一时刻只有一个评测持有 GPU 锁。
"""

import threading
import time
from pathlib import Path

from kernel_agent_loop.evald.flockutil import exclusive_lock


# 用例: 第二个加锁者在第一个释放前拿不到锁
# 参数: tmp_path 是临时目录
def test_exclusive_lock_serializes_callers(tmp_path: Path):
    lock_path = tmp_path / "gpu.lock"
    events = []

    def holder():
        with exclusive_lock(lock_path):
            events.append("hold-enter")
            time.sleep(0.2)
            events.append("hold-exit")

    def waiter():
        time.sleep(0.05)
        with exclusive_lock(lock_path):
            events.append("wait-enter")

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert events == ["hold-enter", "hold-exit", "wait-enter"]
