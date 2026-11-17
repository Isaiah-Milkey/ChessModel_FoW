"""Keep long jobs from overloading this PC (it blue-screened with two heavy jobs at once).

    from guard import Guard
    g = Guard("rl_loop")          # waits until no other guarded job is running, then holds the lock
    workers = g.workers(12, gb_each=0.7)
    ...
    g.wait_ok()                   # call between phases and every few batches: pauses while overloaded

- One heavy job at a time: every guarded job holds an exclusive lock on runs/.heavy_job.lock
  for its whole life. A second job waits until the first ends. The operating system releases the
  lock when the process ends, even after a crash, so it never goes stale.
- wait_ok() pauses (checking every 15 s) while free memory is below min_free_gb, memory use is
  above max_ram_pct, or the GPU is hotter than max_gpu_c.
- workers() lowers a requested worker count to what free memory and the CPU allow.
Settings can be changed with the environment variables GUARD_MIN_FREE_GB, GUARD_MAX_RAM_PCT,
GUARD_MAX_GPU_C; GUARD_OFF=1 turns the guard off (no lock, no pauses).
"""
import os
import subprocess
import sys
import time

try:
    import psutil                     # pip install psutil; without it only the lock and CPU cap apply
except ImportError:
    psutil = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "runs", ".heavy_job.lock")


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(int(v) for v in out.split())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Guard:
    def __init__(self, name, min_free_gb=None, max_ram_pct=None, max_gpu_c=None, lock=True):
        env = os.environ.get
        self.off = env("GUARD_OFF") == "1"
        self.name = name
        self.min_free_gb = float(min_free_gb or env("GUARD_MIN_FREE_GB", 3.0))
        self.max_ram_pct = float(max_ram_pct or env("GUARD_MAX_RAM_PCT", 85))
        self.max_gpu_c = float(max_gpu_c or env("GUARD_MAX_GPU_C", 83))
        self._temp, self._temp_at = None, 0.0
        self._fh = None
        if lock and not self.off:
            self._acquire()

    def _acquire(self):
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        self._fh = open(LOCK, "a+")
        told = False
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if not told:
                    print(f"[guard] {self.name}: another heavy job is running; waiting for it to finish", flush=True)
                    told = True
                time.sleep(30)
        if told:
            print(f"[guard] {self.name}: starting", flush=True)

    def status(self):
        """(problem or None, short description of the machine's state)."""
        if psutil is None:
            return None, "psutil not installed: memory not checked"
        vm = psutil.virtual_memory()
        free = vm.available / 2 ** 30
        if time.time() - self._temp_at > 20:
            self._temp, self._temp_at = gpu_temp(), time.time()
        desc = f"{free:.1f} GB free, memory {vm.percent:.0f}% used, GPU {self._temp} C"
        if free < self.min_free_gb:
            return f"free memory below {self.min_free_gb} GB", desc
        if vm.percent > self.max_ram_pct:
            return f"memory use above {self.max_ram_pct:.0f}%", desc
        if self._temp is not None and self._temp > self.max_gpu_c:
            return f"GPU above {self.max_gpu_c:.0f} C", desc
        return None, desc

    def wait_ok(self):
        """Return at once if the machine is fine; otherwise pause until it is."""
        if self.off:
            return
        problem, desc = self.status()
        if problem is None:
            return
        t = time.time()
        print(f"[guard] {self.name}: pausing, {problem} ({desc})", flush=True)
        while problem is not None:
            time.sleep(15)
            self._temp_at = 0.0
            problem, desc = self.status()
        print(f"[guard] {self.name}: resuming after {time.time() - t:.0f}s ({desc})", flush=True)

    def workers(self, requested, gb_each):
        """Worker processes that fit: leave min_free_gb free and 2 CPU threads for the system."""
        if self.off:
            return requested
        cap = min(requested, (os.cpu_count() or 4) - 2)
        if psutil is None:
            return max(1, cap)
        fits =lambda f: max(1, min(cap, int((f - self.min_free_gb) // gb_each)))
        free = psutil.virtual_memory().available / 2 ** 30
        for _ in range(6):                    # memory from a job that just ended takes a little while to return
            if fits(free) >= cap:
                break
            time.sleep(10)
            free = psutil.virtual_memory().available / 2 ** 30
        n = fits(free)
        if n < requested:
            print(f"[guard] {self.name}: using {n} workers instead of {requested} ({free:.1f} GB free)", flush=True)
        return n
