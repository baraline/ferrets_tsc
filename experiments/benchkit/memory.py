"""Process memory: settled RSS, a wall-clock RSS sampler, the lifetime peak.

MB means 1e6 bytes everywhere in the results (GiB = MB / 1073.74).
"""

import statistics
import threading

import psutil

MB = 1e6


def rss_mb():
    return psutil.Process().memory_info().rss / MB


def peak_wset_mb():
    """Process-lifetime peak resident memory: peak_wset on Windows,
    ru_maxrss (kB) elsewhere (the WSL2 child)."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / MB
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / MB


class RssSampler:
    """Reads the process RSS every ``interval_s`` on a daemon thread.

    ``stop()`` returns the peak and the median of the samples (one sample
    is taken at start and one at stop, so a window shorter than the
    interval still has two)."""

    def __init__(self, interval_s=0.02):
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._samples = [self._proc.memory_info().rss]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self._samples.append(self._proc.memory_info().rss)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._samples.append(self._proc.memory_info().rss)
        s = self._samples
        return dict(peak_rss_mb=max(s) / MB,
                    median_rss_mb=statistics.median(s) / MB,
                    n_samples=len(s))
