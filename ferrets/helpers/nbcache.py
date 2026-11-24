"""Numba cache guard. numba invalidates a cached kernel when its own file
changes, not when an inline helper or a constant it compiles in from another
file does. ``check`` hashes the package files such code comes from and, when
the hash moved, deletes the package's numba cache files so that every kernel
recompiles."""
import hashlib
import os
import threading
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
HASHED = ("kernels/*.py", "helpers/split.py", "constants.py",
          "ablation/kernels.py")


def digest(root=PKG, patterns=HASHED):
    """sha256 over the matched files' relative paths and bytes."""
    h = hashlib.sha256()
    for pat in patterns:
        for p in sorted(Path(root).glob(pat)):
            h.update(p.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def check(root=PKG, patterns=HASHED):
    """Delete root's numba cache files (``__pycache__/*.nbi``, ``*.nbc``)
    when the digest differs from the stamp; True when it did. The stamp is
    written last, atomically, and only when every file went, so a process
    that dies mid-way, or a file another process holds open, makes the next
    import repeat the deletion. Deleting is idempotent, so racing processes
    do no harm, and an unwritable directory leaves the import working."""
    root = Path(root)
    stamp = root / "__pycache__" / "nbcache.stamp"
    new = digest(root, patterns)
    try:
        if stamp.read_text() == new:
            return False
    except OSError:
        pass
    held = 0
    for pat in ("__pycache__/*.nbi", "__pycache__/*.nbc"):
        for f in root.rglob(pat):
            try:
                f.unlink()
            except OSError:
                held += 1
    if held:
        return True
    tmp = stamp.with_name(
        f"nbcache.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(new)
        os.replace(tmp, stamp)
    except OSError:          # another process or thread wrote it just now
        try:
            tmp.unlink()
        except OSError:
            pass
    return True
