"""Linux process ownership for simulation and solver subprocesses."""

import ctypes
import errno
import os
import platform
import select
import threading


def _exit_when_owner_dies(owner_fd):
    select.select([owner_fd], [], [])
    os._exit(1)


def exit_with_owner(owner_pid):
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Process ownership requires Linux x86_64 or aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    # pidfd_open is syscall 434 on both supported architectures. Calling the
    # kernel directly also works with the older glibc bundled with Isaac.
    libc.syscall.restype = ctypes.c_long
    owner_fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(owner_pid), ctypes.c_uint(0))
    if owner_fd < 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            os._exit(1)
        raise OSError(error, os.strerror(error))
    threading.Thread(target=_exit_when_owner_dies, args=(owner_fd,), daemon=True).start()
