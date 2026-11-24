"""Spin barriers for kernels whose threads work as one team in a single
parallel region (the streaming head's chunk pass).

A team kernel runs ``prange(T)`` with T the numba thread count, one
iteration per thread, all of them live at once: numba's omp and workqueue
layers split prange(T) into T chunks run by T threads at the same time (the
caller runs a team of one under any other layer, which gives the same
floats). The threads meet at barriers: each thread has a counter on its own
cache line (flags[tid * 8]), stored with release semantics and read with
acquire semantics through llvmlite atomics, so the writes a thread made
before a barrier are visible to every thread after it."""

import platform

from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

_X86 = platform.machine().lower() in ("amd64", "x86_64", "i386", "i686",
                                      "x86")


@intrinsic
def _load_acquire(typingctx, arr, i):
    """arr[i] (int64), an atomic load with acquire semantics."""
    def codegen(context, builder, sig, args):
        aryty = sig.args[0]
        ary = context.make_array(aryty)(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, aryty, ary,
                                       [args[1]])
        return builder.load_atomic(ptr, ordering="acquire", align=8)
    return types.int64(arr, i), codegen


@intrinsic
def _store_release(typingctx, arr, i, v):
    """arr[i] = v (int64), an atomic store with release semantics."""
    def codegen(context, builder, sig, args):
        aryty = sig.args[0]
        ary = context.make_array(aryty)(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, aryty, ary,
                                       [args[1]])
        builder.store_atomic(args[2], ptr, ordering="release", align=8)
        return context.get_dummy_value()
    return types.void(arr, i, v), codegen


@intrinsic
def _pause(typingctx):
    """The x86 spin-wait hint (nothing elsewhere)."""
    def codegen(context, builder, sig, args):
        if _X86:
            fnty = ir.FunctionType(ir.VoidType(), [])
            fn = cgutils.get_or_insert_function(builder.module, fnty,
                                                "llvm.x86.sse2.pause")
            builder.call(fn, [])
        return context.get_dummy_value()
    return types.void(), codegen


@njit(inline="always")
def _barrier(flags, tid, T, gen):
    """Every thread's counter reaches gen. Each thread passes the same
    barriers in the same order, so the counters only grow within a call;
    the caller zeroes flags before each call."""
    _store_release(flags, tid * 8, gen)
    for o in range(T):
        while _load_acquire(flags, o * 8) < gen:
            _pause()
