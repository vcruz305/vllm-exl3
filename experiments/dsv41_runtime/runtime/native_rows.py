"""C ABI for the CPU-only reader. Caller owns graph/stream lifetime fences."""
import ctypes as C

P, U = C.c_void_p, C.c_uint64


class Work(C.Structure):
    _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]


def load(path):
    lib = C.CDLL(str(path))
    lib.sg_create.argtypes = [C.c_int, C.c_int, U, U, U, U, U, U, U]
    lib.sg_create.restype = P
    lib.sg_error.restype = C.c_char_p
    lib.sg_lookup.argtypes = [P]
    lib.sg_lookup.restype = None
    lib.sg_lookup_test.argtypes = [P]
    lib.sg_lookup_test.restype = C.c_int
    lib.sg_stats.argtypes = [P, C.POINTER(U)]
    lib.sg_stats.restype = None
    lib.sg_destroy.argtypes = [P]
    lib.sg_destroy.restype = None
    return lib


def create(lib, table, lo, hi, capacity=131072, workers=4):
    if table.dim != 256 or table.sb != 8 or table.w_shape[0] != table.s_shape[0]:
        raise ValueError('Original FP8/UE8M0 row geometry required')
    handle = lib.sg_create(table.w_fd, table.s_fd, table.w_off, table.s_off,
                           table.w_shape[0], lo, hi, capacity, workers)
    if not handle:
        raise RuntimeError(lib.sg_error().decode())
    return handle


def stats(lib, handle):
    values = (U * 8)()
    lib.sg_stats(handle, values)
    return dict(zip(('hits', 'misses', 'reads', 'evictions', 'lookups', 'failure',
                     'capacity', 'workers'), values))
