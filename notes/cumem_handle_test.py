# Minimal reproduction: cuMemCreate on GPU 0 with each shareable-handle type.
import ctypes, sys

cuda = ctypes.CDLL("libcuda.so.1")

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

class AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]

class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", CUmemLocation), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", AllocFlags)]

def name(rc):
    s = ctypes.c_char_p()
    cuda.cuGetErrorName(rc, ctypes.byref(s))
    return s.value.decode() if s.value else str(rc)

def check(what, rc):
    print(f"{what:<45} -> {rc} {name(rc)}")
    return rc

check("cuInit(0)", cuda.cuInit(0))
dev = ctypes.c_int()
check("cuDeviceGet(0)", cuda.cuDeviceGet(ctypes.byref(dev), 0))
ctx = ctypes.c_void_p()
check("cuDevicePrimaryCtxRetain", cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev))
check("cuCtxSetCurrent", cuda.cuCtxSetCurrent(ctx))

# CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128
fab = ctypes.c_int()
check("cuDeviceGetAttribute(FABRIC_SUPPORTED)", cuda.cuDeviceGetAttribute(ctypes.byref(fab), 128, dev))
print(f"{'  fabric handles supported by device':<45} -> {fab.value}")

HANDLE_TYPES = {"POSIX_FILE_DESCRIPTOR": 0x1, "FABRIC": 0x8}
for label, htype in HANDLE_TYPES.items():
    prop = CUmemAllocationProp()
    prop.type = 1                  # CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = htype
    prop.location.type = 1         # CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev.value
    gran = ctypes.c_size_t()
    cuda.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0)
    handle = ctypes.c_ulonglong()
    rc = check(f"cuMemCreate({label}, {gran.value >> 20} MiB)",
               cuda.cuMemCreate(ctypes.byref(handle), gran, ctypes.byref(prop), ctypes.c_ulonglong(0)))
    if rc == 0:
        cuda.cuMemRelease(handle)
cuda.cuDevicePrimaryCtxRelease(dev)
