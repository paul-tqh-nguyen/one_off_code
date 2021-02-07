
import ctypes

lib = ctypes.CDLL('../build/tibs-main/libtibs-shared.so')

print(f"dir(lib) {repr(dir(lib))}")

lib.runAllPasses.restype = None
lib.runAllPasses.argtypes = []

result = lib.runAllPasses()

print('Done.')
