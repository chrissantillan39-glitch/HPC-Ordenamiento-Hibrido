import ctypes
import numpy as np
import time

# 1. Cargar la biblioteca dinámica compilada en Ubuntu
lib = ctypes.CDLL('./libmax_avx.so')

# 2. Configurar exactamente los tipos de entrada y salida para evitar Segmentation Faults
lib.find_max_avx.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_size_t
]
lib.find_max_avx.restype = ctypes.c_float

# 3. Generar los datos de prueba
size = 50_000_000
print(f"Generando un arreglo de {size} flotantes...")
arr = np.random.rand(size).astype(np.float32)

# 4. Ejecutar tu código en C (Intrínsecos)
start_c = time.perf_counter()
max_c = lib.find_max_avx(arr, size)
end_c = time.perf_counter()

# 5. Ejecutar la función nativa de NumPy para comparar
start_np = time.perf_counter()
max_np = np.max(arr)
end_np = time.perf_counter()

print("\n--- RESULTADOS ---")
print(f"Valor máximo encontrado (C Intrinsics): {max_c:.6f} | Tiempo: {end_c - start_c:.5f} segundos")
print(f"Valor máximo encontrado (NumPy):        {max_np:.6f} | Tiempo: {end_np - start_np:.5f} segundos")