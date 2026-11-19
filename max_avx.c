#include <immintrin.h> // SIMD AVX
#include <stddef.h>

// Encuentra el valor maximo en un arreglo masivo de flotantes
float find_max_avx(const float* arr, size_t size) {
    if (size == 0) return 0.0f;
    
    size_t i = 0;
    // Inicializamos un registro de 256 bits copiando el primer elemento 8 veces
    __m256 max_vec = _mm256_set1_ps(arr[0]);

    // Procesamos de 8 en 8 (256 bits)
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(&arr[i]);           // Cargar 8 elementos
        max_vec = _mm256_max_ps(max_vec, v);           // Comparar y guardar los maximos
    }

    // "Reducción horizontal": Pasamos los 8 maximos finales a un arreglo temporal
    float temp[8];
    _mm256_storeu_ps(temp, max_vec);
    
    // Encontramos el máximo real entre esos 8 finalistas
    float max_val = temp[0];
    for (int j = 1; j < 8; ++j) {
        if (temp[j] > max_val) {
            max_val = temp[j];
        }
    }

    // Bucle residual por si el tamaño del arreglo no es múltiplo de 8
    for (; i < size; ++i) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }

    return max_val;
}