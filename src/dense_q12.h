/* =====================================================================
   Proyecto Hardware 2026 — Capa neuronal fully-connected en Q12 con AAPCS
   Cabecera con constantes, helpers y prototipos.

   - Definimos el tamaño de la capa: INPUT_SIZE entradas y OUTPUT_SIZE salidas.
   - Todos los valores son int16_t en formato Q12:
       valor_real = valor_entero / 4096.0
       1.0        = 4096
       0.5        = 2048
      -1.0        = -4096
   - Una neurona calcula:
       y = clamp( bias + sum_i input[i] * weight[i] )
   - Como input y weight están en Q12:
       Q12 * Q12 = Q24
       acumulamos en int32_t y al final desplazamos >> 12 para volver a Q12.
   - Funciones C de referencia:
       * neuron_q12_C()       calcula una neurona.
       * dense_layer_q12_C()  calcula una capa completa.
   - Funciones que deben implementar en ensamblador ARM/Thumb.
   - Función de verificación dense_q12_verificar() que compara resultados.

   Todas las funciones deben cumplir el ABI AAPCS (ARM Architecture
   Procedure Call Standard): paso de parámetros en r0-r3, resto en pila.
   =====================================================================
   v0.9 - Enrique Torres
*/

#ifndef DENSE_Q12_H
#define DENSE_Q12_H

#include <stdint.h>

/* Compatibilidad: Keil/ARM suele definir __weak; GCC/ARMClang aceptan attribute weak. */
#ifndef __weak
    #if defined(__GNUC__)
        #define __weak __attribute__((weak))
    #endif
#endif

/* === Flag para exigir ASM (0 = usar stubs; 1 = exigir impl. ASM) === */
#ifndef ENABLE_ASM_IMPL
    #define ENABLE_ASM_IMPL 0
#endif

/* ============================================================
   Constantes de tamaño (puedes cambiarlas para otras pruebas)
   ============================================================ */
#define INPUT_SIZE   8
#define OUTPUT_SIZE  5

/* ============================================================
   Constantes Q12
   ============================================================ */
#define Q_SHIFT  12
#define Q_ONE    (1 << Q_SHIFT)
#define Q_HALF   (1 << (Q_SHIFT - 1))

/* Activación tipo ReLU limitada: salida en rango [0.0, 1.0] */
#define CLAMP_MIN_Q12  0
#define CLAMP_MAX_Q12  Q_ONE

/* ============================================================
   Helpers en cabecera (internal linkage por 'static')
   ============================================================ */

/**
 * @brief   Saturación/clamp a rango int16_t configurable.
 *
 * Recibe un entero de 32 bits y lo limita al rango [lo, hi].
 * Los límites también están expresados en Q12.
 *
 * Ejemplo para ReLU limitada:
 *   lo = 0       -> 0.0
 *   hi = Q_ONE   -> 1.0
 *
 * @param   x   Valor de entrada en int32_t, ya convertido a Q12.
 * @param   lo  Límite inferior en Q12.
 * @param   hi  Límite superior en Q12.
 * @return  Valor saturado en rango [lo, hi] como int16_t.
 */
static inline int16_t clamp_i16_q12(int32_t x, int16_t lo, int16_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return (int16_t)x;
}

/**
 * @brief   Compara dos vectores int16_t elemento a elemento.
 *
 * @param[in] a  Primer vector.
 * @param[in] b  Segundo vector.
 * @param[in] n  Número de elementos.
 * @return  1 si los dos vectores son idénticos, 0 si hay alguna diferencia.
 */
static uint8_t vectores_iguales_i16(const int16_t *a, const int16_t *b, int n) {
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

/* ============================================================
   Firmas públicas (C y ASM)
   ============================================================ */

/**
 * @brief   Calcula una neurona fully-connected en Q12.
 *
 * @param   input       Vector de entrada, int16_t Q12, longitud n.
 * @param   weights     Pesos de ESTA neurona, int16_t Q12, longitud n.
 * @param   n           Número de entradas/pesos.
 * @param   bias_q12    Bias de la neurona, int16_t Q12.
 * @param   clamp_min   Límite inferior de salida, int16_t Q12.
 * @param   clamp_max   Límite superior de salida, int16_t Q12.
 * @return  Salida de la neurona, int16_t Q12, saturada a [clamp_min, clamp_max].
 *
 * @details Operación matemática:
 *          y = clamp(bias + sum_i input[i] * weights[i])
 *
 *          input[i] y weights[i] están en Q12.
 *          input[i] * weights[i] queda en Q24.
 *          El acumulador interno es int32_t Q24.
 *          Al final se hace acc >> 12 para volver a Q12.
 *
 * @note AAPCS: firma fuerza paso de >4 parámetros:
 *       r0 = input
 *       r1 = weights
 *       r2 = n
 *       r3 = bias_q12
 *       [sp + 0] = clamp_min
 *       [sp + 4] = clamp_max
 */
int16_t neuron_q12_C(
    const int16_t *input,
    const int16_t *weights,
    uint16_t n,
    int16_t bias_q12,
    int16_t clamp_min,
    int16_t clamp_max);

/**
 * @brief   Capa fully-connected completa en Q12.
 *
 * @param   input        Vector de entrada común, int16_t Q12, longitud input_size.
 * @param   weights      Matriz de pesos en orden fila mayor:
 *                       weights[o * input_size + i] es el peso de la salida o
 *                       conectado a la entrada i.
 * @param   bias         Vector de bias, uno por neurona de salida, int16_t Q12.
 * @param   output       Vector de salida, int16_t Q12, longitud output_size.
 * @param   input_size   Número de entradas.
 * @param   output_size  Número de neuronas/salidas.
 * @param   clamp_min    Límite inferior de activación, int16_t Q12.
 * @param   clamp_max    Límite superior de activación, int16_t Q12.
 * @return  Checksum de la salida para facilitar verificación/medida.
 *
 * @details Llama a neuron_q12_C() una vez por cada neurona de salida.
 *
 * @note AAPCS: firma fuerza paso de >4 parámetros:
 *       r0 = input
 *       r1 = weights
 *       r2 = bias
 *       r3 = output
 *       [sp + 0]  = input_size
 *       [sp + 4]  = output_size
 *       [sp + 8]  = clamp_min
 *       [sp + 12] = clamp_max
 */
uint32_t dense_layer_q12_C(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max);

/* === Implementaciones que harán en ensamblador (extern) === */

/* Neurona en ARM/Thumb: primera parte de la práctica. */
int16_t neuron_q12_ARM(
    const int16_t *input,
    const int16_t *weights,
    uint16_t n,
    int16_t bias_q12,
    int16_t clamp_min,
    int16_t clamp_max);

int16_t neuron_q12_THB(
    const int16_t *input,
    const int16_t *weights,
    uint16_t n,
    int16_t bias_q12,
    int16_t clamp_min,
    int16_t clamp_max);

/* Capa en C que llama a la neurona ARM: útil para verificar la función interna sola. */
uint32_t dense_layer_q12_C_ARM(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max);

/* Capa en C que llama a la neurona Thumb: útil para verificar la función interna sola. */
uint32_t dense_layer_q12_C_THB(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max);

/* Capa en ARM que llama a neuron_q12_C(): versión mixta externa ASM + interna C. */
uint32_t dense_layer_q12_ARM_C(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max);

/* Capa completa en ARM/Thumb: externa e interna en ensamblador, o versión fusionada. */
uint32_t dense_layer_q12_ARM(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max);

uint32_t dense_layer_q12_THB(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max);

/* Verificador: ejecuta C, C_ARM, C_THB, ARM_C, ARM y THB; compara checksum y buffers. */
uint8_t dense_q12_verificar(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *resultado,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max);

#endif /* DENSE_Q12_H */
