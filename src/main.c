// Proyecto Hardware 2026 — Capa neuronal fully-connected en Q12 con AAPCS
// Referencia C + verificación + main de prueba.
// ARM7TDMI (ARMv4T) / Keil µVision.
// v0.9 - Enrique Torres && chatGPT

// ======================================================================
// INCLUDES Y DEFINICIONES
// ======================================================================
#include "dense_q12.h"
#include <stdint.h>

// ======================================================================
// CASOS DE PRUEBA PARA func_verificar
// ======================================================================

// Tamaño máximo de salida entre todos los casos (búferes de func_verificar).
#define TEST_MAX_OUTPUT  8

// Valor con el que rellenamos los buffers para ver si alguien escribe de más
#define RELLENO_SALIDA  0x5A5A

// Resultado final de los tests, visible en el Watch:
//   0xFF = aún no ha terminado, 1 = todo OK, 0 = algún fallo.
volatile uint8_t resultado_tests = 0xFF;

/* Caso 1: saturación superior. Todas las salidas superan clamp_max. */
#define T1_IN   4
#define T1_OUT  3
static const int16_t t1_input[T1_IN] __attribute__((aligned(8))) = {
     Q_ONE,  Q_ONE,  Q_ONE,  Q_ONE
};
static const int16_t t1_weights[T1_OUT * T1_IN] __attribute__((aligned(8))) = {
     Q_ONE,      Q_ONE,      Q_ONE,      Q_ONE,      //  4.00
     Q_ONE / 2,  Q_ONE / 2,  Q_ONE / 2,  Q_ONE / 2,  //  2.00
     2 * Q_ONE,  0,          0,          0           //  2.00
};
static const int16_t t1_bias[T1_OUT] __attribute__((aligned(8))) = {
     0,  Q_ONE / 2,  Q_ONE / 4
};

/* Caso 2: saturación inferior. Todas las salidas quedan por debajo de clamp_min. */
#define T2_IN   3
#define T2_OUT  4
static const int16_t t2_input[T2_IN] __attribute__((aligned(8))) = {
     Q_ONE / 2, -Q_ONE / 2,  Q_ONE / 4
};
static const int16_t t2_weights[T2_OUT * T2_IN] __attribute__((aligned(8))) = {
    -Q_ONE,      Q_ONE,      0,
    -Q_ONE / 2,  Q_ONE / 2, -Q_ONE / 2,
     0,          0,          0,
     Q_ONE / 4,  Q_ONE / 2, -Q_ONE
};
static const int16_t t2_bias[T2_OUT] __attribute__((aligned(8))) = {
     0, -Q_ONE / 8, -1, -Q_ONE / 4
};

/* Caso 3: clamp simétrico [-1.0, 1.0] con resultados negativos y valores
   impares para comprobar el truncamiento de ASR #12 (redondeo hacia -1 y no 0). */
#define T3_IN   6
#define T3_OUT  4
#define T3_CLAMP_MIN  (-Q_ONE)
#define T3_CLAMP_MAX  ( Q_ONE)
static const int16_t t3_input[T3_IN] __attribute__((aligned(8))) = {
     1, -3,  4095, -4095,  1234, -2345
};
static const int16_t t3_weights[T3_OUT * T3_IN] __attribute__((aligned(8))) = {
    -1,      0,      0,      0,      0,      0,      // acc = -1 -> ASR da -1 (no 0)
    -4096,   4096,  -1,     -1,      7,     -7,
     4095,   4095,   4095,   4095,   4095,   4095,
    -1000,   2000,  -3000,   1500,  -2500,   3500
};
static const int16_t t3_bias[T3_OUT] __attribute__((aligned(8))) = {
     0, -1,  Q_ONE / 2, -Q_ONE / 2
};

/* Caso 4: una sola entrada y una sola salida, rango de clamp que no incluye el 0. */
#define T4_IN   1
#define T4_OUT  1
#define T4_CLAMP_MIN  (Q_ONE / 4)
#define T4_CLAMP_MAX  (Q_ONE / 2)
static const int16_t t4_input[T4_IN] __attribute__((aligned(8))) = {
     3 * Q_ONE / 4
};
static const int16_t t4_weights[T4_OUT * T4_IN] __attribute__((aligned(8))) = {
     Q_ONE / 2
};
static const int16_t t4_bias[T4_OUT] __attribute__((aligned(8))) = {
     Q_ONE / 16
};
/* Caso 4b/4c: mismos datos que el caso 4, pero con clamp que satura. */
#define T4B_CLAMP_MIN  (Q_ONE / 2)
#define T4B_CLAMP_MAX  (Q_ONE)
#define T4C_CLAMP_MIN  0
#define T4C_CLAMP_MAX  (Q_ONE / 8)

/* Caso 5: input_size = 0. La salida es clamp(bias) */
#define T5_IN   0
#define T5_OUT  3
static const int16_t t5_bias[T5_OUT] __attribute__((aligned(8))) = {
    -Q_ONE / 2,  Q_ONE / 2,  2 * Q_ONE
};

/* Caso 6: capa más grande (16 x 8) */
#define T6_IN   16
#define T6_OUT  8
#define T6_CLAMP_MIN  (-Q_ONE / 2)
#define T6_CLAMP_MAX  ( Q_ONE / 2)
static const int16_t t6_input[T6_IN] __attribute__((aligned(8))) = {
    -2145,  1138, -2415,  -438,  2795,  3946,  3130,  -165,
    -4055, -2774, -2282,   610, -2490,  3271, -3908,  3936
};
static const int16_t t6_weights[T6_OUT * T6_IN] __attribute__((aligned(8))) = {
      526,  -326,  1206,    12,   800,   873,  1035, -1423,
      740, -1314,   345,   308,  1699,  -894,   483, -1846,

      973,   930,  1729,  1410, -1306,  1218,  2035, -1095,
     1437,  1974,  1213,    70,  1392,  1877, -1841,  -209,

     -992, -1648, -1208,  1469,  1802,  -882,  1901,  -173,
    -1049,  1694,  1797,     8,  1245,   472,  1875,  1244,

     1499,  -389,   363,   155, -1973, -1777,  -821,  1705,
     1962,   900,  -411,    82,  1929,  1774,  2024,  1509,

    -1773,   328,   546,  1701, -1243,  1543,    72,  1662,
     -549,  1439,  1877, -1522,   134,  2035,   984, -1257,

     -454,  1321,  -850,  1633, -1934, -1293, -2047,  1648,
    -1368,  -420,   -18, -1059, -1683,   151,   878,  -781,

    -2006,   519,  1982,  -944,  -222,  1826,  -974,   333,
    -1357,  1374,  -737,  -940,   458,  1327,  1150,  -832,

     -437,  1032, -1933, -1179,  1141, -1447, -1628,   794,
      847, -1837, -1839,  1851,   637,   439,  1639,   622
};
static const int16_t t6_bias[T6_OUT] __attribute__((aligned(8))) = {
    -1054,  2028,  1774,  -650,  -788,  -353,  1504,   707
};

/* Caso 7: output_size = 0. No se escribe nada y el checksum debe ser 0.
   Reutiliza los datos del caso 1. */
#define T7_OUT  0

/* Caso 8: valores extremos de int16 y clamp de rango completo.
   Esperado: salidas {32767, -32768, 32767, -32768, 32767}, checksum 0x5490633D. */
#define T8_IN   2
#define T8_OUT  5
#define T8_CLAMP_MIN  (-32768)
#define T8_CLAMP_MAX  ( 32767)
static const int16_t t8_input[T8_IN] __attribute__((aligned(8))) = {
    -32768, 32767
};
static const int16_t t8_weights[T8_OUT * T8_IN] __attribute__((aligned(8))) = {
    -32768,      0,
     32767,      0,
    -32768,  32767,
         0,      0,
         0,      0
};
static const int16_t t8_bias[T8_OUT] __attribute__((aligned(8))) = {
     0, 0, 0, -32768, 32767
};

/* Caso 9: clamp_min == clamp_max. Datos del caso 3.
   Esperado: {1024, 1024, 1024, 1024}, checksum 0x02431000. */
#define T9_CLAMP_MIN  (Q_ONE / 4)
#define T9_CLAMP_MAX  (Q_ONE / 4)

/* Caso 10: clamp negativo. Datos del caso 3.
   Esperado: {-1024, -1024, -1024, -2048}, checksum 0x8E80EC00. */
#define T10_CLAMP_MIN  (-Q_ONE / 2)
#define T10_CLAMP_MAX  (-Q_ONE / 4)

/* Caso 11: input_size que no es múltiplo de 4. Datos del caso 6 con otro
   input_size. Clamp = el del caso 6.
   Checksums esperados:
     n=2  -> 0xD9C686DF    n=5 -> 0xFB83BD0C    n=7  -> 0x8C32BCE2
     n=9  -> 0x3B42CC96    n=13 -> 0x2FADCE47 */

/* ======================================================================
   SOLUCIÓN ESPERADA
   ====================================================================== */
static const int16_t esp_main[OUTPUT_SIZE] = { 4096, 0, 0, 64, 0 };
static const int16_t esp_1[T1_OUT]   = { 4096, 4096, 4096 };
static const int16_t esp_2[T2_OUT]   = { 0, 0, 0, 0 };
static const int16_t esp_3[T3_OUT]   = { -1, 1, 935, -4096 };
static const int16_t esp_4[T4_OUT]   = { 1792 };
static const int16_t esp_4b[T4_OUT]  = { 2048 };
static const int16_t esp_4c[T4_OUT]  = { 512 };
static const int16_t esp_5[T5_OUT]   = { 0, 2048, 4096 };
static const int16_t esp_6[T6_OUT]   = { -2048, 1271, 2048, -2048, -1953, -2048, 2048, 1299 };
static const int16_t esp_8[T8_OUT]   = { 32767, -32768, 32767, -32768, 32767 };
static const int16_t esp_9[T3_OUT]   = { 1024, 1024, 1024, 1024 };
static const int16_t esp_10[T3_OUT]  = { -1024, -1024, -1024, -2048 };

// ======================================================================
// REFERENCIA EN C
// ======================================================================

/* ============================================================
   Neurona fully-connected en Q12
   ============================================================ */

int16_t neuron_q12_C(
    const int16_t *input,
    const int16_t *weights,
    uint16_t n,
    int16_t bias_q12,
    int16_t clamp_min,
    int16_t clamp_max) {

    // El bias está en Q12; el acumulador trabaja en Q24.
    // Por tanto, bias_q12 se desplaza 12 bits a la izquierda.
    int32_t acc = ((int32_t)bias_q12) << Q_SHIFT;

    for (uint16_t i = 0; i < n; ++i) {
        // input[i] y weights[i] están en Q12.
        // Su producto queda en Q24 y se acumula directamente.
        acc += (int32_t)input[i] * (int32_t)weights[i];
    }

    // Volvemos de Q24 a Q12.
    // En ARM esto corresponde naturalmente a un desplazamiento aritmético ASR #12.
    int32_t result_q12 = acc >> Q_SHIFT;

    // Activación/saturación final.
    return clamp_i16_q12(result_q12, clamp_min, clamp_max);
}

/* ============================================================
   Capa fully-connected completa en Q12
   ============================================================ */

uint32_t dense_layer_q12_C(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max) {

    uint32_t checksum = 0;

    for (uint16_t o = 0; o < output_size; ++o) {
        const int16_t *weights_o = &weights[o * input_size];

        int16_t y = neuron_q12_C(
            input,
            weights_o,
            input_size,
            bias[o],
            clamp_min,
            clamp_max
        );

        output[o] = y;
        checksum = checksum * 33u + (uint16_t)y;
    }

    return checksum;
}

/* ============================================================
   Capa C llamando a neurona ARM/Thumb
   ============================================================ */

uint32_t dense_layer_q12_C_ARM(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max) {

    uint32_t checksum = 0;

    for (uint16_t o = 0; o < output_size; ++o) {
        const int16_t *weights_o = &weights[o * input_size];

        int16_t y = neuron_q12_ARM(
            input,
            weights_o,
            input_size,
            bias[o],
            clamp_min,
            clamp_max
        );

        output[o] = y;
        checksum = checksum * 33u + (uint16_t)y;
    }

    return checksum;
}

uint32_t dense_layer_q12_C_THB(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max) {

    uint32_t checksum = 0;

    for (uint16_t o = 0; o < output_size; ++o) {
        const int16_t *weights_o = &weights[o * input_size];

        int16_t y = neuron_q12_THB(
            input,
            weights_o,
            input_size,
            bias[o],
            clamp_min,
            clamp_max
        );

        output[o] = y;
        checksum = checksum * 33u + (uint16_t)y;
    }

    return checksum;
}

// ======================================================================
// VERIFICACIÓN AUTOMÁTICA
// ======================================================================

/**
 * @brief   Ejecuta una función de capa y compara su checksum con el de
 *          la referencia C (dense_layer_q12_C) para los mismos argumentos.
 *
 * @param   func         Puntero a la función de capa a verificar.
 * @param   input        Vector de entrada Q12.
 * @param   weights      Matriz de pesos Q12.
 * @param   bias         Vector de bias Q12.
 * @param   input_size   Número de entradas.
 * @param   output_size  Número de salidas/neuronas (<= TEST_MAX_OUTPUT).
 * @param   clamp_min    Límite inferior Q12.
 * @param   clamp_max    Límite superior Q12.
 * @return  1 si el checksum y el vector de salida coinciden con la versión C
 *          0 si difieren.
 */
uint8_t func_verificar(
    uint32_t (*func)(const int16_t *, const int16_t *, const int16_t *, int16_t *,
                     uint16_t, uint16_t, int16_t, int16_t),
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max
) {
    static int16_t out_ref  [TEST_MAX_OUTPUT];
    static int16_t out_func [TEST_MAX_OUTPUT];

    // Rellenamos con el valor de relleno
    for (int i = 0; i < TEST_MAX_OUTPUT; ++i) {
        out_ref[i]  = RELLENO_SALIDA;
        out_func[i] = RELLENO_SALIDA;
    }

    uint32_t chk_ref = dense_layer_q12_C(
        input, weights, bias, out_ref,
        input_size, output_size, clamp_min, clamp_max);

    uint32_t chk_func = func(
        input, weights, bias, out_func,
        input_size, output_size, clamp_min, clamp_max);

    if (chk_func != chk_ref) {
        return 0;
    }

    // Comparamos los buffers enteros: valores correctos en las primeras
    // output_size posiciones y relleno en el resto.
    if (!vectores_iguales_i16(out_ref, out_func, TEST_MAX_OUTPUT)) {
        return 0;
    }

    return 1;
}

/**
 * @brief  Comprueba la referencia C contra una solución esperada calculada aparte.
 * @param  chk_esp  Checksum esperado.
 * @param  out_esp  Vector de salida esperado (NULL = comprobar solo el checksum).
 * @return 1 si coincide, 0 si no.
 */
static uint8_t C_cumple_esperado(
    const int16_t *input, const int16_t *weights, const int16_t *bias,
    uint16_t input_size, uint16_t output_size,
    int16_t clamp_min, int16_t clamp_max,
    uint32_t chk_esp, const int16_t *out_esp) {

    static int16_t out[TEST_MAX_OUTPUT];

    uint32_t chk = dense_layer_q12_C(
        input, weights, bias, out,
        input_size, output_size, clamp_min, clamp_max);

    if (chk != chk_esp) return 0;
    if (out_esp != 0 && !vectores_iguales_i16(out, out_esp, output_size)) return 0;
    return 1;
}

/**
 * @brief   Ejecuta C, C_ARM, C_THB, ARM_C, ARM y THB y verifica resultados.
 *
 * @param   input        Vector de entrada Q12.
 * @param   weights      Matriz de pesos Q12.
 * @param   bias         Vector de bias Q12.
 * @param   resultado    Se deja aquí la salida de la versión C.
 * @param   input_size   Número de entradas.
 * @param   output_size  Número de salidas/neuronas.
 * @param   clamp_min    Límite inferior Q12.
 * @param   clamp_max    Límite superior Q12.
 * @return  1 si todos devuelven el mismo checksum y el mismo vector; 0 si difieren.
 */
uint8_t dense_q12_verificar(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *resultado,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max) {

    static int16_t out_C     [OUTPUT_SIZE];
    static int16_t out_C_ARM [OUTPUT_SIZE];
    static int16_t out_C_THB [OUTPUT_SIZE];
    static int16_t out_ARM_C [OUTPUT_SIZE];
    static int16_t out_ARM   [OUTPUT_SIZE];
    static int16_t out_THB   [OUTPUT_SIZE];

    uint32_t chk_C = dense_layer_q12_C(
        input, weights, bias, out_C,
        input_size, output_size, clamp_min, clamp_max);

    uint32_t chk_C_ARM = dense_layer_q12_C_ARM(
        input, weights, bias, out_C_ARM,
        input_size, output_size, clamp_min, clamp_max);

    uint32_t chk_C_THB = dense_layer_q12_C_THB(
        input, weights, bias, out_C_THB,
        input_size, output_size, clamp_min, clamp_max);

    uint32_t chk_ARM_C = dense_layer_q12_ARM_C(
        input, weights, bias, out_ARM_C,
        input_size, output_size, clamp_min, clamp_max);

    uint32_t chk_ARM = dense_layer_q12_ARM(
        input, weights, bias, out_ARM,
        input_size, output_size, clamp_min, clamp_max);

    uint32_t chk_THB = dense_layer_q12_THB(
        input, weights, bias, out_THB,
        input_size, output_size, clamp_min, clamp_max);

    for (uint16_t i = 0; i < output_size; ++i) {
        resultado[i] = out_C[i];
    }

    if (!(chk_C == chk_C_ARM && chk_C == chk_C_THB &&
          chk_C == chk_ARM_C && chk_C == chk_ARM && chk_C == chk_THB)) {
        return 0;
    }

    if (!vectores_iguales_i16(out_C, out_C_ARM, output_size)) return 0;
    if (!vectores_iguales_i16(out_C, out_C_THB, output_size)) return 0;
    if (!vectores_iguales_i16(out_C, out_ARM_C, output_size)) return 0;
    if (!vectores_iguales_i16(out_C, out_ARM,   output_size)) return 0;
    if (!vectores_iguales_i16(out_C, out_THB,   output_size)) return 0;

    return 1;
}

/* Para un mismo caso: las 5 funciones de capa contra la referencia C, y la
   referencia C contra la solución esperada (chk_esp y out_esp. out_esp = 0
   para comprobar solo el checksum) */
#define CHECK_ALL(in, w, b, ni, no, lo, hi, chk_esp, out_esp) do { \
    ok_tests &= func_verificar(dense_layer_q12_C_ARM, (in), (w), (b), (ni), (no), (lo), (hi)); \
    ok_tests &= func_verificar(dense_layer_q12_C_THB, (in), (w), (b), (ni), (no), (lo), (hi)); \
    ok_tests &= func_verificar(dense_layer_q12_ARM_C, (in), (w), (b), (ni), (no), (lo), (hi)); \
    ok_tests &= func_verificar(dense_layer_q12_ARM,   (in), (w), (b), (ni), (no), (lo), (hi)); \
    ok_tests &= func_verificar(dense_layer_q12_THB,   (in), (w), (b), (ni), (no), (lo), (hi)); \
    ok_tests &= C_cumple_esperado((in), (w), (b), (ni), (no), (lo), (hi), (chk_esp), (out_esp)); \
} while (0)

/* ============================================================
   MAIN de prueba
   ============================================================ */
int main(void) {
    // Vector de entrada de prueba, en Q12.
    // Se puede visualizar en Keil como enteros int16_t.
    // Ejemplos:
    //   Q_ONE      = 4096  ->  1.0
    //   Q_ONE / 2  = 2048  ->  0.5
    //  -Q_ONE / 4  = -1024 -> -0.25
    static const int16_t input[INPUT_SIZE] __attribute__((aligned(8))) = {
         Q_ONE,        //  1.00
         Q_ONE / 2,    //  0.50
        -Q_ONE / 4,    // -0.25
         3 * Q_ONE / 4,//  0.75
         0,            //  0.00
        -Q_ONE / 2,    // -0.50
         Q_ONE / 4,    //  0.25
         Q_ONE / 8     //  0.125
    };

    // Matriz de pesos: OUTPUT_SIZE filas x INPUT_SIZE columnas, en Q12.
    // Cada fila corresponde a una neurona de salida.
    static const int16_t weights[OUTPUT_SIZE * INPUT_SIZE] __attribute__((aligned(8))) = {
        // Neurona 0
         Q_ONE / 2,   Q_ONE / 2,  -Q_ONE / 4,   Q_ONE / 4,
         0,          -Q_ONE / 8,   Q_ONE / 4,   0,

        // Neurona 1
        -Q_ONE / 2,   Q_ONE / 4,   Q_ONE / 2,  -Q_ONE / 4,
         Q_ONE / 8,   Q_ONE / 4,  -Q_ONE / 8,   Q_ONE / 8,

        // Neurona 2
         Q_ONE,      -Q_ONE,       Q_ONE / 2,   0,
         0,           Q_ONE / 4,   Q_ONE / 4,  -Q_ONE / 2,

        // Neurona 3
        -Q_ONE / 4,   Q_ONE / 4,   Q_ONE / 4,   Q_ONE / 4,
        -Q_ONE / 2,   Q_ONE / 2,   0,           Q_ONE / 8,

        // Neurona 4
         Q_ONE / 8,  -Q_ONE / 8,   Q_ONE / 8,  -Q_ONE / 8,
         Q_ONE / 2,   Q_ONE / 2,   Q_ONE / 4,   Q_ONE / 4
    };

    // Un bias por neurona, también en Q12 e int16_t.
    static const int16_t bias[OUTPUT_SIZE] __attribute__((aligned(8))) = {
         0,
         Q_ONE / 8,
        -Q_ONE / 4,
         Q_ONE / 4,
         0
    };

    // Búfer para quedarse con la salida de referencia C.
    static int16_t resultado[OUTPUT_SIZE] __attribute__((aligned(8)));

    // Verificación de las 6 variantes entre sí con la capa base.
    uint8_t ok = dense_q12_verificar(
        input,
        weights,
        bias,
        resultado,
        INPUT_SIZE,
        OUTPUT_SIZE,
        CLAMP_MIN_Q12,
        CLAMP_MAX_Q12
    );

    uint8_t ok_tests = ok;

    // Capa base (clamp 0..1.0): 'resultado' guarda la salida de la C
    ok_tests &= vectores_iguales_i16(resultado, esp_main, OUTPUT_SIZE);

    // Caso 1: saturación superior.
    CHECK_ALL(t1_input, t1_weights, t1_bias, T1_IN, T1_OUT, CLAMP_MIN_Q12, CLAMP_MAX_Q12, 0x00463000u, esp_1);

    // Caso 2: saturación inferior.
    CHECK_ALL(t2_input, t2_weights, t2_bias, T2_IN, T2_OUT, CLAMP_MIN_Q12, CLAMP_MAX_Q12, 0x00000000u, esp_2);

    // Caso 3: clamp simétrico, negativos y truncamiento.
    CHECK_ALL(t3_input, t3_weights, t3_bias, T3_IN, T3_OUT, T3_CLAMP_MIN, T3_CLAMP_MAX, 0x8C61E067u, esp_3);

    // Caso 4: 1 x 1, clamp que no incluye el 0.
    CHECK_ALL(t4_input, t4_weights, t4_bias, T4_IN, T4_OUT, T4_CLAMP_MIN, T4_CLAMP_MAX, 0x00000700u, esp_4);

    // Caso 4b: satura por abajo.
    CHECK_ALL(t4_input, t4_weights, t4_bias, T4_IN, T4_OUT, T4B_CLAMP_MIN, T4B_CLAMP_MAX, 0x00000800u, esp_4b);

    // Caso 4c: satura por arriba.
    CHECK_ALL(t4_input, t4_weights, t4_bias, T4_IN, T4_OUT, T4C_CLAMP_MIN, T4C_CLAMP_MAX, 0x00000200u, esp_4c);

    // Caso 5: input_size = 0 (solo bias).
    CHECK_ALL(t1_input, t1_weights, t5_bias, T5_IN, T5_OUT, CLAMP_MIN_Q12, CLAMP_MAX_Q12, 0x00011800u, esp_5);

    // Caso 6: capa 16 x 8.
    CHECK_ALL(t6_input, t6_weights, t6_bias, T6_IN, T6_OUT, T6_CLAMP_MIN, T6_CLAMP_MAX, 0xB1F1AF49u, esp_6);

    // Caso 7: output_size = 0 (checksum 0).
    CHECK_ALL(t1_input, t1_weights, t1_bias, T1_IN, T7_OUT, CLAMP_MIN_Q12, CLAMP_MAX_Q12, 0x00000000u, 0);

    // Caso 8: extremos int16, clamp de rango completo.
    CHECK_ALL(t8_input, t8_weights, t8_bias, T8_IN, T8_OUT, T8_CLAMP_MIN, T8_CLAMP_MAX, 0x5490633Du, esp_8);

    // Caso 9: clamp_min == clamp_max.
    CHECK_ALL(t3_input, t3_weights, t3_bias, T3_IN, T3_OUT, T9_CLAMP_MIN, T9_CLAMP_MAX, 0x02431000u, esp_9);

    // Caso 10: clamp totalmente negativo.
    CHECK_ALL(t3_input, t3_weights, t3_bias, T3_IN, T3_OUT, T10_CLAMP_MIN, T10_CLAMP_MAX, 0x8E80EC00u, esp_10);

    // Caso 11: input_size no múltiplo de 4 (datos del caso 6. solo checksum).
    CHECK_ALL(t6_input, t6_weights, t6_bias,  2, T6_OUT, T6_CLAMP_MIN, T6_CLAMP_MAX, 0xD9C686DFu, 0);
    CHECK_ALL(t6_input, t6_weights, t6_bias,  5, T6_OUT, T6_CLAMP_MIN, T6_CLAMP_MAX, 0xFB83BD0Cu, 0);
    CHECK_ALL(t6_input, t6_weights, t6_bias,  7, T6_OUT, T6_CLAMP_MIN, T6_CLAMP_MAX, 0x8C32BCE2u, 0);
    CHECK_ALL(t6_input, t6_weights, t6_bias,  9, T6_OUT, T6_CLAMP_MIN, T6_CLAMP_MAX, 0x3B42CC96u, 0);
    CHECK_ALL(t6_input, t6_weights, t6_bias, 13, T6_OUT, T6_CLAMP_MIN, T6_CLAMP_MAX, 0x2FADCE47u, 0);

    // Guardamos el resultado en la variable global para verlo en el Watch.
    resultado_tests = ok_tests;

    // En este entorno no hay SO; nos quedamos en bucle.
    while (1) { /* no retornar */ }
}
