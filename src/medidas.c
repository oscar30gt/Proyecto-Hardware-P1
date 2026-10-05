// Proyecto Hardware 2026 - Medida de ciclos de las variantes de la capa Q12.
// Mide con el Timer0 del LPC2105 (simulado en Keil uVision).
// Se llama desde main() cuando MODO_MEDIDA vale 1.

#include "dense_q12.h"
#include <stdint.h>

/* ============================================================
   Registros del Timer0 y del divisor de periféricos (LPC2105)
   ============================================================ */
#define T0TCR   (*(volatile uint32_t *)0xE0004004)
#define T0TC    (*(volatile uint32_t *)0xE0004008)
#define T0PR    (*(volatile uint32_t *)0xE000400C)
#define VPBDIV  (*(volatile uint32_t *)0xE01FC100)

#define N_VARIANTES  6
#define N_MEDIDAS    7              // las 6 variantes + una llamada vacía para calibrar
#define CHK_BASE     0x21881840u    // checksum esperado de la capa base 8x5

typedef uint32_t (*capa_fn)(const int16_t *, const int16_t *, const int16_t *, int16_t *,
                            uint16_t, uint16_t, int16_t, int16_t);

/* ============================================================
   Datos de la capa base (8 entradas x 5 salidas), en Q12
   ============================================================ */
static const int16_t med_input[INPUT_SIZE] __attribute__((aligned(8))) = {
     Q_ONE,         //  1.00
     Q_ONE / 2,     //  0.50
    -Q_ONE / 4,     // -0.25
     3 * Q_ONE / 4, //  0.75
     0,             //  0.00
    -Q_ONE / 2,     // -0.50
     Q_ONE / 4,     //  0.25
     Q_ONE / 8      //  0.125
};

static const int16_t med_weights[OUTPUT_SIZE * INPUT_SIZE] __attribute__((aligned(8))) = {
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

static const int16_t med_bias[OUTPUT_SIZE] __attribute__((aligned(8))) = {
     0,
     Q_ONE / 8,
    -Q_ONE / 4,
     Q_ONE / 4,
     0
};

/* ============================================================
   Resultados para mirar en el Watch
   Índices: 0=C  1=C_ARM  2=C_THB  3=ARM_C  4=ARM  5=THB  6=vacía
   ============================================================ */
volatile uint32_t ciclos[N_MEDIDAS];
volatile uint32_t chk_medida[N_MEDIDAS];
volatile uint8_t  medida_ok = 0xFF;   // 1 = todas las variantes dan el checksum esperado

// No hace nada: sirve para medir el coste del propio mecanismo de medida.
static uint32_t capa_vacia(const int16_t *a, const int16_t *b, const int16_t *c, int16_t *d,
                           uint16_t e, uint16_t f, int16_t g, int16_t h) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
    return 0;
}

// Tabla con punteros volatile: el compilador no puede saber qué función se llama,
// así que no la inlinea ni la mueve fuera de las lecturas del timer.
static capa_fn volatile variantes[N_MEDIDAS] = {
    dense_layer_q12_C,
    dense_layer_q12_C_ARM,
    dense_layer_q12_C_THB,
    dense_layer_q12_ARM_C,
    dense_layer_q12_ARM,
    dense_layer_q12_THB,
    capa_vacia
};

void medir_variantes(void) {
    static int16_t salida[OUTPUT_SIZE] __attribute__((aligned(8)));
    uint8_t ok_m = 1;

    VPBDIV = 1;    // reloj de periféricos = reloj de CPU: 1 tick = 1 ciclo
    T0PR   = 0;    // sin prescaler
    T0TCR  = 2;    // reset del contador
    T0TCR  = 1;    // arranca el contador

    for (int i = 0; i < N_MEDIDAS; ++i) {
        uint32_t t0  = T0TC;
        uint32_t chk = variantes[i](med_input, med_weights, med_bias, salida,
                                    INPUT_SIZE, OUTPUT_SIZE,
                                    CLAMP_MIN_Q12, CLAMP_MAX_Q12);
        uint32_t t1  = T0TC;

        ciclos[i]     = t1 - t0;
        chk_medida[i] = chk;
        if (i < N_VARIANTES && chk != CHK_BASE) ok_m = 0;
    }

    medida_ok = ok_m;
    while (1) { /* breakpoint aquí: mira ciclos[], chk_medida[] y medida_ok */ }
}
