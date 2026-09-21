#include "dense_q12.h"

#if ENABLE_ASM_IMPL == 0

__weak int16_t neuron_q12_ARM(
    const int16_t *input,
    const int16_t *weights,
    uint16_t n,
    int16_t bias_q12,
    int16_t clamp_min,
    int16_t clamp_max) {

    return neuron_q12_C(input, weights, n, bias_q12, clamp_min, clamp_max);
}

__weak int16_t neuron_q12_THB(
    const int16_t *input,
    const int16_t *weights,
    uint16_t n,
    int16_t bias_q12,
    int16_t clamp_min,
    int16_t clamp_max) {

    return neuron_q12_C(input, weights, n, bias_q12, clamp_min, clamp_max);
}

__weak uint32_t dense_layer_q12_ARM_C(  // DONE
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max) {

    return dense_layer_q12_C(input, weights, bias, output,
                             input_size, output_size, clamp_min, clamp_max);
}

__weak uint32_t dense_layer_q12_ARM(    // DONE
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max) {

    return dense_layer_q12_C(input, weights, bias, output,
                             input_size, output_size, clamp_min, clamp_max);
}

__weak uint32_t dense_layer_q12_THB(
    const int16_t *input,
    const int16_t *weights,
    const int16_t *bias,
    int16_t *output,
    uint16_t input_size,
    uint16_t output_size,
    int16_t clamp_min,
    int16_t clamp_max) {

    return dense_layer_q12_C(input, weights, bias, output,
                             input_size, output_size, clamp_min, clamp_max);
}

#endif /* ENABLE_ASM_IMPL == 0 */
