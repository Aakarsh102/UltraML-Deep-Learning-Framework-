#pragma once

// Weight initialisation helpers. Values are drawn on the host with a
// seedable std::mt19937 and copied to the device, so runs are reproducible
// with manual_seed().

#include "../core/tensor.h"

namespace ultraml {
namespace nn {

void manual_seed(unsigned int seed);

void init_zeros   (Tensor* t);
void init_ones    (Tensor* t);
void init_constant(Tensor* t, float value);
void init_uniform (Tensor* t, float lo, float hi);
void init_normal  (Tensor* t, float mean = 0.0f, float std = 1.0f);

// Glorot: uniform(+-gain * sqrt(6 / (fan_in + fan_out))).
void init_xavier_uniform(Tensor* t, int fan_in, int fan_out,
                         float gain = 1.0f);

// He: uniform(+-gain * sqrt(3 / fan_in)); default gain sqrt(2) for ReLU.
void init_kaiming_uniform(Tensor* t, int fan_in,
                          float gain = 1.4142135624f);

} // namespace nn
} // namespace ultraml
