#include "init.h"

#include <cmath>
#include <random>
#include <vector>

namespace ultraml {
namespace nn {

namespace {

std::mt19937& rng() {
    static std::mt19937 gen(0x11a5eedu);
    return gen;
}

template <typename Dist>
void fill_from_dist(Tensor* t, Dist dist) {
    std::vector<float> host(t->size);
    for (int i = 0; i < t->size; ++i) host[i] = dist(rng());
    copy_from_host(t, host.data());
}

} // namespace

void manual_seed(unsigned int seed) { rng().seed(seed); }

void init_zeros(Tensor* t)               { zero_tensor(t); }
void init_ones (Tensor* t)               { fill_tensor(t, 1.0f); }
void init_constant(Tensor* t, float v)   { fill_tensor(t, v); }

void init_uniform(Tensor* t, float lo, float hi) {
    fill_from_dist(t, std::uniform_real_distribution<float>(lo, hi));
}

void init_normal(Tensor* t, float mean, float std) {
    fill_from_dist(t, std::normal_distribution<float>(mean, std));
}

void init_xavier_uniform(Tensor* t, int fan_in, int fan_out, float gain) {
    float bound = gain * std::sqrt(6.0f / (float)(fan_in + fan_out));
    init_uniform(t, -bound, bound);
}

void init_kaiming_uniform(Tensor* t, int fan_in, float gain) {
    float bound = gain * std::sqrt(3.0f / (float)fan_in);
    init_uniform(t, -bound, bound);
}

} // namespace nn
} // namespace ultraml
