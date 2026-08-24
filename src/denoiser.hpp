#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace kimodo::detail {
class ggml_motion_weights;

// F32 TransformerEncoderBlock. Inputs/outputs are row-major [B,T,D], while
// the implementation creates GGML [D,T,B] views over the same byte order.
std::expected<std::vector<float>, std::string> run_motion_transformer(
    const ggml_motion_weights &weights, std::string_view prefix,
    std::span<const float> motion, std::size_t motion_dim,
    std::span<const float> text_embedding, std::span<const float> timesteps,
    std::span<const float> headings, std::size_t batch, std::size_t frames);

// Exact two-stage Kimodo denoiser for concatenated motion/mask inputs
// [B,T,546].  Returned clean prediction is [B,T,273].
std::expected<std::vector<float>, std::string> run_two_stage_denoiser(
    const ggml_motion_weights &weights, std::span<const float> motion_and_mask,
    std::span<const float> text_embedding, std::span<const float> timesteps,
    std::span<const float> headings, std::span<const float> motion_mask,
    std::size_t batch, std::size_t frames);

// Unconstrained separated CFG wrapper. `motion` is [T,273], embedding is
// [4096], and the result is one clean [T,273] prediction.
std::expected<std::vector<float>, std::string> run_separated_cfg_denoiser(
    const ggml_motion_weights &weights, std::span<const float> motion,
    std::span<const float> embedding, float timestep, float text_weight,
    float constraint_weight, std::size_t frames);

// Deterministic eta=0 DDIM sampling from caller-supplied F32 initial noise.
std::expected<std::vector<float>, std::string> sample_motion_from_noise(
    const ggml_motion_weights &weights, std::span<const float> initial_noise,
    std::span<const float> embedding, std::size_t frames, unsigned steps,
    float text_weight, float constraint_weight);

// Multi-prompt transition sampler. `observed` and `observed_mask` are [T,273]
// normalized motion-representation values/masks. This mirrors the upstream
// concat-mask denoiser: text, constraint, and unconditional CFG branches.
std::expected<std::vector<float>, std::string> sample_motion_from_noise_conditioned(
    const ggml_motion_weights &weights, std::span<const float> initial_noise,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, float first_heading, std::size_t frames,
    unsigned steps, float text_weight, float constraint_weight);
}
