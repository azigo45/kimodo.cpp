// Replay the upstream multi-prompt capture with its recorded initial noise.
// This isolates the motion transition from cross-framework RNG and text-model
// differences, while checking both DDIM trajectories and the final blend.
#include "denoiser.hpp"
#include "ggml_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t features = 273;

std::vector<float> read(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0 || input.tellg() % static_cast<std::streamoff>(sizeof(float)))
        throw std::runtime_error("invalid fixture tensor: " + path);
    std::vector<float> value(static_cast<std::size_t>(input.tellg()) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(value.data()), static_cast<std::streamsize>(value.size() * sizeof(float)));
    if (!input) throw std::runtime_error("short fixture tensor: " + path);
    return value;
}

struct error { float max_abs = 0; double relative_l2 = 0; };
error compare(const std::vector<float> &actual, const std::vector<float> &expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("fixture shape mismatch");
    double squared_error = 0, squared_reference = 0;
    error result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = actual[i] - expected[i];
        result.max_abs = std::max(result.max_abs, std::abs(difference));
        squared_error += static_cast<double>(difference) * difference;
        squared_reference += static_cast<double>(expected[i]) * expected[i];
    }
    result.relative_l2 = std::sqrt(squared_error / squared_reference);
    return result;
}

void unnormalize(std::vector<float> &motion, const std::vector<float> &global_mean,
                 const std::vector<float> &global_std, const std::vector<float> &body_mean,
                 const std::vector<float> &body_std) {
    for (std::size_t row = 0; row < motion.size() / features; ++row) {
        auto *value = motion.data() + row * features;
        for (std::size_t d = 0; d < 5; ++d) value[d] = value[d] * global_std[d] + global_mean[d];
        for (std::size_t d = 0; d < 268; ++d) value[5 + d] = value[5 + d] * body_std[d] + body_mean[d];
    }
}
}

int main(int argc, char **argv) try {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MOTION.gguf FIXTURE_DIR\n", argv[0]);
        return 2;
    }
    const std::string directory = std::string(argv[2]) + "/";
    auto weights = kimodo::detail::ggml_motion_weights::load(argv[1]);
    if (!weights) throw std::runtime_error(weights.error());

    const auto first = kimodo::detail::sample_motion_from_noise(
        **weights, read(directory + "segment_00_sampling_input_000.f32"),
        read(directory + "segment_00_text_features.f32"), 30, 2, 2.F, 2.F);
    if (!first) throw std::runtime_error(first.error());
    const auto first_error = compare(*first, read(directory + "segment_00_sampling_output_001.f32"));

    const auto heading = read(directory + "segment_01_first_heading_angle.f32");
    const auto second = kimodo::detail::sample_motion_from_noise_conditioned(
        **weights, read(directory + "segment_01_sampling_input_000.f32"),
        read(directory + "segment_01_text_features.f32"), read(directory + "segment_01_observed_motion.f32"),
        read(directory + "segment_01_motion_mask.f32"), heading.at(0), 35, 2, 2.F, 2.F);
    if (!second) throw std::runtime_error(second.error());
    const auto second_error = compare(*second, read(directory + "segment_01_sampling_output_001.f32"));

    auto global_mean = (*weights)->f32_values("stats.global_root.mean");
    auto global_std = (*weights)->f32_values("stats.global_root.std");
    auto body_mean = (*weights)->f32_values("stats.body.mean");
    auto body_std = (*weights)->f32_values("stats.body.std");
    if (!global_mean || !global_std || !body_mean || !body_std) throw std::runtime_error("missing motion statistics");
    auto stitched_first = *first;
    auto stitched_second = *second;
    unnormalize(stitched_first, *global_mean, *global_std, *body_mean, *body_std);
    unnormalize(stitched_second, *global_mean, *global_std, *body_mean, *body_std);
    constexpr std::size_t overlap = 5;
    // The captured observed tensor is already translated to local origin;
    // recover the world origin from the first segment's retained tail.
    const float origin_x = stitched_first[(30 - overlap) * features];
    const float origin_z = stitched_first[(30 - overlap) * features + 2];
    for (std::size_t frame = 0; frame < 35; ++frame) {
        stitched_second[frame * features] += origin_x;
        stitched_second[frame * features + 2] += origin_z;
    }
    for (std::size_t frame = 0; frame < overlap; ++frame) {
        const float alpha = 1.F - static_cast<float>(frame) / static_cast<float>(overlap - 1);
        for (std::size_t d = 0; d < features; ++d)
            stitched_first[(30 - overlap + frame) * features + d] =
                alpha * stitched_first[(30 - overlap + frame) * features + d] +
                (1.F - alpha) * stitched_second[frame * features + d];
    }
    stitched_first.insert(stitched_first.end(), stitched_second.begin() + static_cast<std::ptrdiff_t>(overlap * features), stitched_second.end());
    const auto stitched_error = compare(stitched_first, read(directory + "stitched_motion_rep.f32"));
    std::printf("segment0 max_abs=%g rel_l2=%g\nsegment1 max_abs=%g rel_l2=%g\nstitched max_abs=%g rel_l2=%g\n",
                first_error.max_abs, first_error.relative_l2, second_error.max_abs, second_error.relative_l2,
                stitched_error.max_abs, stitched_error.relative_l2);
    return (first_error.max_abs <= 3.e-3F && second_error.max_abs <= 3.e-3F && stitched_error.max_abs <= 3.e-3F) ? 0 : 1;
} catch (const std::exception &error) {
    std::fprintf(stderr, "multi-prompt fixture parity error: %s\n", error.what());
    return 1;
}
