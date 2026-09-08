// Copyright 2026 Alexander Antonov (AzRigTool).
// Licensed under the Apache License, Version 2.0 - see LICENSE in this repository.
//
// Pose-constrained generation.
//
// The motion model is a diffusion model, and `sample_motion_from_noise_conditioned`
// already lets a caller pin part of the representation while the rest is
// denoised around it - that is how multi-prompt transitions carry the previous
// segment's pose into the next one. Only the transition path ever built such a
// condition, so this tool exposes the same mechanism for poses supplied from
// outside, e.g. keyed by hand in a DCC.
//
// The mask follows the one the model was trained with: the root row and every
// joint position are pinned, plus the 6D rotations of the four end effectors.
// Everything else - velocities, foot contacts, non-end-effector orientations -
// is left for the sampler, and every unconstrained frame is fully free.
//
// Because the mask is per element, a pin need not be a whole pose. Pinning
// only the root row makes the character follow a path while the model decides
// how it walks it; pinning only the feet fixes the contacts and leaves the
// rest free. Version 2 of the file therefore carries the mask explicitly
// rather than hard-coding one policy here.
//
// Constraint file (little-endian), written by the DCC side:
//
//     char  magic[4]      "KMDP"
//     u32   version       1, 2 or 3
//     u32   count         number of constrained frames
//     u32   joints        must match the skeleton
//     repeated `count` times, in ascending frame order:
//         u32   frame         0-based index into the take
//         u32   flags         version 2 only; bit 0 pins the root row
//         f32   root[3]       Hips world position, metres, Y-up
//         f32   global[9*J]   row-major 3x3 global rotation per joint
//         u8    positions[J]  version 2 only; 1 pins that joint's position
//         u8    rotations[J]  version 2 only; 1 pins its 6D rotation
//         f32   weight        version 3 only; how strongly this frame is held,
//                             0 to 1. Anything below 1 blends the pinned value
//                             with what the model wanted, which is what lets a
//                             hold fade in and out instead of snapping on.
//
// Version 1 has no flags or mask arrays and means "pin everything": the root
// row, every joint position, and the end effectors' rotations.

#include "denoiser.hpp"
#include "ggml_weights.hpp"
#include "llm_text_encoder.hpp"
#include "motion_decode.hpp"
#include "skeleton.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kimodo::detail::skeleton_spec;

struct pose_frame {
    std::uint32_t frame = 0;
    float weight = 1.F;
    bool pin_root = true;
    std::array<float, 3> root{};
    std::vector<float> global;              // 9 * joints, row-major 3x3 per joint
    std::vector<std::uint8_t> pin_position; // per joint
    std::vector<std::uint8_t> pin_rotation; // per joint
};

template <typename T>
T read_pod(std::istream &in, const char *what) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) throw std::runtime_error(std::string("truncated constraint file at ") + what);
    return value;
}

std::vector<pose_frame> read_constraints(const std::filesystem::path &path,
                                         const skeleton_spec &skeleton) {
    const std::size_t joints = skeleton.joints();
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());

    char magic[4]{};
    in.read(magic, 4);
    if (!in || std::memcmp(magic, "KMDP", 4) != 0)
        throw std::runtime_error(path.string() + " is not a Kimodo pose constraint file");
    const auto version = read_pod<std::uint32_t>(in, "version");
    if (version < 1 || version > 3)
        throw std::runtime_error("unsupported constraint file version "
                                 + std::to_string(version));
    const auto count = read_pod<std::uint32_t>(in, "count");
    const auto file_joints = read_pod<std::uint32_t>(in, "joints");
    if (file_joints != joints)
        throw std::runtime_error("constraint file is for " + std::to_string(file_joints)
                                 + " joints, the model has " + std::to_string(joints));
    if (count == 0) throw std::runtime_error("constraint file pins no frames");

    std::vector<pose_frame> poses;
    poses.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        pose_frame pose;
        pose.frame = read_pod<std::uint32_t>(in, "frame index");
        if (version >= 2) pose.pin_root = (read_pod<std::uint32_t>(in, "flags") & 1u) != 0;
        for (float &value : pose.root) value = read_pod<float>(in, "root position");
        pose.global.resize(joints * 9);
        in.read(reinterpret_cast<char *>(pose.global.data()),
                static_cast<std::streamsize>(pose.global.size() * sizeof(float)));
        if (!in) throw std::runtime_error("truncated constraint file at joint rotations");

        pose.pin_position.assign(joints, version >= 2 ? 0 : 1);
        pose.pin_rotation.assign(joints, 0);
        if (version >= 2) {
            in.read(reinterpret_cast<char *>(pose.pin_position.data()),
                    static_cast<std::streamsize>(joints));
            in.read(reinterpret_cast<char *>(pose.pin_rotation.data()),
                    static_cast<std::streamsize>(joints));
            if (!in) throw std::runtime_error("truncated constraint file at masks");
            if (version >= 3) {
                pose.weight = read_pod<float>(in, "weight");
                if (!(pose.weight >= 0.F) || pose.weight > 1.F)
                    throw std::runtime_error(
                        "frame " + std::to_string(pose.frame)
                        + " has a weight of " + std::to_string(pose.weight)
                        + "; it must be between 0 and 1");
            }
        } else {
            for (unsigned joint : skeleton.end_effectors) pose.pin_rotation[joint] = 1;
        }

        // six() rebuilds a rotation by Gram-Schmidt, so it silently rewrites a
        // mirrored or scaled matrix into something else - a negatively scaled
        // control would pin a pose nobody authored. Refuse it instead.
        for (std::size_t joint = 0; joint < joints; ++joint) {
            const float *m = pose.global.data() + joint * 9;
            for (int column = 0; column < 3; ++column) {
                const float length = std::sqrt(m[column] * m[column]
                                               + m[3 + column] * m[3 + column]
                                               + m[6 + column] * m[6 + column]);
                if (std::abs(length - 1.F) > 1.e-3F)
                    throw std::runtime_error(
                        "frame " + std::to_string(pose.frame) + ", joint "
                        + std::to_string(joint) + ": axis " + std::to_string(column)
                        + " is scaled by " + std::to_string(length)
                        + " - a pinned pose must carry pure rotations");
            }
            const float determinant =
                m[0] * (m[4] * m[8] - m[5] * m[7])
                - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6]);
            if (determinant < 0.F)
                throw std::runtime_error(
                    "frame " + std::to_string(pose.frame) + ", joint "
                    + std::to_string(joint) + " is mirrored (determinant "
                    + std::to_string(determinant)
                    + ") - the solver cannot represent a mirrored joint");
        }

        if (index && pose.frame <= poses.back().frame)
            throw std::runtime_error("constrained frames must be in ascending order");
        if (!pose.pin_root
            && std::find(pose.pin_position.begin(), pose.pin_position.end(), 1)
                   == pose.pin_position.end()
            && std::find(pose.pin_rotation.begin(), pose.pin_rotation.end(), 1)
                   == pose.pin_rotation.end())
            throw std::runtime_error("frame " + std::to_string(pose.frame)
                                     + " is listed but pins nothing");
        poses.push_back(std::move(pose));
    }
    return poses;
}

// Forward kinematics on the model's own skeleton, so the pinned positions are
// exactly the proportions the network was trained on rather than the DCC rig's.
void pose_positions(const pose_frame &pose, const skeleton_spec &skeleton,
                    std::vector<std::array<float, 3>> &out) {
    const std::size_t joints = skeleton.joints();
    out.assign(joints, {});
    for (std::size_t joint = 0; joint < joints; ++joint) {
        const int parent = skeleton.parents[joint];
        if (parent < 0) {
            out[joint] = pose.root;
            continue;
        }
        const float *rotation = pose.global.data() + static_cast<std::size_t>(parent) * 9;
        const auto &offset = skeleton.offsets[joint];
        for (int axis = 0; axis < 3; ++axis)
            out[joint][static_cast<std::size_t>(axis)] =
                out[static_cast<std::size_t>(parent)][static_cast<std::size_t>(axis)]
                + rotation[axis * 3 + 0] * offset[0]
                + rotation[axis * 3 + 1] * offset[1]
                + rotation[axis * 3 + 2] * offset[2];
    }
}

// Same layout `condition_row` produces for a transition, built from a pose
// instead of from a previously sampled row.
//
// The split between row[0]/row[2] and the position block matters: the root's
// travel lives in row[0]/row[2] while joint positions stay relative to the
// hips. Folding the travel into the positions instead puts them metres from
// zero, far outside the range those channels were normalized over, and a pose
// pinned in the middle of a run is then ignored.
float write_condition(const pose_frame &pose, const skeleton_spec &skeleton,
                      float origin_x, float origin_z, float *row, float *mask) {
    const std::size_t joints = skeleton.joints();
    const std::size_t rotation_begin = 5 + 3 * joints;

    std::vector<std::array<float, 3>> posed;
    pose_positions(pose, skeleton, posed);

    const auto right = skeleton.hips[0], left = skeleton.hips[1];
    const float heading = std::atan2(posed[right][2] - posed[left][2],
                                     -(posed[right][0] - posed[left][0]));

    const float hips_x = posed[0][0], hips_z = posed[0][2];
    row[0] = hips_x - origin_x;   // accumulated root travel
    row[1] = posed[0][1];         // root height
    row[2] = hips_z - origin_z;
    row[3] = std::cos(heading);
    row[4] = std::sin(heading);
    for (std::size_t joint = 0; joint < joints; ++joint) {
        row[5 + joint * 3] = posed[joint][0] - hips_x;
        row[6 + joint * 3] = posed[joint][1];
        row[7 + joint * 3] = posed[joint][2] - hips_z;
        if (pose.pin_position[joint])
            for (std::size_t d = 0; d < 3; ++d)
                mask[5 + joint * 3 + d] = pose.weight;
    }
    if (pose.pin_root) for (std::size_t d = 0; d < 5; ++d) mask[d] = pose.weight;

    for (std::size_t joint = 0; joint < joints; ++joint) {
        const float *rotation = pose.global.data() + joint * 9;
        const std::size_t first = rotation_begin + joint * 6;
        // Column-major 6D continuity, matching sequence.cpp's v[(d%3)*3 + d/3].
        for (int d = 0; d < 6; ++d) {
            row[first + static_cast<std::size_t>(d)] = rotation[(d % 3) * 3 + d / 3];
            if (pose.pin_rotation[joint])
                mask[first + static_cast<std::size_t>(d)] = pose.weight;
        }
    }
    return heading;
}

void write_f32(const std::filesystem::path &path, const std::vector<float> &values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open " + path.string());
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

}  // namespace

int main(int argc, char **argv) try {
    if (argc < 9 || argc > 12) {
        std::cerr << "usage: " << argv[0]
                  << " MOTION.gguf TEXT_BUNDLE PROMPT.txt POSES.kmdp FRAMES STEPS SEED"
                     " OUTPUT_DIR [PIN_WEIGHT] [TEXT_WEIGHT] [HARD]\n"
                     "  PIN_WEIGHT   how hard the solve is pulled toward the pins (default 2)\n"
                     "  TEXT_WEIGHT  how literally the prompt is followed (default 2)\n"
                     "  HARD         1 forces the pinned channels exactly, 0 pulls (default 0)\n";
        return 2;
    }
    const std::string motion_path = argv[1];
    const std::string bundle_path = argv[2];
    std::ifstream prompt_file(argv[3]);
    const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
    if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read prompt");
    const std::filesystem::path poses_path = argv[4];
    const auto frames = static_cast<std::size_t>(std::stoul(argv[5]));
    const auto steps = static_cast<unsigned>(std::stoul(argv[6]));
    const auto seed = static_cast<std::uint64_t>(std::stoull(argv[7]));
    const std::filesystem::path output = argv[8];
    const float pin_weight = argc > 9 ? std::stof(argv[9]) : 2.F;
    const float text_weight = argc > 10 ? std::stof(argv[10]) : 2.F;
    const bool hard = argc > 11 && std::stoi(argv[11]) != 0;
    if (frames < 2) throw std::runtime_error("frames must be at least 2");
    if (!(pin_weight >= 0.F) || !(text_weight >= 0.F))
        throw std::runtime_error("weights must be zero or positive");

    auto weights = kimodo::detail::ggml_motion_weights::load(motion_path);
    if (!weights) throw std::runtime_error(weights.error());
    const auto *skeleton = kimodo::detail::find_skeleton((*weights)->skeleton_key());
    if (!skeleton) throw std::runtime_error("unsupported skeleton in the motion model");

    const std::size_t dim = skeleton->motion_dim();
    const std::size_t body = dim - 5;
    const auto poses = read_constraints(poses_path, *skeleton);
    if (poses.back().frame >= frames)
        throw std::runtime_error("a constrained frame lies past the end of the take");

    // Upstream measures a take from the first pinned frame's ground origin.
    std::vector<std::array<float, 3>> first_posed;
    pose_positions(poses.front(), *skeleton, first_posed);
    const float origin_x = first_posed[0][0], origin_z = first_posed[0][2];

    std::vector<float> observed(frames * dim, 0.F), mask(frames * dim, 0.F);
    float heading = 0.F;
    for (std::size_t index = 0; index < poses.size(); ++index) {
        const std::size_t base = static_cast<std::size_t>(poses[index].frame) * dim;
        const float pose_heading = write_condition(poses[index], *skeleton, origin_x, origin_z,
                                                   observed.data() + base, mask.data() + base);
        if (index == 0) heading = pose_heading;
    }

    auto global_mean = (*weights)->f32_values("stats.global_root.mean");
    auto global_std = (*weights)->f32_values("stats.global_root.std");
    auto body_mean = (*weights)->f32_values("stats.body.mean");
    auto body_std = (*weights)->f32_values("stats.body.std");
    if (!global_mean || !global_std || !body_mean || !body_std)
        throw std::runtime_error("motion GGUF lacks compatible motion statistics");
    if (global_mean->size() != 5 || body_mean->size() != body)
        throw std::runtime_error("motion statistics do not match the skeleton");

    const auto scale = [](float stddev) { return std::sqrt(stddev * stddev + 1.e-5F); };
    for (std::size_t row = 0; row < frames; ++row) {
        float *values = observed.data() + row * dim;
        for (std::size_t d = 0; d < 5; ++d)
            values[d] = (values[d] - (*global_mean)[d]) / scale((*global_std)[d]);
        for (std::size_t d = 0; d < body; ++d)
            values[5 + d] = (values[5 + d] - (*body_mean)[d]) / scale((*body_std)[d]);
    }

    // Encoding the prompt means loading a 13 GB text model, which is about
    // 18 seconds of a 50 second run and identical every time the prompt has
    // not changed. So TEXT_BUNDLE may instead be a 4096-float embedding file,
    // and every run writes the embedding it used next to its output for the
    // caller to keep.
    std::array<float, 4096> embedding{};
    if (std::filesystem::is_regular_file(bundle_path)) {
        const auto bytes = std::filesystem::file_size(bundle_path);
        if (bytes != embedding.size() * sizeof(float))
            throw std::runtime_error(bundle_path + " is "
                                     + std::to_string(bytes)
                                     + " bytes; a cached embedding is "
                                     + std::to_string(embedding.size() * sizeof(float)));
        std::ifstream in(bundle_path, std::ios::binary);
        in.read(reinterpret_cast<char *>(embedding.data()),
                static_cast<std::streamsize>(bytes));
        if (!in) throw std::runtime_error("cannot read " + bundle_path);
        for (float value : embedding)
            if (!std::isfinite(value))
                throw std::runtime_error(bundle_path + " holds a non-finite value");
    } else {
        auto encoder = kimodo::detail::llm_text_encoder::load(bundle_path);
        if (!encoder) throw std::runtime_error(encoder.error());
        auto encoded = (*encoder)->encode(prompt);
        if (!encoded) throw std::runtime_error(encoded.error());
        embedding = *encoded;
    }

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.F, 1.F);
    std::vector<float> noise(frames * dim);
    for (float &value : noise) value = normal(rng);

    auto sampled = kimodo::detail::sample_motion_from_noise_conditioned(
        **weights, noise, embedding, observed, mask, heading, frames, steps,
        text_weight, pin_weight, hard);
    if (!sampled) throw std::runtime_error(sampled.error());

    auto decoded = kimodo::detail::decode_motion(*sampled, frames, *skeleton, *global_mean,
                                                 *global_std, *body_mean, *body_std);
    if (!decoded) throw std::runtime_error(decoded.error());

    // The condition was measured from the first pinned pose's ground origin,
    // the way a transition is, so the take comes back around zero. Upstream
    // adds that origin back to the root row; doing it on the decoded root
    // positions is the same shift and needs no round trip through the
    // normalized representation.
    for (std::size_t frame = 0; frame + 1 <= decoded->root_positions.size() / 3;
         ++frame) {
        decoded->root_positions[frame * 3 + 0] += origin_x;
        decoded->root_positions[frame * 3 + 2] += origin_z;
    }

    std::filesystem::create_directories(output);
    write_f32(output / "embedding.f32",
              std::vector<float>(embedding.begin(), embedding.end()));
    write_f32(output / "root_positions.f32", decoded->root_positions);
    write_f32(output / "local_rotations_xyzw.f32", decoded->local_xyzw);
    std::cout << "generated " << frames << " frames with " << skeleton->joints()
              << " joints, " << poses.size() << " pinned\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
