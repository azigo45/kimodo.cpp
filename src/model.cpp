#include <kimodo/kimodo.hpp>
#include "gguf.hpp"
#ifdef KIMODO_HAVE_GGML
#include "ggml_weights.hpp"
#include "denoiser.hpp"
#include "motion_decode.hpp"
#include "llm_text_encoder.hpp"
#endif

#include <cmath>
#include <algorithm>
#include <random>

namespace kimodo {
struct model::impl {
    detail::gguf_file motion;
    std::string motion_path;
#ifdef KIMODO_HAVE_GGML
    mutable std::unique_ptr<detail::ggml_motion_weights> weights;
    std::unique_ptr<detail::llm_text_encoder> text;
#endif
};
model::model(std::unique_ptr<impl> state) : impl_(std::move(state)) {}
model::~model() = default;

std::expected<std::unique_ptr<model>, std::string> model::load(std::string_view motion_path, std::string_view text_path) {
    auto file = detail::read_gguf_header(motion_path);
    if (!file) return std::unexpected(file.error());
    if (auto valid = detail::validate_motion_gguf(*file); !valid) return std::unexpected(valid.error());
    auto state = std::make_unique<impl>();
    state->motion = std::move(*file);
    state->motion_path = std::string(motion_path);
#ifdef KIMODO_HAVE_GGML
    if (!text_path.empty()) {
        auto text = detail::llm_text_encoder::load(text_path);
        if (!text) return std::unexpected(text.error());
        state->text = std::move(*text);
    }
#else
    if (!text_path.empty()) return std::unexpected("Kimodo was built without GGML support");
#endif
    return std::unique_ptr<model>(new model(std::move(state)));
}

std::expected<motion_data, std::string> model::generate_text(
    std::string_view utf8_prompt, unsigned frames, unsigned steps, std::uint64_t seed,
    float text_cfg, float constraint_cfg) const {
#ifdef KIMODO_HAVE_GGML
    if (!impl_->text) return std::unexpected("model was loaded without a native text bundle");
    auto embedding = impl_->text->encode(utf8_prompt);
    if (!embedding) return std::unexpected(embedding.error());
    return generate_embedding(*embedding, frames, steps, seed, text_cfg, constraint_cfg);
#else
    (void) utf8_prompt; (void) frames; (void) steps; (void) seed; (void) text_cfg; (void) constraint_cfg;
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<motion_data, std::string> model::generate_embedding(
    const std::array<float, embedding_width> &embedding, unsigned frames, unsigned steps,
    std::uint64_t seed, float text_cfg, float constraint_cfg) const {
    if (frames == 0 || frames > 10000) return std::unexpected("frames must be in 1..10000");
    if (steps == 0 || steps > 1000) return std::unexpected("diffusion_steps must be in 1..1000");
    if (!std::isfinite(text_cfg) || !std::isfinite(constraint_cfg)) return std::unexpected("CFG weights must be finite");
    for (float value : embedding) if (!std::isfinite(value)) return std::unexpected("embedding contains a non-finite value");
#ifdef KIMODO_HAVE_GGML
    // Weight residency is deferred until inference so model-load stays a
    // bounded metadata operation.  The graph integration consumes this exact
    // session; no separate unchecked tensor loader exists in the runtime.
    if (!impl_->weights) {
        auto loaded = detail::ggml_motion_weights::load(impl_->motion_path);
        if (!loaded) return std::unexpected(loaded.error());
        impl_->weights = std::move(*loaded);
    }
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> noise(static_cast<size_t>(frames)*273);
    for (float &value : noise) value = normal(rng);
    auto sampled = detail::sample_motion_from_noise(*impl_->weights, noise, embedding, frames, steps, text_cfg, constraint_cfg);
    if (!sampled) return std::unexpected(sampled.error());
    auto global_mean=impl_->weights->f32_values("stats.global_root.mean"), global_std=impl_->weights->f32_values("stats.global_root.std");
    auto body_mean=impl_->weights->f32_values("stats.body.mean"), body_std=impl_->weights->f32_values("stats.body.std");
    if (!global_mean) return std::unexpected(global_mean.error());
    if (!global_std) return std::unexpected(global_std.error());
    if (!body_mean) return std::unexpected(body_mean.error());
    if (!body_std) return std::unexpected(body_std.error());
    auto decoded=detail::decode_smplx22(*sampled,frames,*global_mean,*global_std,*body_mean,*body_std);
    if (!decoded) return std::unexpected(decoded.error());
    motion_data result;
    result.frames=frames; result.joints=22;
    result.local_rotations_xyzw=std::move(decoded->local_xyzw);
    result.root_positions=std::move(decoded->root_positions);
    return result;
#else
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<motion_data, std::string> model::generate_text_sequence(
    std::span<const prompt_segment> segments, unsigned transition_frames,
    unsigned steps, std::uint64_t seed, float text_cfg, float constraint_cfg) const {
#ifdef KIMODO_HAVE_GGML
    if (!impl_->text) return std::unexpected("model was loaded without a native text bundle");
    if (segments.empty() || segments.size() > 16) return std::unexpected("sequence requires 1..16 prompt segments");
    if (steps == 0 || steps > 1000 || transition_frames == 0 || transition_frames > 60)
        return std::unexpected("invalid sequence sampling parameters");
    if (!impl_->weights) {
        auto loaded = detail::ggml_motion_weights::load(impl_->motion_path);
        if (!loaded) return std::unexpected(loaded.error());
        impl_->weights = std::move(*loaded);
    }
    auto gm=impl_->weights->f32_values("stats.global_root.mean"), gs=impl_->weights->f32_values("stats.global_root.std");
    auto bm=impl_->weights->f32_values("stats.body.mean"), bs=impl_->weights->f32_values("stats.body.std");
    if (!gm || !gs || !bm || !bs) return std::unexpected("motion GGUF lacks normalization statistics");
    std::mt19937_64 rng(seed); std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> joined, previous;
    for (size_t index=0; index<segments.size(); ++index) {
        const auto &segment=segments[index];
        if (segment.prompt.empty() || segment.frames < 2 || segment.frames > 300)
            return std::unexpected("each sequence segment must contain a prompt and have 2..300 frames");
        if (index && transition_frames >= segment.frames) return std::unexpected("transition must be shorter than every following segment");
        auto embedding=impl_->text->encode(segment.prompt); if (!embedding) return std::unexpected(embedding.error());
        std::vector<float> noise(static_cast<size_t>(segment.frames)*273); for (float &value : noise) value=normal(rng);
        std::vector<float> current;
        if (index == 0) {
            auto sampled=detail::sample_motion_from_noise(*impl_->weights,noise,*embedding,segment.frames,steps,text_cfg,constraint_cfg);
            if (!sampled) return std::unexpected(sampled.error()); current=std::move(*sampled);
        } else {
            // Derived from NVIDIA's Apache-2.0 `_multiprompt` sampler:
            // https://github.com/nv-tlabs/kimodo/blob/main/kimodo/model/kimodo_model.py
            // Preserve the prior tail as observed motion for the next DDIM
            // run, then use its transition frames to replace the old tail.
            std::vector<float> observed(noise.size()), observed_mask(noise.size());
            const auto overlap=static_cast<size_t>(transition_frames);
            const auto previous_start=previous.size()-overlap*273;
            // FullBodyConstraintSet conditions smooth root, heading, local
            // joint positions, and global rotations (203 values), but not
            // generated velocities or contact labels.
            for (size_t frame=0; frame<overlap; ++frame) {
                std::copy_n(previous.data()+previous_start+frame*273,203,observed.data()+frame*273);
                std::fill_n(observed_mask.data()+frame*273,203,1.f);
            }
            const float origin_x=observed[0]*(*gs)[0]+(*gm)[0];
            const float origin_z=observed[2]*(*gs)[2]+(*gm)[2];
            for (size_t frame=0; frame<overlap; ++frame) {
                auto *row=observed.data()+frame*273;
                row[0]=((row[0]*(*gs)[0]+(*gm)[0])-origin_x)/(*gs)[0];
                row[2]=((row[2]*(*gs)[2]+(*gm)[2])-origin_z)/(*gs)[2];
            }
            const auto p=(previous.size()/273-overlap)*273;
            const float heading=std::atan2(previous[p+4]*(*gs)[4]+(*gm)[4],previous[p+3]*(*gs)[3]+(*gm)[3]);
            auto sampled=detail::sample_motion_from_noise_conditioned(*impl_->weights,noise,*embedding,observed,observed_mask,heading,segment.frames,steps,text_cfg,constraint_cfg);
            if (!sampled) return std::unexpected(sampled.error()); current=std::move(*sampled);
            // `_multiprompt` samples in the translated local coordinates, then
            // restores the prior segment's planar smooth-root origin.
            for (size_t frame=0; frame<segment.frames; ++frame) {
                auto *row=current.data()+frame*273;
                row[0]=((row[0]*(*gs)[0]+(*gm)[0])+origin_x)/(*gs)[0];
                row[2]=((row[2]*(*gs)[2]+(*gm)[2])+origin_z)/(*gs)[2];
            }
            const auto start=joined.size()-overlap*273;
            for (size_t frame=0; frame<overlap; ++frame) {
                const float alpha=overlap==1?.5f:1.f-float(frame)/float(overlap-1);
                for (size_t d=0; d<273; ++d) joined[start+frame*273+d]=alpha*joined[start+frame*273+d]+(1.f-alpha)*current[frame*273+d];
            }
            joined.insert(joined.end(),current.begin()+static_cast<std::ptrdiff_t>(overlap*273),current.end());
        }
        if (index == 0) joined=current;
        previous=std::move(current);
    }
    const auto frames=static_cast<unsigned>(joined.size()/273);
    auto decoded=detail::decode_smplx22(joined,frames,*gm,*gs,*bm,*bs);
    if (!decoded) return std::unexpected(decoded.error());
    motion_data result; result.frames=frames; result.joints=22;
    result.local_rotations_xyzw=std::move(decoded->local_xyzw); result.root_positions=std::move(decoded->root_positions);
    return result;
#else
    (void) segments; (void) transition_frames; (void) steps; (void) seed; (void) text_cfg; (void) constraint_cfg;
    return std::unexpected("Kimodo was built without GGML support");
#endif
}
} // namespace kimodo
