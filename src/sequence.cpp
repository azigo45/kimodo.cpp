#include "denoiser.hpp"
#include "ggml_weights.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace kimodo::detail {
namespace {
constexpr int parent[22]={-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19};
constexpr float offset[22][3]={{0,0,0},{.052299179F,-.093935639F,-.027606763F},{-.057192899F,-.106548190F,-.022217851F},{-.001495834F,.112929940F,-.024981268F},{.058866613F,-.416441321F,-.006556974F},{-.048074268F,-.397559673F,-.014061437F},{.006900469F,.145636231F,-.006858510F},{-.041737989F,-.437583506F,-.029511765F},{.014489345F,-.446852267F,-.018029511F},{-.010334037F,.056081813F,.021115851F},{.049293540F,-.065279245F,.126259089F},{-.040575184F,-.065286517F,.127075911F},{-.011025756F,.171365142F,-.028827066F},{.047724526F,.087643057F,-.008375450F},{-.046636276F,.086612143F,-.014864366F},{.024654359F,.175390735F,.024463326F},{.126284808F,.057680372F,-.013885141F},{-.109341696F,.053674292F,-.009117880F},{.272907287F,-.069853373F,-.039094493F},{-.292028785F,-.035440356F,-.024564851F},{.276173830F,.021254137F,-.002478220F},{-.271878421F,-.004834589F,-.016445294F}};
struct mat { double v[9]; };
mat mul(const mat&a,const mat&b){mat r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)r.v[i*3+j]+=a.v[i*3+k]*b.v[k*3+j];return r;}
mat trans(const mat&a){mat r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)r.v[i*3+j]=a.v[j*3+i];return r;}
mat cont6(const float *x) { double a[3]={x[0],x[1],x[2]}, n=std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); for(double &q:a)q/=n; double z[3]={a[1]*x[5]-a[2]*x[4],a[2]*x[3]-a[0]*x[5],a[0]*x[4]-a[1]*x[3]}; n=std::sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);for(double&q:z)q/=n; double b[3]={z[1]*a[2]-z[2]*a[1],z[2]*a[0]-z[0]*a[2],z[0]*a[1]-z[1]*a[0]};return {{a[0],b[0],z[0],a[1],b[1],z[1],a[2],b[2],z[2]}}; }
void rotate(const mat&m,const float *x,float *o){for(int i=0;i<3;++i)o[i]=static_cast<float>(m.v[i*3]*x[0]+m.v[i*3+1]*x[1]+m.v[i*3+2]*x[2]);}
}
std::expected<sequence_transition, std::string> prepare_sequence_transition(
    const ggml_motion_weights &, std::span<const float> previous,
    std::size_t continuation_frames, unsigned transition_frames) {
    constexpr size_t features = 273;
    const size_t overlap=transition_frames;
    if (!overlap || overlap>=continuation_frames || previous.size()<=(overlap*features))
        return std::unexpected("invalid sequence transition");
    sequence_transition result;
    result.observed.resize((continuation_frames+overlap)*features);
    result.observed_mask.resize(result.observed.size());
    const size_t previous_start=previous.size()-overlap*features;
    constexpr std::array<std::pair<size_t, size_t>, 3> constrained = {{{0, 71}, {113, 125}, {191, 203}}};
    for (size_t frame=0; frame<overlap; ++frame) {
        const size_t base=frame*features;
        std::array<float, features> value{};
        const float *raw=previous.data()+previous_start+base;
        std::copy_n(raw,features,value.data());
        mat decoded[22],local[22],global[22]; for(int j=0;j<22;++j)decoded[j]=cont6(value.data()+71+j*6);
        for(int j=0;j<22;++j) local[j]=parent[j]<0?decoded[j]:mul(trans(decoded[parent[j]]),decoded[j]);
        float root[3]={value[0]+value[5],value[6],value[2]+value[7]}, posed[22][3]{};
        for(int j=0;j<22;++j){if(parent[j]<0){global[j]=local[j];posed[j][0]=root[0];posed[j][1]=root[1];posed[j][2]=root[2];}else{global[j]=mul(global[parent[j]],local[j]);float d[3];rotate(global[parent[j]],offset[j],d);for(int k=0;k<3;++k)posed[j][k]=posed[parent[j]][k]+d[k];}}
        // FullBodyConstraintSet: smooth root, root Y, heading, all joint
        // positions; EndEffectorConstraintSet adds its four rotation blocks.
        value[0]=value[0]; value[1]=root[1]; value[2]=value[2];
        // `compute_heading_angle`: right hip minus left hip.
        const float dx=posed[2][0]-posed[1][0], dz=posed[2][2]-posed[1][2], angle=std::atan2(dz,-dx);
        value[3]=std::cos(angle); value[4]=std::sin(angle);
        for(int j=0;j<22;++j){value[5+j*3]=posed[j][0]-value[0];value[6+j*3]=posed[j][1];value[7+j*3]=posed[j][2]-value[2];}
        for(int j: {7,8,20,21}) for(int d=0;d<6;++d)
            value[71+j*6+d]=static_cast<float>(global[j].v[(d%3)*3+d/3]);
        std::copy_n(value.data(),203,result.observed.data()+base);
        for (const auto &[first,last] : constrained)
            std::fill(result.observed_mask.begin()+static_cast<std::ptrdiff_t>(base+first),
                      result.observed_mask.begin()+static_cast<std::ptrdiff_t>(base+last),1.F);
    }
    result.origin_x=result.observed[0];
    result.origin_z=result.observed[2];
    for (size_t frame=0; frame<overlap; ++frame) {
        auto *row=result.observed.data()+frame*features;
        row[0]-=result.origin_x;
        row[2]-=result.origin_z;
    }
    // First heading comes from the first retained full-body constraint.
    const size_t first=(previous.size()-overlap*features);
    const float *raw=previous.data()+first;
    std::array<float, features> value{}; std::copy_n(raw,features,value.data());
    mat decoded[22],local[22],global[22]; for(int j=0;j<22;++j)decoded[j]=cont6(value.data()+71+j*6);
    for(int j=0;j<22;++j)local[j]=parent[j]<0?decoded[j]:mul(trans(decoded[parent[j]]),decoded[j]);
    float root[3]={value[0]+value[5],value[6],value[2]+value[7]}, posed[22][3]{};
    for(int j=0;j<22;++j){if(parent[j]<0){global[j]=local[j];for(int k=0;k<3;++k)posed[j][k]=root[k];}else{global[j]=mul(global[parent[j]],local[j]);float d[3];rotate(global[parent[j]],offset[j],d);for(int k=0;k<3;++k)posed[j][k]=posed[parent[j]][k]+d[k];}}
    result.first_heading=std::atan2(posed[2][2]-posed[1][2],-(posed[2][0]-posed[1][0]));
    return result;
}

std::expected<std::vector<float>, std::string> sample_motion_sequence_from_noise(
    const ggml_motion_weights &weights, std::span<const sampled_sequence_segment> segments,
    unsigned transition_frames, unsigned steps, float text_weight, float constraint_weight) {
    constexpr size_t features = 273;
    if (segments.empty() || !transition_frames)
        return std::unexpected("sequence requires segments and a transition");
    auto gm=weights.f32_values("stats.global_root.mean"), gs=weights.f32_values("stats.global_root.std");
    auto bm=weights.f32_values("stats.body.mean"), bs=weights.f32_values("stats.body.std");
    if (!gm || !gs || !bm || !bs) return std::unexpected("motion GGUF lacks motion statistics");
    // Upstream Stats normalizes with sqrt(std^2 + 1e-5), rather than raw std.
    auto scale = [](float stddev) { return std::sqrt(stddev * stddev + 1.e-5F); };
    auto unnormalize = [&](std::vector<float> &motion) { for(size_t row=0;row<motion.size()/features;++row) { auto *v=motion.data()+row*features; for(size_t d=0;d<5;++d)v[d]=v[d]*scale((*gs)[d])+(*gm)[d]; for(size_t d=0;d<268;++d)v[5+d]=v[5+d]*scale((*bs)[d])+(*bm)[d]; } };
    auto normalize = [&](std::vector<float> &motion) { for(size_t row=0;row<motion.size()/features;++row) { auto *v=motion.data()+row*features; for(size_t d=0;d<5;++d)v[d]=(v[d]-(*gm)[d])/scale((*gs)[d]); for(size_t d=0;d<268;++d)v[5+d]=(v[5+d]-(*bm)[d])/scale((*bs)[d]); } };

    std::vector<float> joined, previous;
    for (size_t index=0; index<segments.size(); ++index) {
        const auto &segment=segments[index];
        const size_t sampled_frames=segment.frames+(index ? transition_frames : 0);
        if (segment.frames < 2 || segment.embedding.size()!=4096 ||
            segment.initial_noise.size()!=sampled_frames*features)
            return std::unexpected("invalid sampled sequence segment");
        std::vector<float> current;
        if (!index) {
            auto sampled=sample_motion_from_noise(weights,segment.initial_noise,segment.embedding,
                                                  sampled_frames,steps,text_weight,constraint_weight);
            if (!sampled) return std::unexpected(sampled.error());
            current=std::move(*sampled);
            unnormalize(current);
        } else {
            const size_t overlap=transition_frames;
            if (overlap >= segment.frames || previous.size()<overlap*features)
                return std::unexpected("transition must be shorter than every following segment");
            auto transition=prepare_sequence_transition(weights,previous,segment.frames,transition_frames);
            if (!transition) return std::unexpected(transition.error());
            const float origin_x=transition->origin_x;
            const float origin_z=transition->origin_z;
            normalize(transition->observed);
            auto sampled=sample_motion_from_noise_conditioned(weights,segment.initial_noise,segment.embedding,
                transition->observed,transition->observed_mask,transition->first_heading,sampled_frames,steps,text_weight,constraint_weight);
            if (!sampled) return std::unexpected(sampled.error());
            current=std::move(*sampled);
            unnormalize(current);
            for (size_t frame=0; frame<sampled_frames; ++frame) {
                auto *row=current.data()+frame*features;
                row[0]+=origin_x;
                row[2]+=origin_z;
            }
            const size_t start=joined.size()-overlap*features;
            for (size_t frame=0; frame<overlap; ++frame) {
                const float alpha=overlap==1?.5F:1.F-float(frame)/float(overlap-1);
                for (size_t d=0; d<features; ++d)
                    joined[start+frame*features+d]=alpha*joined[start+frame*features+d]+(1.F-alpha)*current[frame*features+d];
            }
            joined.insert(joined.end(),current.begin()+static_cast<std::ptrdiff_t>(overlap*features),current.end());
        }
        if (!index) joined=current;
        previous=std::move(current);
    }
    return joined;
}
} // namespace kimodo::detail
