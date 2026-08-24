# kimodo.cpp

GGML/C++ implementation of NVIDIA's Kimodo text-to-motion model.

## Status

`Kimodo-SMPLX-RP-v1` accepts either a UTF-8 prompt or a precomputed LLM2Vec
embedding and generates unconstrained SMPL-X22 local rotations and root
translations on CPU or Vulkan. The text encoder uses eight-layer Vulkan chunks
by default; set `KIMODO_TEXT_LAYER_CHUNK=1..32` to tune VRAM use.

Included: checked GGUF loading, safetensors conversion, DDIM sampling, C/C++
APIs, CPU/Vulkan parity tests, and a local text-to-motion demo. Constraints,
SOMA, G1, GLB export, and quantised models are not implemented yet.

## Build and test on Linux

Install a C++23 compiler, CMake 3.25+, Ninja, Python 3 with the Hugging Face
CLI (`pip install huggingface_hub`), and the Vulkan loader/headers for Vulkan
support. GGML is a pinned Git submodule:

```sh
git submodule update --init --recursive
scripts/download_gguf_weights.sh --output "$PWD"
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The standard test suite requires the local motion GGUF, text bundle, and
fixtures. It never downloads weights by itself. `release`, `asan-ubsan`, and
`fuzz` presets are also available.

Nix is optional and provides these dependencies reproducibly:

```sh
nix develop path:. --command cmake --preset debug
nix develop path:. --command cmake --build --preset debug
```

For sanitizer work:

```sh
nix develop path:. --command cmake --preset asan-ubsan
nix develop path:. --command cmake --build --preset asan-ubsan
nix develop path:. --command env \
  LD_LIBRARY_PATH="$PWD/build/asan-ubsan/ggml/src:$PWD/build/asan-ubsan/ggml/src/ggml-vulkan:$LD_LIBRARY_PATH" \
  ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ctest --preset asan-ubsan --output-on-failure
```

Leak detection is disabled because Vulkan loader/driver allocations are global
to the process. The GGUF parser fuzzer requires Clang.

## API

`include/kimodo/kimodo_capi.h` is the C API. Model loading checks the motion
GGUF and text bundle before inference. Use `kimodo_generate_embedding` for
4096 F32 values or `kimodo_generate` for text. Both return SMPL-X22 root
translations and local XYZW rotations.

## Demo

After building the debug preset and downloading the native GGUF bundle:

```sh
go run ./demo -addr 0.0.0.0:8094
```

Open `http://localhost:8094`. The left sidebar contains the prompt and a
persistent history; choosing a previous animation restores its prompt for a
new generation.

## Weights

Ready-to-run native GGML weights are published under the Hugging Face
`LocalAI-io` organisation (not GitHub's `localai-org`). The reusable
[Llama-3-Kimodo-GGML](https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML)
text encoder and the upstream-linked
[Kimodo-SMPLX-RP-v1-GGML](https://huggingface.co/LocalAI-io/Kimodo-SMPLX-RP-v1-GGML)
diffusion model are separate, so users download rather than recreate them:

```sh
scripts/download_gguf_weights.sh --output "$PWD"
```

The installer verifies each published manifest and SHA-256 hashes. Use
`--motion-only` when supplying a precomputed 4096-float LLM2Vec embedding.

The GGUF bundle includes converted Meta Llama 3 material and Kimodo is
non-commercial research-only. Review the published model card and upstream
licences before downloading or redistributing.

## License

The C++ port and its original tooling are licensed under Apache-2.0; see
[LICENSE](LICENSE). GGML and the model weights retain their respective
licences.

### Regenerating the bundle

This is only needed to reproduce a conversion. The SMPL-X checkpoint and Llama
base model are gated. After accepting their Hugging Face licences and
authenticating, download the exact revisions and hash manifests with:

```sh
nix develop path:. --command hf auth login
nix develop path:. --command scripts/download_weights.sh \
  --output "$PWD/models" --with-text
```

Convert the local LLM2Vec model to the native component bundle with:

```sh
nix develop path:. --command scripts/convert_llm2vec_bundle.sh \
  "$PWD/models/llama3-8b-instruct-base" "$PWD/generated/llm2vec-text-bundle"
```

Validate a prospective release without network access, then explicitly upload
it from an account allowed to publish to `LocalAI-io`:

```sh
nix develop path:. --command python scripts/publish_gguf.py --component motion
nix develop path:. --command python scripts/publish_gguf.py --component motion \
  --upload --confirm-upstream-licences
nix develop path:. --command python scripts/publish_gguf.py --component text \
  --upload --confirm-upstream-licences
```
