# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Kev-0.8B** (`jaredpalmer/kev-0.8b`) is a model of the registry (variants `bf16`, the default,
  and `f32`). It is published as a base model plus a LoRA adapter, so the library downloads the
  original files, checks them by SHA-256, converts them into a GGUF with a converter written in
  Rust (merge of the LoRA, the transformations of the Qwen3.5 family, the vocabulary), keeps the
  result in the cache and loads it. No hosted weights, no Python or PyTorch at runtime.
  The converter output equals the one of llama.cpp's official script tensor by tensor (the
  weights a LoRA adapted, the `bf16` and `f32` files: identical to the byte; only 13 of the 288
  values of `A_log` differ by one unit in the last place, because PyTorch's `exp` is not
  correctly rounded). See `docs/model-files.md` for where the files go and how big they are.
- A reader of `head.pt` files that never executes the pickle: an allow-list of three globals and
  of opcodes, limits on every size, an `IncompatibleModelError` naming the opcode or the global
  and its offset for anything else.
- The converter refuses, instead of guessing, a base model with a number of MTP layers other
  than one and any number of its config that does not fit the 32 bits of a GGUF integer.
- A verification record also for converted models: the second `from_pretrained` neither converts
  nor checks again; a converted file that no longer matches its record is rebuilt once from the
  verified originals.
- `demos/`: runnable, heavily commented scripts that need only the standard library and
  `archai_jev` (they are not part of the wheel). `000_truth_invaders.py` is a retro arcade game
  played by the model; `001` to `007` teach one concept each: the four tasks and typed answers,
  a grounding check, a duplicate finder, every error the library raises on purpose, calibration
  and `temperature=`, async (`aask_many`), and testing your own code with `MockScorer`.

## [0.1.0] - 2026-10-06

### Added
- `Jev.from_pretrained(...)`: loads the default demo model, a model of the registry or a local
  folder with a manifest; every file is verified (SHA-256), the weights, tokenizer and head are
  checked, and a self-check of the model's numbers runs on first use. `list_models()`.
- The default model `nickprock/archai-jev-qwen-1.5b` (Q8_0, about 1.6 GB, downloaded on first use),
  with four declared tasks (safety, intent, entailment, similarity) and a declared calibration
  (temperature 2.09, measured in-distribution only). Any other question raises
  `UnsupportedRequestError` listing the tasks.
- Inference engine on llama.cpp (CPU): the prompt is built and tokenized by the library (identical
  ids to the training formatters and to Kev's prompt: parity tested), the state is decoded once
  per request and each question runs on a copy of it, and the answer is read at the last token.
- New exceptions: `InferenceError`, and `UnsupportedRequestError` now has its real emissions: a
  prompt longer than the model's context, a question the model cannot answer, options that read the
  same to the model. Nothing is ever truncated.
- `ModelInfo` gains `dtype`, `source`, `calibration_source`, `tasks`, `notice`, `is_default`.
- `calibrated_softmax(logits, temperature=1.0)`: temperature-scaled softmax computed in Rust.
- Single abi3 wheel per platform supporting Python 3.10 and newer.

### Changed
- `calibrated_softmax` now rejects a non-finite `temperature` (`inf`) with `ValueError`.
