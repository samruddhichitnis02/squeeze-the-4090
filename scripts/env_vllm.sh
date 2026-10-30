#!/usr/bin/env bash
# Source this file to activate the vLLM environment (separate from the BERT training env).
# FlashInfer JIT-compiles kernels and needs nvcc; pip ships one inside the venv, so point CUDA_HOME there.
VENV="$HOME/gpu-assignment/.venv-vllm"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
# The pip nvcc (13.4) and pip CUDA headers (13.0) disagree, so FlashInfer's JIT-compiled sampler fails to build.
# Use vLLM's built-in PyTorch sampler instead. Attention still runs on FlashAttention.
export VLLM_USE_FLASHINFER_SAMPLER=0
