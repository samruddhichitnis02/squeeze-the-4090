"""Memory techniques shared by batch_scale.py and train_bert.py so both apply them identically.

All default to off, which reproduces the original fp32 AdamW setup exactly.
"""
import contextlib

import torch


def add_args(p):
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast (weights and optimizer stay fp32)")
    p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing")
    p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")


def prepare_model(model, a):
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def make_optimizer(model, a, lr):
    if a.optim == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def autocast_ctx(a):
    if a.bf16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def describe(a):
    return dict(bf16=a.bf16, grad_ckpt=a.grad_ckpt, optim=a.optim)
