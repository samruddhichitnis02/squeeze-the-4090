import torch
import numpy as np
import transformers
import datasets
import sklearn
from transformers import AutoModel

print("torch:", torch.__version__)
print("numpy:", np.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("sklearn:", sklearn.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))

# Real test: can transformers build a torch model on GPU?
from transformers import BertConfig, BertModel
cfg = BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=256)
m = BertModel(cfg).cuda()
ids = torch.randint(0, 100, (2, 16), device="cuda")
out = m(ids).last_hidden_state
print("bert forward on GPU ok, shape:", tuple(out.shape))
print("ALL GOOD")
