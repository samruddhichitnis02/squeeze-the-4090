import bitsandbytes as bnb
import torch

print("bitsandbytes:", bnb.__version__)
w = torch.nn.Linear(4096, 4096).cuda()
opt = bnb.optim.AdamW8bit(w.parameters(), lr=1e-3)
x = torch.randn(8, 4096, device="cuda")
for _ in range(3):
    loss = w(x).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
states = [s for st in opt.state.values() for s in st.values() if torch.is_tensor(s)]
print("optimizer state dtypes:", sorted({str(s.dtype) for s in states}))
print("8-bit optimizer step OK, loss:", float(loss))
