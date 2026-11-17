import sys
from pathlib import Path
import torch
from model.rwkv7_backbone import RWKV7BackboneForCausalLM
torch.set_num_threads(2)
model = RWKV7BackboneForCausalLM.from_pretrained(
    sys.argv[1], dtype=torch.bfloat16, local_files_only=True,
).eval()
tokens = torch.randint(0, 50257, (1, 1026), generator=torch.Generator().manual_seed(42))
with torch.inference_mode():
    prefix = model(tokens[:, :1025], use_cache=True, logits_to_keep=8)
    decoded = model(tokens[:, 1025:], use_cache=True, past_key_values=prefix.past_key_values,
                    position_ids=torch.tensor([[1025]]), logits_to_keep=1)
    torch.save(torch.cat((prefix.logits, decoded.logits), dim=1), Path(sys.argv[2]))
