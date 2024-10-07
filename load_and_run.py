import torch
from smollgpt import GPTModel, vocab_size,block_size , emb_size, num_heads, device, encode


checkpoint = torch.load("saved_models/final_gpt.pt", weights_only=True)
new_model = GPTModel(vocab_size, emb_size, num_heads)
new_model.load_state_dict(checkpoint)
m2 = new_model.to(device)
start_text = " "*block_size

num_tokens_to_generate = 1500
m2.generate(encode(start_text), num_tokens_to_generate)