import torch

def save_weights(model, path):
    (print(f"Saving to {path}"))
    torch.save(model.state_dict(), path)

def compute_vocab_size(modulus):
    # add 2 because of the + and = tokens
    return modulus + 2