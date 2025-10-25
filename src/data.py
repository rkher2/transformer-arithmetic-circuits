import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

def create_vocabulary(modulus):
    tokens = []
    for i in range(modulus):
        tokens.append(str(i))
    tokens.append("+")
    tokens.append("=")

    token_to_int = dict()
    int_to_token = dict()
    for i, token in enumerate(tokens):
        token_to_int[token] = i
        int_to_token[i] = token
    return token_to_int, int_to_token

class ModularAdditionDataset(Dataset):
    def __init__(self, modulus):
        self.modulus = modulus
        self.token_to_int, self.int_to_token = create_vocabulary(modulus)
        self.vocab_size = len(self.token_to_int)

        self.plus_token = self.token_to_int['+']
        self.equals_token = self.token_to_int['=']
        self.data = []

        for a in range(modulus):
            for b in range(modulus):
                c = (a + b) % modulus
                self.data.append((a, b, c))
        random.shuffle(self.data)

        self.data_size = len(self.data)
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        a, b, c = self.data[index]
        
        input_vector = torch.tensor([a, self.plus_token, b, self.equals_token])
        #target_vector = torch.tensor([self.plus_token, b, self.equals_token, c])
        return input_vector, c



