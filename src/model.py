import torch
import torch.nn as nn
from src.config import NUM_LAYERS, NUM_HEADS, DIM_MODEL, DIM_MLP, MODULUS, BATCH_SIZE

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, DIM_MODEL)
        self.positional_embedding = nn.Embedding(max_seq_len, DIM_MODEL)

        self.layers = []
        for i in range(NUM_LAYERS):
            layer = nn.TransformerEncoderLayer(
                d_model=DIM_MODEL,
                nhead=NUM_HEADS,
                dim_feedforward=DIM_MLP,
                dropout=0.1,
                batch_first=True)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

        self.final = nn.LayerNorm(DIM_MODEL) # normalize
        self.output_head = nn.Linear(DIM_MODEL, vocab_size) # convert to x 

        # x has dimensions <batch_size, sequence_length>
    def forward(self, x):
        sequence_length = x.shape[1]

        positions = torch.arange(0, sequence_length, device=x.device)
        positions = positions.reshape(1, -1)
        positional_embedding = self.positional_embedding(positions) 
        token_embedding = self.token_embedding(x) 
        
        x = token_embedding + positional_embedding
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(sequence_length, device=x.device)
        
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        x = self.final(x)
        logits = self.output_head(x)
        
        return logits