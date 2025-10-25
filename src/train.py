import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import src.config as config
from src.data import ModularAdditionDataset
from src.model import SimpleTransformer
from src.utils import save_weights, compute_vocab_size

def train():
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # set up data
    vocab_size = compute_vocab_size(config.MODULUS)
    dataset = ModularAdditionDataset(config.MODULUS)
    
    train_size = math.floor(config.TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # make model
    model = SimpleTransformer(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    

    # make training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{config.NUM_EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # reset gradients every time
            optimizer.zero_grad()
            
            # logits has 3 dimensions: batch, seq_len, vocab_size
            logits = model(inputs)
            
            # get logits for last token position only (predicted c); shape is same as logits without seq_len dimension
            logits = logits[:, -1, :] 
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
        
        average_loss = total_loss / len(train_loader)
        
        model.eval()
        total_correct = 0
        total_samples = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
                
            logits = model(inputs)[:, -1, :] 
            predictions = torch.argmax(logits, dim=-1)
                
            total_correct = total_correct + (predictions == labels).sum().item()
            total_samples = total_samples + inputs.size(0)
                
        accuracy = (total_correct / total_samples) * 100
        print(f"epoch: {epoch+1}; loss function: {average_loss:.5f}; accuracy on test data: {accuracy:.2f}%")

    # save 
    save_weights(model, config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()