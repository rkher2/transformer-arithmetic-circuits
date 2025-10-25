# data parameters
MODULUS = 113
TRAIN_SPLIT = 0.8

# model parameters
NUM_LAYERS = 3   # number of transformer layers
NUM_HEADS = 4   # number of attention heads
DIM_MODEL = 128   # size of embedding vectors
DIM_HEAD = DIM_MODEL // NUM_HEADS   # number of components of embedding vectors that each attention head works with
DIM_MLP = DIM_MODEL * 4 # size of multilayer perceptron

# training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
NUM_EPOCHS = 50

# file path
MODEL_SAVE_PATH = "models/modular_addition.pt"