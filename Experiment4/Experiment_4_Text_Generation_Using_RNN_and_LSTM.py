# Text Generation Using RNN and LSTM for 100 Poem Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import time

# Load the Dataset
text = ""
with open("/kaggle/input/poems-dataset/poems-100.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        text += " ".join(row) + " "                          # Combine All Lines into a Single Text

# Tokenize the Text into Words
tokens = text.split()

# Create a Dictionary to Map Words to Indices
word_to_idx = {}
idx_to_word = {}
vocab_size = 0

for word in tokens:
    if word not in word_to_idx:
        word_to_idx[word] = vocab_size
        idx_to_word[vocab_size] = word
        vocab_size += 1

# Convert Tokens to Indices
token_indices = [word_to_idx[word] for word in tokens]

print(f"Vocabulary Size: {vocab_size}")

# Create Sequences and Targets
seq_length = 10
sequences = []
targets = []

for i in range(len(token_indices) - seq_length):
    seq = token_indices[i:i + seq_length]
    target = token_indices[i + seq_length]
    sequences.append(seq)
    targets.append(target)

# Convert to PyTorch Tensors
sequences = torch.tensor(sequences, dtype = torch.long)
targets = torch.tensor(targets, dtype = torch.long)

# Define One-Hot Encoding for RNN Model
class OneHotRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim):
        super(OneHotRNN, self).__init__()
        self.rnn = nn.RNN(vocab_size, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.rnn(x)
        out = self.fc(output[:, -1, :])
        return out

# Define LSTM Model with Embedding Layer
class PoemLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(PoemLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        out = self.fc(output[:, -1, :])
        return out

# Hyperparameters
embed_dim = 100
hidden_dim = 128
output_dim = vocab_size

# Initialize Models
onehot_model = OneHotRNN(vocab_size, hidden_dim, output_dim)
embedding_model = PoemLSTM(vocab_size, embed_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
onehot_optimizer = optim.Adam(onehot_model.parameters(), lr = 0.001)
embedding_optimizer = optim.Adam(embedding_model.parameters(), lr = 0.001)

# Loss Tracking
onehot_losses, embedding_losses = [], []

# Training Function with Tracking
def train_model(model, optimizer, name):
    start_time = time.time()
    for epoch in range(100):
        total_loss = 0
        for i in range(0, len(sequences), 32):
            batch_seq = sequences[i:i + 32]
            batch_target = targets[i:i + 32]

            # One-Hot Encoding for OneHotRNN
            if name == "OneHotRNN":
                batch_seq = F.one_hot(batch_seq, num_classes = vocab_size).float()

            # Forward Pass
            outputs = model(batch_seq)
            loss = criterion(outputs, batch_target)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(sequences) // 32)
        if name == "OneHotRNN":
            onehot_losses.append(avg_loss)
        else:
            embedding_losses.append(avg_loss)

        print(f"{name} Epoch [{epoch+1}/100], Avg Loss: {avg_loss:.4f}")
    print(f"{name} Training Time: {time.time() - start_time:.2f}s\n")

# Poem Generation Function
def generate_poem(model, seed_text, num_words = 50, model_type = "EmbeddingLSTM"):
    model.eval()
    words = seed_text.split()
    with torch.no_grad():
        for _ in range(num_words):
            seq = [word_to_idx.get(word, 0) for word in words[-seq_length:]]
            seq = torch.tensor(seq, dtype = torch.long).unsqueeze(0)

            if model_type == "OneHotRNN":
                seq = F.one_hot(seq, num_classes = vocab_size).float()

            output = model(seq)
            probabilities = F.softmax(output, dim = 1)
            predicted_idx = torch.multinomial(probabilities, 1).item()

            words.append(idx_to_word[predicted_idx])

    return " ".join(words)

# Train Models
train_model(onehot_model, onehot_optimizer, "OneHotRNN")
train_model(embedding_model, embedding_optimizer, "EmbeddingLSTM")

# Generate Poems
seed_text = "I wandered lonely as a"
print("\nGenerated Poem (OneHotRNN):", generate_poem(onehot_model, seed_text, model_type = "OneHotRNN"))
print("\nGenerated Poem (EmbeddingLSTM):", generate_poem(embedding_model, seed_text, model_type = "EmbeddingLSTM"))
