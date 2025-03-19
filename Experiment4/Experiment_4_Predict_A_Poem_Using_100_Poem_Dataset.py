# Predict A Poem Using 100 Poem Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv

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

class PoemRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(PoemRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        out = self.fc(output[:, -1, :])                    # Use the Last Hidden State for Prediction
        return out

# Hyperparameters
embed_dim = 100
hidden_dim = 128
output_dim = vocab_size

# Initialize the Model
model = PoemRNN(vocab_size, embed_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Training loop
num_epochs = 250
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(sequences), batch_size):
        batch_seq = sequences[i:i + batch_size]
        batch_target = targets[i:i + batch_size]

        # Forward Pass
        outputs = model(batch_seq)
        loss = criterion(outputs, batch_target)

        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def generate_poem(model, seed_text, num_words = 50):
    model.eval()
    words = seed_text.split()
    with torch.no_grad():
        for _ in range(num_words):
            # Get the Last `seq_length` Words
            seq = [word_to_idx.get(word, 0) for word in words[-seq_length:]]  # Use 0 for OOV Words
            seq = torch.tensor(seq, dtype = torch.long).unsqueeze(0)
            output = model(seq)

            # Apply Softmax
            probabilities = F.softmax(output, dim = 1)

            # Sample from the Probability Distribution
            predicted_idx = torch.multinomial(probabilities, 1).item()

            words.append(idx_to_word[predicted_idx])

    return " ".join(words)

# Generate a Poem
seed_text = "I wandered lonely as a"
generated_poem = generate_poem(model, seed_text, num_words = 50)
print(generated_poem)
