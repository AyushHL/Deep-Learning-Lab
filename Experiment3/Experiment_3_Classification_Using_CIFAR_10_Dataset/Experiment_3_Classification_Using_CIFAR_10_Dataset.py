# Classifiction Using CIFAR-10 Dataset

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from os.path import join
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, TensorDataset

def set_seed(seed):
    # Set the Seed for Reproducibility
    random.seed(seed)                              # Python Random Module
    np.random.seed(seed)                           # Numpy Random Module
    torch.manual_seed(seed)                        # PyTorch CPU
    torch.cuda.manual_seed(seed)                   # PyTorch GPU
    torch.cuda.manual_seed_all(seed)               # For Multi-GPU Setups
    torch.backends.cudnn.deterministic = True      # Ensure Deterministic Algorithms
    torch.backends.cudnn.benchmark = False         # Disable Non-Deterministic Algorithms

# Set the Seed
seed_value = 42             # Example Seed Value
set_seed(seed_value)

num_classes = 10
learning_rate = 0.003

# Dimensions of Image
image = Image.open("/kaggle/input/cifar10-classification-image/cifar10/train/airplane/0001.png")
array = np.array(image)
height, width, channels = array.shape
print(f"Width: {width}, Height: {height}, Channels: {channels}")

# Check If CUDA is Available, Else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {device}')

# Paths to Datasets
input_train_dir = "/kaggle/input/cifar10-classification-image/cifar10/train"
input_test_dir = "/kaggle/input/cifar10-classification-image/cifar10/test"

# Transformation for Training (With Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),                                                        # CIFAR-10 is 32x32
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261])  # CIFAR-10 Mean/Std
])

# Transformation for Testing (No Augmentation)
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261])
])

# Initialize Datasets
train_dataset = []
train_labels = []
test_dataset = []
test_labels = []
label_map = {name: idx for idx, name in enumerate(os.listdir(input_train_dir))}    # Dynamically Map Class Names

# Load Training Data
for category in os.listdir(input_train_dir):
    path = join(input_train_dir, category)
    for image_file in os.listdir(path):
        if not image_file.endswith('.png'):                         # CIFAR-10 Uses PNG Images
            continue
        image_path = join(path, image_file)
        try:
            image = Image.open(image_path).convert("RGB")           # Convert to 3-Channel RGB
            image_tensor = train_transform(image).to(device)
            train_dataset.append(image_tensor)
            train_labels.append(label_map[category])
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

# Load Test Data
for category in os.listdir(input_test_dir):
    path = join(input_test_dir, category)
    for image_file in os.listdir(path):
        if not image_file.endswith('.png'):
            continue
        image_path = join(path, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = test_transform(image).to(device)
            test_dataset.append(image_tensor)
            test_labels.append(label_map[category])
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

print(f"Training Set: {len(train_dataset)} images, Test Set: {len(test_dataset)} images")
print(f"Sample Tensor Shape: {train_dataset[0].shape} (should be torch.Size([3, 32, 32]))")

split = 0.8

# Convert Lists to Tensors
train_dataset = torch.stack(train_dataset)
train_labels = torch.tensor(train_labels)

# Define Split Ratio (E.g., 80% Train, 20% Validation)
train_size = int(split * len(train_dataset))
val_size = len(train_dataset) - train_size

# Use random_split to Split Dataset
train_data, val_data = random_split(list(zip(train_dataset, train_labels)), [train_size, val_size])

# Convert Back to Separate Lists
train_dataset, train_labels = zip(*train_data)
val_dataset, val_labels = zip(*val_data)

# Convert Back to PyTorch Tensors
train_dataset = list(train_dataset)
train_labels = list(train_labels)
val_dataset = list(val_dataset)
val_labels = list(val_labels)

print(f"Training Size: {len(train_dataset)}, Validation Size: {len(val_dataset)}")

# Creating Data Loaders

val_dataset = TensorDataset(torch.stack(val_dataset), torch.tensor(val_labels))
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = True)

train_dataset = TensorDataset(torch.stack(train_dataset), torch.tensor(train_labels))
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)

test_dataset = TensorDataset(torch.stack(test_dataset), torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

# Check Batch Shape
for images, labels in train_loader:
    print(f"Batch Tensor Shape: {images.shape}, Labels Shape: {labels.shape}")
    break

# Function to Initialize Weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity = "relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Preactivation Block without Skip Connections
class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, dropout_prob = 0.3):
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.dropout1 = nn.Dropout2d(p = dropout_prob)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.dropout2 = nn.Dropout2d(p = dropout_prob)

        self.activation = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.dropout1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.dropout2(out)

        return out

# ResNet-10 Model without Skip Connections
class ResNet10(nn.Module):
    def __init__(self, num_classes = 10, dropout_prob = 0.3):
        super(ResNet10, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(64, 2, stride = 1, dropout_prob = dropout_prob)
        self.layer2 = self._make_layer(128, 2, stride = 2, dropout_prob = dropout_prob)
        self.layer3 = self._make_layer(256, 2, stride = 2, dropout_prob = dropout_prob)
        self.layer4 = self._make_layer(512, 2, stride = 2, dropout_prob = dropout_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Apply Weight Initialization
        initialize_weights(self)

    def _make_layer(self, out_channels, blocks, stride, dropout_prob):
        layers = []
        layers.append(PreActBlock(self.in_channels, out_channels, stride, dropout_prob))
        self.in_channels = out_channels                                                           # Update for Next Blocks

        for _ in range(1, blocks):
            layers.append(PreActBlock(out_channels, out_channels, dropout_prob = dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs = 1000, save_req = False):
    best_accuracy = 0.0                                             # Store the Best Accuracy
    best_loss = float("inf")                                        # Store the Best Loss

    for epoch in range(num_epochs):
        model.train()                                               # Set the Model to Training Mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Training Loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute Accuracy
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (correct / total) * 100

        # Validation Loop
        model.eval()                                                  # Set the Model to Evaluation Mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():                                         # No Gradients Needed During Evaluation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward Pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predictions = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predictions == labels).sum().item()

        val_accuracy = (val_correct / val_total) * 100
        val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%", end = ", ")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the Best Model Based on Validation Accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_loss = val_loss
            if save_req:
                torch.save(model.state_dict(), "bestModel.pth")
                print(f"✅ Best Model Saved with Validation Accuracy: {best_accuracy:.2f}%")

    return best_accuracy, best_loss

# Training with Best Configuration
model = ResNet10(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
best_acc, best_loss = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs = 500, save_req = True)

print(f"Final Training -> Best Accuracy: {best_acc:.2f}, Loss: {best_loss:.2f} with Best Configuration")

model = ResNet10(num_classes = num_classes).to(device)
model.load_state_dict(torch.load("bestModel.pth"))
model.eval()                                                     # Set Model to Evaluation Mode

# Initialize Variables to Calculate Loss and Accuracy
total_loss = 0.0
correct = 0
total = 0

# Use no_grad to Disable Gradient Computation for Inference
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward Pass
        outputs = model(images)

        # Calculate Loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Calculate Accuracy
        _, predicted = torch.max(outputs, 1)                     # Get Predicted Class Indices
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate Average Loss and Accuracy
avg_loss = total_loss / len(test_loader)
accuracy = (correct / total) * 100                               # Convert to Percentage

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
