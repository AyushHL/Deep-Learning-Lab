# Fundamentals of Tensor Operations using PyTorch and NumPy

import torch
import numpy as np

print("Creating Tensors")
# 1D, 2D, 3D tensors
tensor_1d = torch.tensor([1, 2, 3])
tensor_2d = torch.tensor([[1, 2], [3, 4]])
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
np_1d = np.array([1, 2, 3])
np_2d = np.array([[1, 2], [3, 4]])
np_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"1D PyTorch: {tensor_1d}")
print(f"2D PyTorch: \n{tensor_2d}")
print(f"3D PyTorch: \n{tensor_3d}")
print(f"1D NumPy: {np_1d}")
print(f"2D NumPy: \n{np_2d}")
print(f"3D NumPy: \n{np_3d}")

print("Basic Operations")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"Addition: {a + b}")
print(f"Subtraction: {a - b}")
print(f"Multiplication: {a * b}")
print(f"Division: {a / b}")

print("Dot Product and Matrix Multiplication")
vec1 = torch.tensor([1, 2, 3])
vec2 = torch.tensor([4, 5, 6])
dot = torch.dot(vec1, vec2)
print(f"Dot Product: {dot}")
mat1 = torch.tensor([[1, 2], [3, 4]])
mat2 = torch.tensor([[5, 6], [7, 8]])
matmul = torch.matmul(mat1, mat2)
print(f"Matrix Multiplication:\n{matmul}")

print("Indexing and Slicing")
tensor = torch.tensor([[10, 20, 30], [40, 50, 60]])
print(f"Original Tensor:\n{tensor}")
print(f"Element [1,2]: {tensor[1, 2]}")
print(f"First Row: {tensor[0]}")
print(f"Boolean Masking > 30: {tensor[tensor > 30]}")
print(f"Extracted Subtensor: {tensor[:, 1:]}")

print("Reshaping Tensors")
x = torch.arange(6)
print(f"Original x: {x}")
print(f"x.view(2, 3):\n{x.view(2, 3)}")
print(f"x.reshape(2, 3):\n{x.reshape(2, 3)}")
y = torch.tensor([[1, 2], [3, 4]])
print(f"Unsqueeze y (add dim 0):\n{y.unsqueeze(0)}")
print(f"Squeeze y (remove dim):\n{y.unsqueeze(0).squeeze()}")

# NumPy Comparison
np_x = np.arange(6)
print(f"NumPy Reshape:\n{np_x.reshape(2, 3)}")

print("Broadcasting")
a = torch.tensor([[1], [2], [3]])
b = torch.tensor([10, 20, 30])
print(f"a shape: {a.shape}, b shape: {b.shape}")
print(f"Broadcasted Addition:\n{a + b}")

print("In-place vs Out-of-place Operations")
t = torch.tensor([1.0, 2.0, 3.0])
print(f"Original t: {t}")

# Out-of-place
t_add = t + 2
print(f"t + 2 (Out-of-place): {t_add}")
print(f"t after out-of-place: {t}")

# In-place
t.add_(2)
print(f"t after add_ (In-place): {t}")
