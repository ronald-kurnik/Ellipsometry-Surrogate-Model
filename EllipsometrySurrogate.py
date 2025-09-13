import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Simplified forward model (all real for demo)
def ellipsometry_forward(wl, d, n_f=1.46, n_s=3.5, theta=np.deg2rad(70)):
    N0 = 1.0
    N1 = n_f
    N2 = n_s
    
    theta1 = np.arcsin((N0 / N1) * np.sin(theta))
    theta2 = np.arcsin((N1 / N2) * np.sin(theta1))
    
    r01p = (N1 * np.cos(theta) - N0 * np.cos(theta1)) / (N1 * np.cos(theta) + N0 * np.cos(theta1))
    r01s = (N0 * np.cos(theta) - N1 * np.cos(theta1)) / (N0 * np.cos(theta) + N1 * np.cos(theta1))
    r12p = (N2 * np.cos(theta1) - N1 * np.cos(theta2)) / (N2 * np.cos(theta1) + N1 * np.cos(theta2))
    r12s = (N1 * np.cos(theta1) - N2 * np.cos(theta2)) / (N1 * np.cos(theta1) + N2 * np.cos(theta2))
    
    beta = 2 * np.pi * d / wl * N1 * np.cos(theta1)
    
    exp_term = np.exp(-2j * beta)  # Note: Use -2 * 1j * beta if parser issues
    
    rp = (r01p + r12p * exp_term) / (1 + r01p * r12p * exp_term)
    rs = (r01s + r12s * exp_term) / (1 + r01s * r12s * exp_term)
    
    rho = rp / rs
    psi = np.arctan(np.abs(rho)) * 180 / np.pi
    delta = np.angle(rho) * 180 / np.pi
    
    return psi, delta

# Generate training data
wavelengths = np.linspace(400, 800, 50)
thicknesses = np.linspace(50, 150, 100)

X_train = []
y_train = []
for d in thicknesses:
    for wl in wavelengths:
        X_train.append([wl, d])
        psi, delta = ellipsometry_forward(wl, d)
        y_train.append([psi, delta])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
y_train_norm = (y_train - y_mean) / y_std

# Tensors
X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)

# NN model
class SurrogateNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SurrogateNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')

# Test on d=100 nm
test_d = 100.0
X_test = np.array([[wl, test_d] for wl in wavelengths])
X_test_norm = (X_test - X_mean) / X_std
X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)

with torch.no_grad():
    y_pred_norm = model(X_test_t)
    y_pred = y_pred_norm.numpy() * y_std + y_mean

y_true = np.array([ellipsometry_forward(wl, test_d) for wl in wavelengths])

# MAE
mae_psi = np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))
mae_delta = np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))
print(f'MAE Psi: {mae_psi:.4f} degrees')
print(f'MAE Delta: {mae_delta:.4f} degrees')

# Assuming wavelengths, y_pred, and y_true are defined from the original code
plt.figure(figsize=(10, 5))

# Plot Psi
plt.subplot(1, 2, 1)
plt.plot(wavelengths, y_pred[:, 0], label='Predicted Psi', color='#1f77b4')
plt.plot(wavelengths, y_true[:, 0], label='True Psi', color='#ff7f0e')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Psi (degrees)')
plt.title('Psi vs. Wavelength (d = 100 nm)')
plt.legend()
plt.grid()

# Plot Delta
plt.subplot(1, 2, 2)
plt.plot(wavelengths, y_pred[:, 1], label='Predicted Delta', color='#1f77b4')
plt.plot(wavelengths, y_true[:, 1], label='True Delta', color='#ff7f0e')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Delta (degrees)')
plt.title('Delta vs. Wavelength (d = 100 nm)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()