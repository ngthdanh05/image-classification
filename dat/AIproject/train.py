import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import SimpleCNN
from data_loader import get_dataloaders

# ====================== Cài đặt ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang dùng: {device} (RTX 3050 4GB)")

batch_size = 16          # giảm xuống 16 nếu hết VRAM
epochs = 15
learning_rate = 0.001

train_loader, val_loader, classes = get_dataloaders(batch_size=batch_size)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Optimizer nó làm gì.

# ====================== Huấn luyện ======================
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {100*correct/total:.2f}%")

# Lưu model
torch.save(model.state_dict(), "cat_dog_model.pth")
print("Đã lưu model: cat_dog_model.pth")