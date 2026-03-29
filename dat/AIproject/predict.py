import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    class_names = ['mèo', 'chó']
    print(f"✅ Dự đoán: Đây là con **{class_names[predicted.item()]}**")
    print(f"Độ tin cậy: {prob[0][predicted.item()].item()*100:.1f}%")

# Ví dụ dùng
if __name__ == "__main__":
    predict_image("path/to/your/test_image.jpg")   # ← thay đường dẫn ảnh của bạn