import streamlit as st
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
  )
    self.second = nn.Sequential(
      nn.Flatten(),
      nn.Linear(16 * 14 * 14, 64),
      nn.ReLU(),
      nn.Linear(64,10)
  )
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

CLASS_NAMES = {
    0: "T-shirt/Top",
    1: "Trousers",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandals",
    6: "Shirt",
    7: "Sneakers",
    8: "Bag",
    9: "Boots"
}

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = CheckImage()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

st.set_page_config(page_title="Fasion Classifier", layout="centered")
st.title("Fasion Classifier")

model, device = load_model()

uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Загруженное изображение")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    st.write("Результат:", CLASS_NAMES[pred])