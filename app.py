import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
from model import LightWeightCNN  # 呼叫你寫的 AI 大腦結構

# 1. 定義類別 (必須與資料夾的英文字母順序完全一致)
classes = ['電容 (Capacitor)', 'IC 晶片 (IC)', '發光二極體 (LED)', '電阻 (Resistor)', '電晶體 (Transistor)']

# 2. 喚醒 GPU 並載入剛剛訓練好的權重 (大腦記憶)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightWeightCNN(num_classes=5).to(device)
model.load_state_dict(torch.load('component_model.pth', map_location=device))
model.eval()  # 切換成推論(實戰)模式

# 3. 影像前處理 (必須與訓練時一模一樣，把手機照片變成 128x128 矩陣)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 4. 定義預測函數 (接收網頁傳來的手機照片，回傳預測結果)
def predict_component(img):
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():  # 實戰時不需要計算梯度，節省效能
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # 將機率轉換為字典格式，讓 Gradio 畫出漂亮的長條圖
    result_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    return result_dict


# 5. 打造充滿科技感的 Web UI
demo = gr.Interface(
    fn=predict_component,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"]),  # 允許上傳圖片或開啟手機相機
    outputs=gr.Label(num_top_classes=3),  # 顯示前 3 名最有可能的答案
    title="🔬 實驗室 AI 視覺助理",
    description="請用手機拍下單一電子元件（電阻、電容、IC、電晶體、LED），AI 將即時為您辨識！"
)

if __name__ == "__main__":
    # share=True 是最強魔法！它會產生一個公開網址，讓你用手機的 4G/5G 就能連線到筆電
    demo.launch(share=True)