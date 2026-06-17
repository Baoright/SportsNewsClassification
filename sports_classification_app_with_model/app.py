from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 1. Định nghĩa model LSTM
class LSTMPhoBERT(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=9, num_layers=2, dropout=0.5):
        super(LSTMPhoBERT, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        return self.fc(out)

# 2. Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPhoBERT().to(device)
model.load_state_dict(torch.load("D:/WebPhanLoai/sports_classification_app_with_model/lstm_phobert.pth", map_location=device))
model.eval()

# 3. Load PhoBERT và tokenizer
phobert = AutoModel.from_pretrained("vinai/phobert-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# 4. Hàm tạo embedding từ văn bản
def get_embedding(text):
    tokens = tokenizer.encode(text, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        features = phobert(tokens)[0]  # (batch_size, seq_len, hidden_size)
        sentence_embedding = features.mean(dim=1)  # (batch_size, hidden_size)
    return sentence_embedding

# 5. Tạo ánh xạ từ nhãn sang tên môn thể thao
label_map = {
    0: "bida",
    1: "bóng rổ",
    2: "bóng đá",
    3: "cầu lông",
    4: "cờ vua",
    5: "quần vợt",
    6: "thể thao điện tử",
    7: "điền kinh",
    8: "đua xe"
}

# 6. Flask App
app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        content = request.form["content"]
        if content.strip() == "":
            result = "Vui lòng nhập nội dung!"
        elif len(content.strip()) < 50:
            result = "Nội dung quá ngắn. Cần ít nhất 20 ký tự."
        else:
            with torch.no_grad():
                embedding = get_embedding(content)
                output = model(embedding)
                pred = torch.argmax(output, dim=1).item()
                predicted_sport = label_map.get(pred, "Không xác định")
                result = f"Phân loại môn thể thao: {predicted_sport}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
