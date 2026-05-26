import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class PhoBertClassifier:
    def __init__(self, model_path="vinai/phobert-base"):
        """
        Khởi tạo mô hình PhoBERT.
        - Lúc mới code: Dùng tạm "vinai/phobert-base" để test luồng chạy.
        - Lúc bảo vệ đề tài: Đổi model_path thành thư mục chứa mô hình bạn đã Fine-tune.
        """
        print("🚀 Đang tải mô hình ngôn ngữ PhoBERT...")
        
        # 1. Tải bộ cắt từ tiếng Việt chuẩn của PhoBERT
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        # 2. Tải kiến trúc mô hình với 7 nhãn phân loại (Theo đúng thiết kế NCKH của bạn)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=7,
            ignore_mismatched_sizes=True # Bỏ qua lỗi mismatch nếu dùng model gốc chưa train
        )
        self.model.eval() # Chuyển mô hình sang chế độ suy luận (không train)
        
        # Thiết lập chạy trên GPU nếu có, không thì CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Danh sách 7 loại văn bản theo chuẩn của bạn
        self.labels = [
            "Công văn", "Quyết định", "Thông báo", 
            "Giấy mời", "Tờ trình", "Báo cáo", "Kế hoạch"
        ]

    def predict(self, text: str) -> dict:
        """
        Nhận vào chuỗi văn bản OCR và trả về nhãn dự đoán kèm độ tự tin.
        """
        if not text or len(text.strip()) == 0:
            return {"label": "Không xác định", "confidence": 0.0}

        # Tiền xử lý: Cắt văn bản sao cho không vượt quá 256 tokens (vì đầu văn bản chứa nhiều keyword nhất)
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256, 
            padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Suy luận
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Tính toán xác suất (Softmax) và lấy nhãn có điểm cao nhất
            probs = torch.softmax(logits, dim=-1)[0]
            predicted_class_id = torch.argmax(probs).item()
            confidence = probs[predicted_class_id].item()

        return {
            "label": self.labels[predicted_class_id],
            "confidence": round(confidence * 100, 2)
        }

# Khởi tạo sẵn một instance (Singleton) để gọi nhiều lần không bị nạp lại model
phobert_engine = PhoBertClassifier()