import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class DocumentClassifier:
    def __init__(self, model_path="doc_classifier_svm.pkl"):
        self.model_path = model_path
        self.is_trained = False
        
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2)),
            ('clf', LinearSVC(C=1.0, random_state=42))
        ])

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            print(f"[NLP] Đã load mô hình phân loại từ {self.model_path}")

    def rule_based_predict(self, text: str) -> str:
        text = text.lower()
        
        # BỘ TỪ ĐIỂN CHỈ TẬP TRUNG VÀO NGHIỆP VỤ (LĨNH VỰC/PHÒNG BAN)
        categories = {
            "PHÒNG ĐÀO TẠO": ["sinh viên", "học kỳ", "tín chỉ", "tốt nghiệp", "môn học", "thời khóa biểu", "điểm thi", "học vụ", "giảng viên", "đào tạo", "tuyển sinh", "giáo trình"],
            "TÀI CHÍNH - KẾ TOÁN": ["học phí", "kế toán", "hóa đơn", "thanh toán", "tài khoản", "ngân hàng", "lợi nhuận", "cổ đông", "tài chính", "ngân sách", "kinh phí", "báo cáo thuế", "phân phối lợi nhuận"],
            "NHÂN SỰ - TỔ CHỨC": ["bổ nhiệm", "miễn nhiệm", "nhân viên", "hợp đồng lao động", "lương", "chấm công", "nghỉ việc", "công tác", "kỷ luật", "khen thưởng", "điều động"],
            "QUẢN TRỊ - HÀNH CHÍNH": ["hội trường", "đại hội", "an ninh", "bảo vệ", "cơ sở vật chất", "văn phòng phẩm", "công tác chuẩn bị", "bảo trì", "vệ sinh", "ban tổ chức"],
            "KỸ THUẬT - DỰ ÁN": ["thi công", "nghiệm thu", "kỹ thuật", "dự án", "cấp nước", "xây dựng", "vật tư", "hệ thống", "công trình", "nâng cấp", "đấu nối", "đường ống", "bản vẽ", "tiến độ"],
            "KINH DOANH - KHÁCH HÀNG": ["khách hàng", "sản phẩm", "nhãn mác", "thị trường", "doanh thu", "khuyến mãi", "đại lý", "bao bì", "phân phối", "xuất khẩu", "đóng gói"]
        }
        
        scores = {cat: 0 for cat in categories}
        for cat, keywords in categories.items():
            for kw in keywords:
                # Đếm số lần xuất hiện của từ khóa
                scores[cat] += len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
        
        best_cat = max(scores, key=scores.get)
        
        # Ngưỡng (Threshold): Nếu văn bản quá ngắn hoặc không có từ khóa nào rõ ràng
        if scores[best_cat] == 0:
            return "VĂN BẢN CHUNG"
            
        return best_cat

    def predict(self, text: str) -> str:
        if not text or len(text.strip()) < 10:
            return "KHÔNG XÁC ĐỊNH"
            
        if self.is_trained:
            predicted_label = self.model.predict([text])[0]
            return predicted_label
        else:
            return self.rule_based_predict(text)

    def train_model(self, X_train: list, y_train: list):
        print("[NLP] Bắt đầu huấn luyện mô hình Machine Learning...")
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        self.is_trained = True
        print(f"[NLP] Huấn luyện xong! Đã lưu tại {self.model_path}")
        y_pred = self.model.predict(X_train)
        print("\n📊 BÁO CÁO ĐỘ CHÍNH XÁC (ACCURACY REPORT):")
        print(classification_report(y_train, y_pred))