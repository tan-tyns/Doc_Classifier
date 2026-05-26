"""
OCR Correction Engine using PhoBERT
====================================
Sử dụng PhoBERT để cải thiện độ chính xác của text được VietOCR quét.

Các phương pháp được sử dụng:
1. Masked Language Modeling (MLM) - Sửa từ bị sai
2. Token Classification - Nhập các entity quan trọng
3. Context-aware correction - Sửa lỗi dựa trên context xung quanh
"""

import torch
import re
from typing import List, Dict, Tuple
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)
from difflib import SequenceMatcher
import unicodedata


class OCRCorrectionEngine:
    """
    Engine để sửa lỗi OCR tiếng Việt sử dụng PhoBERT.
    """
    
    def __init__(self, model_name: str = "vinai/phobert-base"):
        """
        Khởi tạo OCR Correction Engine.
        
        Args:
            model_name: Tên model PhoBERT từ Hugging Face
        """
        print("🚀 Đang tải PhoBERT Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("🚀 Đang tải PhoBERT MLM Model...")
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # Thiết lập device (GPU nếu có)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlm_model.to(self.device)
        self.mlm_model.eval()
        
        # Pipeline cho Fill-Mask
        self.fill_mask_pipeline = pipeline(
            "fill-mask",
            model=self.mlm_model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Dictionary các lỗi OCR phổ biến tiếng Việt
        self.common_ocr_errors = {
            # Lỗi dấu/accent
            "thảng": "tháng",
            "thạng": "tháng", 
            "théng": "tháng",
            "thăng": "tháng",
            "tháng": "tháng",  # Chính xác
            
            # Lỗi chữ tương tự
            "rồ": "để",
            "sẽ": "sẽ",
            "từ": "từ",
            
            # Các kí tự bị nhầm
            "l": "I",  # Chữ L nhỏ vs I lớn
            "0": "O",  # Số 0 vs chữ O
            "1": "I",  # Số 1 vs chữ I
        }
        
        # Các từ khóa quan trọng cần đảm bảo chính xác
        self.critical_keywords = [
            "công văn", "cong van",
            "quyết định", "quyet dinh", 
            "thông báo", "thong bao",
            "giấy mời", "giay moi",
            "tờ trình", "to trinh",
            "báo cáo", "bao cao",
            "kế hoạch", "ke hoach",
            "tháng", "năm", "ngày",
            "nơi", "ban hành", "ký", "duyệt"
        ]

    def _normalize_text(self, text: str) -> str:
        """Normalize text trước khi xử lý."""
        text = unicodedata.normalize('NFC', text)
        return text.strip()

    def correct_word_candidates(self, word: str, num_candidates: int = 5) -> List[Tuple[str, float]]:
        """
        Sử dụng PhoBERT MLM để tìm các từ có thể thay thế cho từ bị sai.
        
        Args:
            word: Từ cần sửa
            num_candidates: Số lượng gợi ý trả về
            
        Returns:
            Danh sách (từ_sửa, điểm_tin_cậy)
        """
        try:
            # Tạo masked sentence
            masked_text = f"Văn bản {word} được duyệt [MASK] tháng năm."
            
            # Sử dụng fill-mask pipeline
            results = self.fill_mask_pipeline(masked_text, top_k=num_candidates)
            
            candidates = [
                (r['token_str'].strip(), r['score']) 
                for r in results
            ]
            return candidates
        except Exception as e:
            print(f"⚠️ Lỗi trong correc_word_candidates: {e}")
            return [(word, 0.0)]

    def correct_ocr_text(self, text: str) -> str:
        """
        Sửa toàn bộ text OCR.
        
        Args:
            text: Text từ OCR
            
        Returns:
            Text đã được sửa
        """
        text = self._normalize_text(text)
        corrected_text = text
        
        # 1. Sửa các lỗi OCR phổ biến
        for error, correct in self.common_ocr_errors.items():
            # Tìm kiếm case-insensitive
            pattern = re.compile(re.escape(error), re.IGNORECASE)
            corrected_text = pattern.sub(correct, corrected_text)
        
        # 2. Xử lý các trường hợp đặc biệt: Ngày tháng năm
        corrected_text = self._correct_date_numbers(corrected_text)
        
        # 3. Sửa các từ khóa quan trọng
        corrected_text = self._correct_critical_keywords(corrected_text)
        
        return corrected_text

    def _correct_date_numbers(self, text: str) -> str:
        """Sửa các số trong biểu thức ngày tháng năm."""
        # Pattern: ngày/tháng/năm
        def replace_date(match):
            full_match = match.group(0)
            # Sửa các chữ số bị nhầm (l -> 1, O -> 0, etc.)
            result = full_match.replace('l', '1').replace('O', '0').replace('o', '0')
            return result
        
        # Tìm pattern ngày/tháng/năm (có dấu hoặc không dấu)
        pattern = r'(?:ngày|tháng|thảng|thạng|nam|năm|nă\w+)\s+[\dloO]+'
        text = re.sub(pattern, replace_date, text, flags=re.IGNORECASE)
        
        return text

    def _correct_critical_keywords(self, text: str) -> str:
        """Sửa các từ khóa quan trọng bằng phương pháp matching."""
        corrected_text = text
        
        for keyword in self.critical_keywords:
            # Tìm các từ gần giống keyword (similarity > 0.7)
            words = corrected_text.split()
            for i, word in enumerate(words):
                similarity = SequenceMatcher(None, word.lower(), keyword.lower()).ratio()
                if 0.6 < similarity < 1.0:  # Từ gần giống nhưng không hoàn toàn đúng
                    words[i] = keyword
                    print(f"  ✏️ Sửa '{word}' → '{keyword}' (Tương tự: {similarity:.2%})")
            
            corrected_text = " ".join(words)
        
        return corrected_text

    def enhance_text_context(self, text: str, context_window: int = 128) -> str:
        """
        Sử dụng PhoBERT để cải thiện text dựa trên context.
        
        Args:
            text: Text cần cải thiện
            context_window: Số token xung quanh để xem xét
            
        Returns:
            Text đã được cải thiện
        """
        text = self._normalize_text(text)
        
        # Cắt text thành các chunks
        words = text.split()
        enhanced_words = []
        
        for i, word in enumerate(words):
            # Lấy context xung quanh từ này
            start = max(0, i - context_window // 2)
            end = min(len(words), i + context_window // 2)
            context_words = words[start:end]
            context_text = " ".join(context_words)
            
            # Nếu từ có vẻ bị sai (có ký tự lạ), cố gắng sửa
            if self._looks_like_ocr_error(word):
                try:
                    # Tạo masked text
                    masked_text = context_text.replace(word, "[MASK]")
                    results = self.fill_mask_pipeline(masked_text, top_k=3)
                    
                    if results:
                        # Lấy từ được dự đoán cao nhất
                        best_candidate = results[0]['token_str'].strip()
                        if results[0]['score'] > 0.3:  # Ngưỡng tin cậy
                            enhanced_words.append(best_candidate)
                            print(f"  💡 Sửa bằng context: '{word}' → '{best_candidate}'")
                            continue
                except Exception as e:
                    print(f"  ⚠️ Lỗi enhance context: {e}")
            
            enhanced_words.append(word)
        
        return " ".join(enhanced_words)

    def _looks_like_ocr_error(self, word: str) -> bool:
        """
        Kiểm tra xem từ này có vẻ như là lỗi OCR không.
        """
        # Từ quá ngắn hoặc có ký tự lạ
        if len(word) < 2:
            return True
        
        # Chứa chữ số và chữ lẫn lộn (trừ các từ hợp lệ)
        if re.search(r'[loO0]', word):
            return True
        
        # Chứa ký tự không hợp lệ
        if not re.match(r'^[a-zA-Zàáãạảăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\-\s]+$', word):
            return True
        
        return False

    def batch_correct_texts(self, texts: List[str]) -> List[str]:
        """
        Sửa nhiều text một lúc.
        
        Args:
            texts: Danh sách các text cần sửa
            
        Returns:
            Danh sách các text đã được sửa
        """
        corrected = []
        for text in texts:
            corrected.append(self.correct_ocr_text(text))
        return corrected


# Khởi tạo instance Singleton
try:
    ocr_corrector = OCRCorrectionEngine()
    print("✅ OCR Correction Engine khởi tạo thành công!")
except Exception as e:
    print(f"❌ Lỗi khởi tạo OCR Correction Engine: {e}")
    ocr_corrector = None
