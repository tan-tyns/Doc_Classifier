"""
Test OCR Correction Engine
===========================
Kiểm tra xem OCR Correction Engine hoạt động tốt chưa.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.ocr_correction import ocr_corrector
from app.core.information_extraction import info_extractor


# Test cases: Các đoạn text có lỗi OCR tiêu biểu
test_cases = [
    {
        "name": "Test 1: Lỗi 'tháng' → 'thảng'",
        "input": "Bình Dương, ngày 01 thảng 10 năm 2014",
        "expected": "Bình Dương, ngày 01 tháng 10 năm 2014"
    },
    {
        "name": "Test 2: Lỗi 'tháng' → 'thạng'", 
        "input": "Hà Nội, ngày 15 thạng 05 năm 2023",
        "expected": "Hà Nội, ngày 15 tháng 05 năm 2023"
    },
    {
        "name": "Test 3: Loại văn bản bị sai",
        "input": "GIÁY MỜI / GIấy MỜI / GIAS MOI",
        "expected": "GIẤY MỜI"
    },
    {
        "name": "Test 4: Document type keywords",
        "input": "Công Văn về việc phê duyệt quyêt định",
        "expected": "Công Văn về việc phê duyệt quyết định"
    },
    {
        "name": "Test 5: Full document header",
        "input": """TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Bình Dương, ngày 01 thảng 10 năm 2014

GIÁY MỜI

Về các giải pháp nâng cao chất lượng đào tạo""",
        "expected": "Should fix 'thảng' and 'GIÁY'"
    }
]


def main():
    print("=" * 80)
    print("🧪 TEST OCR CORRECTION ENGINE".center(80))
    print("=" * 80)
    
    if ocr_corrector is None:
        print("❌ OCR Correction Engine không được khởi tạo!")
        return
    
    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"📋 {test['name']}")
        print(f"{'='*80}")
        print(f"\n🔴 INPUT OCR:")
        print(f"  {test['input']}")
        
        # Bước 1: Sửa lỗi OCR cơ bản
        print(f"\n🟡 Bước 1: Sửa lỗi OCR phổ biến...")
        corrected = ocr_corrector.correct_ocr_text(test['input'])
        print(f"  ✅ {corrected}")
        
        # Bước 2: Cải thiện bằng context
        print(f"\n🟡 Bước 2: Cải thiện bằng context (nếu cần)...")
        try:
            enhanced = ocr_corrector.enhance_text_context(corrected)
            print(f"  ✅ {enhanced}")
        except Exception as e:
            print(f"  ⚠️ Bước 2 bỏ qua (có lỗi): {e}")
            enhanced = corrected
        
        # Bước 3: Trích xuất thông tin từ text đã sửa
        if test['name'].startswith("Test 5"):
            print(f"\n🟡 Bước 3: Trích xuất thông tin cơ bản...")
            try:
                info = info_extractor.extract_date(enhanced)
                print(f"  📅 Ngày tháng năm: {info}")
            except Exception as e:
                print(f"  ⚠️ Không thể trích xuất: {e}")
        
        print(f"\n✅ EXPECTED: {test['expected']}")


if __name__ == "__main__":
    main()
