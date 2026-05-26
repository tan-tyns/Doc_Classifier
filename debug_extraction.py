import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.information_extraction import info_extractor

# Test OCR text từ giấy mời
test_text = """TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Bình Dương, ngày 01 tháng 10 năm 2014

GIẤY MỜI

Về các giải pháp nâng cao chất lượng đào tạo và nghiên cứu về Hóa học, Trường Đại học Thủ Dầu Một tổ chức Hội thảo khoa học với chủ đề: "Hóa học vì sự phát triển bền vững"

Kính mời:
Các đại biểu, quý thầy cô giáo...

Nội dung chính của hội thảo:
1. Các vấn đề đặt ra...
"""

print("=" * 70)
print("📄 OCR TEXT INPUT:")
print("=" * 70)
print(test_text)
print("\n" + "=" * 70)
print("🔍 EXTRACTION RESULTS:")
print("=" * 70)

# Extract all info
info = info_extractor.extract_all_info(test_text)

print(f"\n📅 Ngày tháng năm      : {info.get('ngay_thang_nam')}")
print(f"📄 Loại văn bản        : {info.get('loai_van_ban')}")
print(f"🔢 Số hiệu             : {info.get('so_hieu')}")
print(f"🏙️  Nơi ban hành        : {info.get('thanh_pho')}")

print(f"\n📌 TITLE (tieu_de):")
print(f"   {info.get('tieu_de')}")
print(f"\n📌 SUMMARY (trich_yeu):")
print(f"   {info.get('trich_yeu')}")

print(f"\n📖 MAIN CONTENT (noi_dung):")
print(f"   {info.get('noi_dung')[:300]}...")

