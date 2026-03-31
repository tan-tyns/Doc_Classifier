import os
import pandas as pd
from engine.ocr_core import AdminOCR

# Cấu hình
POPPLER_PATH = r"D:\kooper\poppler-25.12.0\Library\bin" # Thay bằng đường dẫn của bạn
RAW_DATA_DIR = "data/raw"
OUTPUT_CSV = "data/dataset_final.csv"

def main():
    engine = AdminOCR()
    dataset = []

    # Quét các thư mục nhãn (quyet_dinh, ke_hoach...)
    labels = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

    for label in labels:
        folder_path = os.path.join(RAW_DATA_DIR, label)
        print(f"\n📂 Đang lùa data nhãn: {label.upper()}")
        
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                file_path = os.path.join(folder_path, file_name)
                try:
                    text = engine.process_any_file(file_path, poppler_path=POPPLER_PATH)
                    if len(text.strip()) > 20:
                        dataset.append({"text": text, "label": label})
                        print(f"  ✅ OK: {file_name}")
                except Exception as e:
                    print(f"  ❌ LỖI {file_name}: {e}")

    # Lưu kết quả
    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 Xong! Đã tạo file {OUTPUT_CSV} với {len(df)} mẫu dữ liệu.")

if __name__ == "__main__":
    main()