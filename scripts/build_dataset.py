import sys
import os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import concurrent.futures
import re # ✅ IMPORT THƯ VIỆN REGEX ĐỂ BÓC TÁCH THÔNG TIN

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from app.core.preprocess_image import preprocess_image_pro # Nhớ sửa thành preprocess.py nếu bạn đã đổi tên
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

print("🚀 Khởi tạo AI (PaddleOCR & VietOCR)...")
det_model = PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=False, show_log=False)
config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
rec_model = Predictor(config)

dummy_img = Image.new('RGB', (100, 32), color=(255, 255, 255))
_ = rec_model.predict(dummy_img)

# =================================================================
# ✅ HÀM MỚI: DÙNG REGEX ĐỂ BÓC TÁCH THÔNG TIN TỪ VĂN BẢN OCR
# =================================================================
def extract_metadata(text):
    so_hieu = ""
    ngay_ban_hanh = ""
    trich_yeu = ""

    # 1. Bóc tách Số/Ký hiệu (Tìm các chuỗi đứng sau chữ "Số:" hoặc "Số")
    match_so = re.search(r'(?i)Số:\s*([^\n]+)', text)
    if not match_so:
        match_so = re.search(r'(?i)Số\s+([0-9a-zA-Z/A-Z\-]+)', text)
    if match_so:
        so_hieu = match_so.group(1).strip()

   # 2. Bóc tách Ngày ban hành (Tìm cả 2 mẫu: "ngày ... tháng ... năm" HOẶC "ngày dd/mm/yyyy")
    match_ngay = re.search(r'(?i)ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', text)
    if not match_ngay: # Nếu không thấy chữ "tháng", thử tìm mẫu "27/11/2018"
        match_ngay = re.search(r'(?i)ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})', text)
        
    if match_ngay:
        ngay_ban_hanh = f"{match_ngay.group(1)}/{match_ngay.group(2)}/{match_ngay.group(3)}"

    # 3. Bóc tách Trích yếu (Tìm phần chữ đứng sau "V/v", "V/v:", hoặc "Về việc")
    match_vv = re.search(r'(?i)(?:V/v|Về việc)[:\s]+([^\n]+)', text)
    if match_vv:
        trich_yeu = match_vv.group(1).strip()

    return so_hieu, ngay_ban_hanh, trich_yeu


def process_single_crop(raw_box, gray_processed):
    try:
        pts = np.array(raw_box, dtype=np.float32).reshape(4, 2)
        w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        if w <= 0 or h <= 0: return ""
        
        dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        crop = cv2.warpPerspective(gray_processed, M, (w, h))
        if h > w * 1.2: crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        crop_padded = cv2.copyMakeBorder(crop, 4, 4, 8, 8, cv2.BORDER_CONSTANT, value=255)
        new_w = max(1, min(int(crop_padded.shape[1] * (32 / crop_padded.shape[0])), 512)) 
        crop_resized = cv2.resize(crop_padded, (new_w, 32), interpolation=cv2.INTER_CUBIC)
        
        return rec_model.predict(Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2RGB)))
    except: return ""

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return ""
    
    processed = preprocess_image_pro(img)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    res = det_model.ocr(processed, rec=False)
    if not res or not res[0]: return ""
    boxes = res[0]
    
    vertical = sum(1 for b in boxes if max(np.linalg.norm(np.array(b)[0] - np.array(b)[3]), np.linalg.norm(np.array(b)[1] - np.array(b)[2])) > max(np.linalg.norm(np.array(b)[0] - np.array(b)[1]), np.linalg.norm(np.array(b)[2] - np.array(b)[3])) * 1.2)
    if vertical > len(boxes) * 0.5:
        processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        res = det_model.ocr(processed, rec=False)
        boxes = res[0] if res and res[0] else []

    boxes = sorted(boxes, key=lambda b: (int(np.mean([p[1] for p in b]) // 20), int(np.mean([p[0] for p in b]))))
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_box = {executor.submit(process_single_crop, box, gray): box for box in boxes}
        for future in concurrent.futures.as_completed(future_to_box):
            box = future_to_box[future]
            text = future.result()
            if text.strip():
                results.append({"text": text, "y": np.mean([p[1] for p in box]), "x": np.mean([p[0] for p in box]), "h": int(max(np.linalg.norm(np.array(box)[0] - np.array(box)[3]), np.linalg.norm(np.array(box)[1] - np.array(box)[2])))})

    results.sort(key=lambda i: (i["y"] // 15, i["x"]))
    final_doc, prev_y, prev_h = "", -100, 0
    for item in results:
        if prev_y == -100: final_doc += item["text"]
        else: final_doc += (" " if (item["y"] - prev_y) < (prev_h * 1.5) else "\n") + item["text"]
        prev_y, prev_h = item["y"], item["h"]
        
    return final_doc

if __name__ == "__main__":
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    PROCESSED_FILE = os.path.join(ROOT_DIR, "data", "processed", "dataset.csv")
    
    data_list = []
    
    print("\n🔍 ĐANG QUÉT CÁC THƯ MỤC TRONG `data/raw/`...")
    for label_name in os.listdir(RAW_DIR):
        folder_path = os.path.join(RAW_DIR, label_name)
        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
            
            print(f"📂 Đang xử lý nhãn [{label_name}] - ({len(image_files)} ảnh):")
            for filename in tqdm(image_files):
                img_path = os.path.join(folder_path, filename)
                extracted_text = extract_text_from_image(img_path)
                
                if extracted_text.strip():
                    # Gọi hàm bóc tách dữ liệu
                    so_hieu, ngay_ban_hanh, trich_yeu = extract_metadata(extracted_text)
                    
                    # ✅ LƯU TOÀN BỘ VÀO DATASET
                    data_list.append({
                        "filename": filename,
                        "text": extracted_text,
                        "loai_van_ban": label_name, # Lấy nhãn từ tên thư mục chứa ảnh
                        "so_hieu": so_hieu,
                        "ngay_ban_hanh": ngay_ban_hanh,
                        "trich_yeu": trich_yeu
                    })
    
    df = pd.DataFrame(data_list)
    df.to_csv(PROCESSED_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ ĐÃ TẠO DATASET THÀNH CÔNG TẠI: {PROCESSED_FILE}")
    print(f"📊 Tổng số văn bản bóc tách được: {len(df)} mẫu.")