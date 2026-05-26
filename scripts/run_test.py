import sys
import os
import cv2
import time
import numpy as np
from PIL import Image
import concurrent.futures
from pdf2image import convert_from_path

# =================================================================
# BƯỚC 0: CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG & POPPLER
# =================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Đường dẫn đến file thực thi của Poppler (Dùng để đọc PDF)
POPPLER_PATH = os.path.join(ROOT_DIR, "engine","poppler", "Library", "bin")
# 1. IMPORT HÀM TIỀN XỬ LÝ 
from app.core.preprocess_image import preprocess_image_pro

# 2. IMPORT CÁC THƯ VIỆN AI 
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# 3. (Đã tạm bỏ) IMPORT MÔ HÌNH NLP PHOBERT
# PhoBERT import được tạm ẩn để tránh nạp model khi chạy test local
# from app.core.nlp_engine import phobert_engine

# 4. IMPORT TRÍCH XUẤT THÔNG TIN TÀI LIỆU
from app.core.information_extraction import info_extractor

# 5. IMPORT OCR CORRECTION ENGINE (PHOBERT-BASED)
# Để tránh lỗi khi chạy local, hãy uncomment dòng dưới
# from app.core.ocr_correction import ocr_corrector
ocr_corrector = None  # Tạm ẩn - sẽ enable khi đã test

# =================================================================
# KHỞI TẠO CÁC MÔ HÌNH AI (Chỉ chạy 1 lần khi bật tool)
# =================================================================
print("🚀 Đang tải mô hình PaddleOCR (Tìm khung chữ)...")
det_model = PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=False, show_log=False)

print("🚀 Đang tải mô hình VietOCR (Đọc tiếng Việt có dấu)...")
config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
rec_model = Predictor(config)

print("⏳ Đang khởi động lõi AI dịch chữ...")
dummy_img = Image.new('RGB', (100, 32), color=(255, 255, 255))
try:
    _ = rec_model.predict(dummy_img)
except Exception as e:
    print(f"⚠️ LỖI KHỞI TẠO VIETOCR: {e}")

# =================================================================
# HÀM PHỤ TRỢ
# =================================================================
def process_single_crop(raw_box, gray_processed):
    """Cắt từng khung chữ và đưa vào VietOCR dịch"""
    try:
        pts = np.array(raw_box, dtype=np.float32).reshape(4, 2)
        width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        
        if width <= 0 or height <= 0:
            return ""

        dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        crop = cv2.warpPerspective(gray_processed, M, (width, height))
        
        if height > width * 1.2:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        margin_y, margin_x = 4, 8 
        crop_padded = cv2.copyMakeBorder(crop, margin_y, margin_y, margin_x, margin_x, 
                                         cv2.BORDER_CONSTANT, value=255)
        
        target_h = 32
        scale = target_h / crop_padded.shape[0]
        new_w = int(crop_padded.shape[1] * scale)
        new_w = max(1, min(new_w, 512)) 

        crop_resized = cv2.resize(crop_padded, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        pil_img = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2RGB))
        
        return rec_model.predict(pil_img)
    except Exception as e:
        print(f"❌ Lỗi VietOCR ở 1 dòng chữ: {e}")
        return ""

def load_images_from_file(file_path):
    """Phân loại và đọc file (Hỗ trợ cả Ảnh và PDF nhiều trang)"""
    cv2_images = []
    if file_path.lower().endswith('.pdf'):
        print(f"📄 Phát hiện file PDF. Đang dùng Poppler bung các trang thành ảnh...")
        try:
            pil_images = convert_from_path(file_path, dpi=200, poppler_path=POPPLER_PATH)
            for pil_img in pil_images:
                cv2_images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            print(f"✅ Đã bung thành công {len(cv2_images)} trang PDF.")
        except Exception as e:
            print(f"❌ Lỗi đọc PDF: {e}")
            print(f"💡 Kiểm tra lại đường dẫn POPPLER_PATH: {POPPLER_PATH}")
    else:
        img = cv2.imread(file_path)
        if img is not None:
            cv2_images.append(img)
        else:
            print(f"❌ Không thể đọc file: {file_path}")
    return cv2_images


# =================================================================
# LUỒNG CHẠY CHÍNH TỪNG BƯỚC MỘT
# =================================================================
if __name__ == "__main__":
    start_time = time.time()

    # ĐƯỜNG DẪN FILE BẠN MUỐN TEST (Sửa tên file tại đây)
    TEST_FILE_PATH = os.path.join(ROOT_DIR, "data", "raw", "thu_moi", "giaymoi_1.pdf")
    
    print(f"\n[{TEST_FILE_PATH}] Đang nạp dữ liệu...")
    
    pages = load_images_from_file(TEST_FILE_PATH)
    
    if not pages:
        print("⚠️ Không có trang ảnh nào để xử lý. Dừng chương trình.")
        sys.exit()

    full_document_text = ""

    # Chạy vòng lặp qua từng trang (Nếu là ảnh thường thì vòng lặp chỉ chạy 1 lần)
    for page_num, img in enumerate(pages, 1):
        print("\n" + "="*70)
        print(f"🔍 ĐANG XỬ LÝ TRANG SỐ {page_num}/{len(pages)}")
        print("="*70)

        # 1. Tiền xử lý
        processed_img = preprocess_image_pro(img)
        gray_processed = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        # 2. Quét PaddleOCR
        result = det_model.ocr(processed_img, rec=False)
        boxes = result[0] if result and result[0] else []
        if not boxes:
            print(f"⚠️ Không tìm thấy chữ trên trang {page_num}.")
            continue

        # 3. Chống lật ngược ảnh
        is_vertical = 0
        for box in boxes:
            box_arr = np.array(box)
            w = max(np.linalg.norm(box_arr[0] - box_arr[1]), np.linalg.norm(box_arr[2] - box_arr[3]))
            h = max(np.linalg.norm(box_arr[0] - box_arr[3]), np.linalg.norm(box_arr[1] - box_arr[2]))
            if h > w * 1.2:
                is_vertical += 1
                
        if is_vertical > len(boxes) * 0.5:
            processed_img = cv2.rotate(processed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gray_processed = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            result = det_model.ocr(processed_img, rec=False)
            boxes = result[0] if result and result[0] else []

        # 4. Sắp xếp vị trí dòng
        boxes = sorted(boxes, key=lambda b: (int(np.mean([p[1] for p in b]) // 20), int(np.mean([p[0] for p in b]))))

        # 5. Dịch chữ ĐA LUỒNG
        results_with_pos = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_to_box = {executor.submit(process_single_crop, box, gray_processed): box for box in boxes}
            for future in concurrent.futures.as_completed(future_to_box):
                box = future_to_box[future]
                text = future.result()
                if text.strip():
                    y_center = np.mean([p[1] for p in box])
                    x_center = np.mean([p[0] for p in box])
                    pts = np.array(box, dtype=np.float32).reshape(4, 2)
                    box_h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                    results_with_pos.append({
                        "text": text,
                        "y": y_center,
                        "x": x_center,
                        "h": box_h
                    })

        # 6. Ghép văn bản cho trang hiện tại
        results_with_pos.sort(key=lambda item: (item["y"] // 15, item["x"]))
        page_text = ""
        prev_y = -100
        prev_h = 0
        
        for item in results_with_pos:
            if prev_y == -100:
                page_text += item["text"]
            else:
                y_diff = item["y"] - prev_y
                if y_diff < (prev_h * 1.5):
                    page_text += " " + item["text"]
                else:
                    page_text += "\n" + item["text"]
            prev_y = item["y"]
            prev_h = item["h"]

        # Cộng dồn chữ trang này vào tổng văn bản
        full_document_text += page_text + "\n\n"

    # =================================================================
    # BƯỚC TỰA: CHỈNH SỬA LỖI OCR BẰNG PHOBERT (TỰA CHỌN)
    # =================================================================
    corrected_document_text = full_document_text
    if ocr_corrector is not None:
        print("\n" + "="*70)
        print("🔧 BƯỚC: CHỈNH SỬA LỖI OCR BẰNG PHOBERT")
        print("="*70)
        print("⏳ Đang sửa lỗi OCR phổ biến...")
        
        try:
            # Sửa text bằng OCR Correction Engine
            corrected_document_text = ocr_corrector.correct_ocr_text(full_document_text)
            print("✅ Sửa lỗi OCR phổ biến thành công!")
            
            # Tùy chọn: Cải thiện thêm bằng context (có thể mất thời gian)
            # print("⏳ Đang cải thiện bằng context...")
            # corrected_document_text = ocr_corrector.enhance_text_context(corrected_document_text)
            # print("✅ Cải thiện bằng context thành công!")
        except Exception as e:
            print(f"⚠️ Lỗi sửa OCR, sử dụng text gốc: {e}")
    else:
        print("\n⚠️ OCR Correction Engine chưa được enable. Bỏ qua bước sửa lỗi.")

    # =================================================================
    # BƯỚC CUỐI: IN KẾT QUẢ VÀ TRÍCH XUẤT THÔNG TIN
    # =================================================================
    print("\n" + "="*70)
    print("KẾT QUẢ VĂN BẢN TRÍCH XUẤT ĐƯỢC TOÀN BỘ CÁC TRANG:")
    print("="*70)
    print("\n📝 TEXT TRƯỚC SỬA LỖI OCR:")
    print(full_document_text.strip())
    print("\n✏️ TEXT SAU SỬA LỖI OCR (PhoBERT):")
    print(corrected_document_text.strip())
    
    # =================================================================
    # BƯỚC TRÍCH XUẤT THÔNG TIN CHÍNH
    # =================================================================
    print("\n" + "="*70)
    print("📋 TRÍCH XUẤT THÔNG TIN TÀI LIỆU")
    print("="*70)
    
    # Sử dụng text đã sửa lỗi nếu có, nếu không sử dụng text gốc
    extracted_info = info_extractor.extract_all_info(corrected_document_text)
    
    print(f"📅 Ngày tháng năm      : {extracted_info.get('ngay_thang_nam', 'Không xác định')}")
    print(f"📄 Loại văn bản        : {extracted_info.get('loai_van_ban', 'Không xác định')}")
    print(f"🔢 Số hiệu             : {extracted_info.get('so_hieu', 'Không xác định')}")
    print(f"🏙️  Nơi ban hành        : {extracted_info.get('thanh_pho', 'Không xác định')}")
    print(f"📌 Trích yếu           : {extracted_info.get('trich_yeu', 'Không xác định')}")
    print("\n📖 Nội dung (tóm tắt):")
    print(extracted_info.get('noi_dung', 'Nội dung không rõ'))

    print("="*70)
    print(f"⏱ Tổng thời gian chạy (Xử lý ảnh + OCR): {time.time() - start_time:.2f} giây")