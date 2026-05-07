import sys
import os

# ✅ THÊM DÒNG NÀY ĐẦU TIÊN: Trỏ đường dẫn hệ thống ngược ra ngoài thư mục gốc
# Để Python có thể nhìn thấy thư mục 'app'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import cv2
import time
import numpy as np
from PIL import Image
import concurrent.futures

# 1. IMPORT HÀM TIỀN XỬ LÝ (Lúc này Python đã hiểu 'app' là gì)
from app.core.preprocess_image import preprocess_image_pro

# 2. IMPORT CÁC THƯ VIỆN AI
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# Khởi tạo mô hình PaddleOCR (Chỉ dùng để TÌM VỊ TRÍ CHỮ)
print("🚀 Đang tải mô hình PaddleOCR (Tìm khung chữ)...")
det_model = PaddleOCR(
    use_angle_cls=True, 
    lang="vi", 
    use_gpu=False, 
    show_log=False,
    det_limit_side_len=2048, # ✅ Ép AI nhìn ảnh ở độ phân giải cao (tối đa 2048px), cấm thu nhỏ
    det_db_thresh=0.1,       # ✅ Giảm ngưỡng nhạy cảm để bắt được các nét chữ nhỏ/mỏng
    det_db_box_thresh=0.3,   # ✅ Giảm ngưỡng chấp nhận để không bỏ sót các khung chữ khó
    det_db_unclip_ratio=1.5  # ✅ Bơm phồng khung chữ ra một chút để không cắt lẹm dấu tiếng Việt
)

# Khởi tạo mô hình VietOCR (Dùng để ĐỌC TIẾNG VIỆT)
print("🚀 Đang tải mô hình VietOCR (Đọc tiếng Việt có dấu)...")
config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
rec_model = Predictor(config)

# =================================================================
# ✅ HÀM PHỤ: XỬ LÝ ĐỘC LẬP TỪNG DÒNG CHỮ (DÙNG CHO ĐA LUỒNG)
# =================================================================
def process_single_crop(raw_box, gray_processed):
    try:
        # Cắt từng dòng chữ ra khỏi tờ giấy
        pts = np.array(raw_box, dtype=np.float32).reshape(4, 2)
        width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        
        dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        crop = cv2.warpPerspective(gray_processed, M, (width, height))
        
        if height > width * 1.2:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Thêm khoảng đệm (padding)
        margin_y, margin_x = 4, 8 
        crop_padded = cv2.copyMakeBorder(crop, margin_y, margin_y, margin_x, margin_x, 
                                         cv2.BORDER_CONSTANT, value=255)
        
        # Đưa về kích thước chuẩn
        target_h = 32
        scale = target_h / crop_padded.shape[0]
        new_w = int(crop_padded.shape[1] * scale)
        new_w = min(new_w, 512) # Clamp width

        crop_resized = cv2.resize(crop_padded, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Chuyển định dạng OpenCV sang PIL
        pil_img = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2RGB))
        
        # Đọc tiếng Việt
        text = rec_model.predict(pil_img)
        return text
    except Exception as e:
        return ""


def test_pipeline(image_path):
    print(f"\n[{image_path}] Đang nạp ảnh...")
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("❌ LỖI: Không tìm thấy ảnh hoặc sai đường dẫn!")
        return

    start_time = time.time()

    # Bước 1: TIỀN XỬ LÝ ẢNH
    print("✨ Bước 1: Tiền xử lý ảnh (Khử bóng, bù sáng, nhị phân...)")
    processed_img = preprocess_image_pro(original_img)
    cv2.imwrite("debug_sau_xu_ly.jpg", processed_img)

    # Bước 2: TÌM VỊ TRÍ CHỮ
    print("🔍 Bước 2: PaddleOCR tìm vị trí các dòng chữ...")
    result = det_model.ocr(processed_img, rec=False)

    if not result or result[0] is None:
        print("⚠️ AI không tìm thấy chữ nào trong bức ảnh này.")
        return
        
    boxes = result[0]

    # =================================================================
    # ✅ THÊM MỚI: TỰ ĐỘNG XOAY ẢNH NẾU PHÁT HIỆN ẢNH NẰM NGANG
    # =================================================================
    vertical_boxes = 0
    for box in boxes:
        pts = np.array(box, dtype=np.float32)
        w = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
        h = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
        # Đếm số lượng khung chữ bị nằm dọc
        if h > w * 1.2:
            vertical_boxes += 1
            
    # Nếu > 50% khung chữ là nằm dọc -> Ảnh đang nằm ngang
    if vertical_boxes > len(boxes) * 0.5:
        print("🔄 Phát hiện ảnh bị xoay ngang! Đang tự động dựng thẳng ảnh...")
        # Xoay ảnh 90 độ ngược chiều kim đồng hồ để chữ đứng thẳng lại
        processed_img = cv2.rotate(processed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite("debug_sau_xu_ly.jpg", processed_img) # Lưu đè lại ảnh đã xoay thẳng
        
        # Quét lại vị trí chữ trên bức ảnh đã được dựng thẳng
        result = det_model.ocr(processed_img, rec=False)
        boxes = result[0] if result and result[0] else []


    print("🇻🇳 Bước 3: VietOCR cắt từng dòng và dịch (Đang chạy ĐA LUỒNG)...")
    print("\n" + "="*70)
    print("KẾT QUẢ VĂN BẢN TRÍCH XUẤT ĐƯỢC:")
    print("="*70)
    
    # Tăng hệ số chia sẻ dòng ngang lên 20 để phân tách các dòng sát nhau tốt hơn
    boxes = sorted(boxes, key=lambda b: (int(np.mean([p[1] for p in b]) // 20), int(np.mean([p[0] for p in b]))))
    
    gray_processed = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    # Vẽ ảnh debug
    img_with_boxes = processed_img.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_with_boxes, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite("debug_boxes.jpg", img_with_boxes)
    print("📸 Đã lưu ảnh vẽ khung chữ tại: debug_boxes.jpg (Mở file này ra để xem AI có khoanh trượt dòng nào không!)")
    
    # =================================================================
    # BƯỚC 4: CHẠY SONG SONG VÀ GOM NHÓM KẾT QUẢ THÀNH ĐOẠN VĂN
    # =================================================================
    max_threads = min(6, (os.cpu_count() or 1) + 2) 
    
    # Dùng đa luồng để dịch toàn bộ các khung chữ
    results_with_coords = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit task kèm theo box để lát nữa còn biết tọa độ mà ghép
        future_to_box = {executor.submit(process_single_crop, box, gray_processed): box for box in boxes}
        
        for future in concurrent.futures.as_completed(future_to_box):
            box = future_to_box[future]
            text = future.result()
            if text.strip():
                # Lưu lại text cùng với tọa độ tâm Y và X của khung
                center_y = np.mean([p[1] for p in box])
                center_x = np.mean([p[0] for p in box])
                # Lưu thêm chiều cao của khung để ước lượng khoảng cách dòng
                height = int(max(np.linalg.norm(np.array(box[0]) - np.array(box[3])), 
                                 np.linalg.norm(np.array(box[1]) - np.array(box[2]))))
                results_with_coords.append({
                    "text": text,
                    "y": center_y,
                    "x": center_x,
                    "h": height
                })

    # =================================================================
    # BƯỚC 5: HẬU XỬ LÝ - GHÉP DÒNG THÀNH ĐOẠN VĂN (DOCUMENT LAYOUT)
    # =================================================================
    # 1. Sắp xếp lại toàn bộ kết quả từ trên xuống dưới, trái qua phải
    results_with_coords.sort(key=lambda item: (item["y"] // 15, item["x"]))

    final_document = ""
    prev_y = -100
    prev_h = 0

    for item in results_with_coords:
        text = item["text"]
        curr_y = item["y"]
        curr_h = item["h"]

        # Nếu là dòng đầu tiên
        if prev_y == -100:
            final_document += text
        else:
            # Tính khoảng cách theo trục Y so với dòng trước đó
            y_diff = curr_y - prev_y
            
            # Logic nối dòng:
            # Nếu khoảng cách Y nhỏ hơn 1.5 lần chiều cao chữ -> Khả năng cao là cùng một đoạn văn bị rớt dòng
            if y_diff < (prev_h * 1.5):
                # Nối tiếp bằng 1 khoảng trắng (cùng đoạn)
                final_document += " " + text
            else:
                # Nếu khoảng cách xa hơn -> Xuống dòng tạo đoạn mới
                final_document += "\n" + text
        
        prev_y = curr_y
        prev_h = curr_h

    print("👉 " + final_document.replace("\n", "\n👉 "))
            
    print("\n" + "="*70)
    print(f"Tổng thời gian chạy (Xử lý ảnh + Paddle + VietOCR): {time.time() - start_time:.2f} giây")

if __name__ == "__main__":
    test_image_file = r"data\raw\thu_moi\img036.jpg" 
    
    if os.path.exists(test_image_file):
        test_pipeline(test_image_file)
    else:
        print(f"Vui lòng kiểm tra lại xem file '{test_image_file}' có tồn tại không nhé!")