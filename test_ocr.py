import cv2
import numpy as np
import os
import warnings
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from pdf2image import convert_from_path

# Tắt các cảnh báo không cần thiết
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# --- CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG) ---
# Tín thay đường dẫn này trỏ tới thư mục 'bin' của Poppler đã giải nén nhé
POPPLER_PATH = r"D:\kooper\poppler-25.12.0\Library\bin" 

def deskew_image(cv2_img):
    """Tự động nắn thẳng văn bản bị chụp nghiêng"""
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45: angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            h, w = cv2_img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
            cv2_img = cv2.warpAffine(cv2_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cv2_img

def process_file_to_text(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return

    # 1. Khởi tạo Models (Dùng bản seq2seq cho ổn định)
    print("\n[1] Đang nạp mô hình AI...")
    det_model = PaddleOCR(use_angle_cls=False, lang='vi', use_gpu=False, rec=False, show_log=False)
    
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = 'cpu'
    rec_model = Predictor(config)

    # 2. Xử lý đầu vào (PDF hoặc Ảnh)
    ext = os.path.splitext(file_path)[1].lower()
    images_to_process = []

    if ext == '.pdf':
        print(f"[2] Đang chuyển PDF sang ảnh (Trang 1)...")
        # Quét trang 1 với DPI 250 để cân bằng giữa tốc độ và độ nét
        pages = convert_from_path(file_path, dpi=250, first_page=1, last_page=1, poppler_path=POPPLER_PATH)
        for page in pages:
            images_to_process.append(cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR))
    else:
        print(f"[2] Đang đọc file ảnh...")
        img = cv2.imread(file_path)
        if img is not None: images_to_process.append(img)

    if not images_to_process:
        print("❌ Lỗi: Không thể xử lý file đầu vào.")
        return

    # 3. Quét OCR
    full_text = []
    for idx, img_cv in enumerate(images_to_process):
        print(f"[3] Đang quét OCR...")
        img_cv = deskew_image(img_cv) # Nắn thẳng
        
        result = det_model.ocr(img_cv, rec=False)
        
        if result[0] is not None:
            # 1. Lấy thông tin box kèm theo tọa độ tâm và chiều cao
            raw_boxes = []
            for box in result[0]:
                pts = np.array(box, dtype="int32")
                xmin, ymin = np.min(pts, axis=0)
                xmax, ymax = np.max(pts, axis=0)
                y_center = (ymin + ymax) / 2
                height = ymax - ymin
                raw_boxes.append({
                    'box': box,
                    'y_center': y_center,
                    'height': height,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })

            # 2. Sắp xếp tất cả theo Y trước
            raw_boxes.sort(key=lambda x: x['y_center'])

            # 3. Nhóm các box vào từng dòng dựa trên ngưỡng (threshold)
            lines = []
            if raw_boxes:
                current_line = [raw_boxes[0]]
                for i in range(1, len(raw_boxes)):
                    # Nếu box tiếp theo chênh lệch Y không quá 1/2 chiều cao box trước đó -> cùng dòng
                    if abs(raw_boxes[i]['y_center'] - current_line[-1]['y_center']) < (current_line[-1]['height'] / 2):
                        current_line.append(raw_boxes[i])
                    else:
                        lines.append(current_line)
                        current_line = [raw_boxes[i]]
                lines.append(current_line)

            # 4. Quét từng dòng, sắp xếp box trong dòng theo X (trái sang phải) và nhận diện
            final_lines = []
            print("\n--- ĐANG TRÍCH XUẤT THEO ĐỊNH DẠNG ---")
            for line in lines:
                line.sort(key=lambda x: x['xmin']) # Sắp xếp trái -> phải
                line_texts = []
                for item in line:
                    crop = img_cv[max(0, item['ymin']-2):item['ymax']+2, max(0, item['xmin']-2):item['xmax']+2]
                    if crop.size > 0:
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        text = rec_model.predict(pil_img)
                        line_texts.append(text)
                
                # Nối các chữ trong cùng 1 dòng bằng khoảng cách xa (để giả lập chia cột)
                full_line_text = "    ".join(line_texts) 
                print(full_line_text)
                final_lines.append(full_line_text)

    # Kết quả cuối cùng giữ nguyên xuống dòng
    result_string = "\n".join(final_lines)
    return result_string

if __name__ == "__main__":
    # Tín đổi tên file này thành file bạn muốn test (có thể là .pdf hoặc .jpg)
    target_file = "data/raw/thu_moi/congvan.pdf" 
    process_file_to_text(target_file)