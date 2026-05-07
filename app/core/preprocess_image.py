import cv2
import numpy as np

# ================================================================
# BƯỚC 0: KIỂM TRA CHẤT LƯỢNG ẢNH ĐẦU VÀO
# ================================================================
def check_image_quality(gray: np.ndarray) -> dict:
    """
    Đánh giá ảnh trước khi xử lý để chọn chiến lược phù hợp.
    - variance of Laplacian < 100: ảnh mờ → cần sharpen mạnh hơn
    - std < 30: ảnh thiếu tương phản → cần CLAHE mạnh hơn
    """
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    brightness = gray.mean()
    return {
        "is_blurry": blur_score < 100,
        "is_low_contrast": contrast < 30,
        "is_dark": brightness < 80,
        "blur_score": blur_score,
    }

# ================================================================
# BƯỚC 1: CHUẨN HÓA KÍCH THƯỚC
# ================================================================
def normalize_size(img: np.ndarray, target_height: int = 2000) -> np.ndarray:
    """
    Resize ảnh về chiều cao chuẩn, giữ tỷ lệ.
    PaddleOCR hoạt động tốt nhất ở 1500–2500px chiều cao.
    """
    h, w = img.shape[:2]
    if h < target_height:
        scale = target_height / h
        img = cv2.resize(img, (int(w * scale), target_height),
                         interpolation=cv2.INTER_CUBIC)  # CUBIC tốt hơn LINEAR khi phóng to
    return img

# ================================================================
# BƯỚC 2: KHỬ BÓNG ĐỔ (shadow removal)
# ================================================================
def remove_shadow(gray: np.ndarray) -> np.ndarray:
    """
    Loại bỏ bóng đổ bằng cách chia ảnh gốc cho background ước lượng.
    Đặc biệt hiệu quả với ảnh chụp điện thoại bị bóng tay/ánh sáng không đều.
    """
    # Dilate để "xóa" chữ, chỉ còn background
    dilated = cv2.dilate(gray, np.ones((21, 21), np.uint8))
    # Blur mạnh để làm mịn background
    bg = cv2.GaussianBlur(dilated, (21, 21), 0)
    # Chia để normalize ánh sáng
    result = cv2.divide(gray, bg, scale=255)
    return result

# ================================================================
# BƯỚC 3: KHỬ NHIỄU
# ================================================================
def denoise(gray: np.ndarray, is_blurry: bool) -> np.ndarray:
    """
    - Ảnh đã mờ: dùng bilateral (giữ cạnh, không làm mờ thêm)
    - Ảnh sắc nét: dùng fastNlMeans (mạnh hơn, loại nhiễu scan/fax)
    """
    if is_blurry:
        return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    else:
        return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

# ================================================================
# BƯỚC 4: TĂNG TƯƠNG PHẢN (CLAHE thích nghi)
# ================================================================
def enhance_contrast(gray: np.ndarray, is_low_contrast: bool) -> np.ndarray:
    """
    clipLimit cao hơn khi ảnh thiếu sáng/tương phản kém.
    tileGridSize nhỏ hơn → local contrast tốt hơn cho văn bản nhỏ.
    """
    clip = 3.0 if is_low_contrast else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)

# ================================================================
# BƯỚC 5: LÀM SẮC NÉT (unsharp masking)
# ================================================================
def sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Unsharp masking: cộng thêm phần "detail" đã trừ khỏi blur.
    Kết quả rõ nét hơn nhiều so với kernel sharpen cứng.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened

# ================================================================
# BƯỚC 6: PHÁT HIỆN VÀ CHỈNH GÓC NGHIÊNG (deskew)
# ================================================================
def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Dùng HoughLinesP để detect góc nghiêng của văn bản rồi xoay lại.
    Bỏ qua nếu góc < 0.5° (không đáng kể) hoặc > 45° (detect sai).
    """
    # Threshold nhanh để HoughLines chạy trên binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Chỉ lấy các đường gần nằm ngang (văn bản)
            if -45 < angle < 45:
                angles.append(angle)
    
    if not angles:
        return gray
    
    median_angle = np.median(angles)
    
    # Bỏ qua góc nghiêng quá nhỏ
    if abs(median_angle) < 0.5:
        return gray
    
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ================================================================
# BƯỚC 7: NHỊ PHÂN HÓA (binarization)
# ================================================================
def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Kết hợp Otsu (global) và Adaptive (local) rồi lấy AND.
    - Otsu tốt với ảnh scan đồng đều
    - Adaptive tốt với ảnh chụp có ánh sáng không đều
    - AND của 2 cái → ít nhiễu nhất, chỉ giữ pixel cả 2 đồng ý là chữ
    """
    # Otsu global
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive local
    adaptive = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 35, 12)
    
    # Kết hợp: lấy pixel sáng hơn (ít noise hơn)
    combined = cv2.bitwise_and(otsu, adaptive)
    return combined

# ================================================================
# BƯỚC 8: MORPHOLOGY – vá nét chữ bị đứt
# ================================================================
def clean_morphology(binary: np.ndarray) -> np.ndarray:
    """
    - Close (dilate → erode): nối các nét bị đứt đoạn
    - Kernel nhỏ (2x1) để không làm dày chữ quá mức
    """
    # Kernel ngang để kết nối chữ bị đứt theo chiều ngang
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    return closed

def remove_red_stamps(img: np.ndarray) -> np.ndarray:
    """
    Phát hiện và xóa các vùng có màu đỏ (con dấu, chữ ký mực đỏ), 
    trả lại nền trắng để không che khuất chữ đen.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Dải màu đỏ trong HSV (Đỏ có 2 dải ở đầu và cuối dải Hue)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Làm phình mask một chút để xóa sạch viền đỏ
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    
    # Thay thế các pixel màu đỏ bằng màu trắng (255, 255, 255)
    result = img.copy()
    result[mask > 0] = (255, 255, 255)
    return result

# ================================================================
# PIPELINE TỔNG HỢP – thay thế preprocess_image_pro()
# ================================================================
def preprocess_image_pro(img: np.ndarray) -> np.ndarray:
    """
    Pipeline tối ưu hơn cho OCR (fix deskew + tránh over-processing)
    """

    # 1. Resize vừa đủ (đừng quá to)
    img = normalize_size(img, target_height=1600)

    # 2. Remove stamp trước
    img = remove_red_stamps(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Deskew SỚM (quan trọng)
    gray = deskew(gray)

    # 4. Check quality
    quality = check_image_quality(gray)

    # 5. Chỉ remove shadow khi cần
    if quality["is_low_contrast"]:
        gray = remove_shadow(gray)

    # 6. Denoise
    gray = denoise(gray, is_blurry=quality["is_blurry"])

    # 7. Contrast
    gray = enhance_contrast(gray, is_low_contrast=quality["is_low_contrast"])

    # 8. Sharpen nhẹ hơn (tránh quá tay)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    gray = cv2.addWeighted(gray, 1.3, blurred, -0.3, 0)

    # 9. Padding
    padded = cv2.copyMakeBorder(gray, 30, 30, 30, 30,
                                 cv2.BORDER_CONSTANT, value=255)

    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)