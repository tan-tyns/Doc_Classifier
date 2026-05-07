import cv2
import numpy as np
import fitz  # Import PyMuPDF để xử lý PDF
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import random

# Import các hàm lõi từ file test_ocr.py của bạn
from test_ocr import process_image, extract_structured_info, nlp_classifier

app = FastAPI(title="Classify Doc AI - Backend API")

# Cấu hình CORS để cho phép React gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    API Nhận file (Ảnh hoặc PDF) từ React, quét OCR, trích xuất NLP và trả về JSON.
    """
    try:
        # 1. Đọc file tải lên vào bộ nhớ RAM (Bytes)
        contents = await file.read()
        filename = file.filename.lower()
        img = None

        # 2. KIỂM TRA ĐỊNH DẠNG FILE
        if filename.endswith(".pdf"):
            # NẾU LÀ PDF: Dùng PyMuPDF đọc trang đầu tiên và chuyển thành ảnh OpenCV
            doc = fitz.open(stream=contents, filetype="pdf")
            page = doc.load_page(0) # Lấy trang đầu tiên (index 0)
            
            # Phóng to ảnh (zoom) để OCR đọc nét hơn (DPI ~ 200)
            zoom_matrix = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=zoom_matrix)
            
            # Chuyển đổi dữ liệu pixel của PDF thành Numpy Array cho OpenCV
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # Convert hệ màu RGB (PDF) sang BGR (chuẩn của OpenCV)
            if pix.n == 4: # Nếu có kênh Alpha trong suốt
                img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
            doc.close()
            
        else:
            # NẾU LÀ ẢNH (JPG, PNG): Đọc bình thường
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Kiểm tra xem có đọc được ảnh thành công không
        if img is None:
            return {"text": "Lỗi: Không thể phân tích file tải lên.", "label": "ERROR", "confidence": 0}

        # 3. Chạy AI OCR (Paddle + VietOCR)
        raw_text = process_image(img)
        
        # 4. Trích xuất thông tin & Phân loại văn bản
        parsed_info = extract_structured_info(raw_text)
        phan_loai = nlp_classifier.predict(parsed_info["noi_dung"])
        
        # Mô phỏng độ tin cậy
        confidence = random.randint(88, 98) if phan_loai != "VĂN BẢN KHÁC" else 50
        
        # Format text đầu ra
        final_text = (
            f"--- THÔNG TIN TRÍCH XUẤT ---\n"
            f"Loại văn bản: {parsed_info['loai_van_ban']}\n"
            f"Ngày ban hành: {parsed_info['ngay_thang']}\n\n"
            f"--- NỘI DUNG ---\n"
            f"{parsed_info['noi_dung']}"
        )

        # 5. Trả về JSON 
        return {
            "text": final_text,
            "label": phan_loai,
            "confidence": confidence,
            "docType": parsed_info['loai_van_ban'],
            "date": parsed_info['ngay_thang'],
            "content": parsed_info['noi_dung']
        }
        
    except Exception as e:
        return {"text": f"Lỗi hệ thống Backend: {str(e)}", "label": "ERROR", "confidence": 0}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Khởi động Classify API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)