import cv2
import numpy as np
import os
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from pdf2image import convert_from_path

class AdminOCR:
    def __init__(self):
        # Khởi tạo 1 lần dùng mãi mãi để tiết kiệm RAM
        self.det_model = PaddleOCR(use_angle_cls=False, lang='vi', use_gpu=False, rec=False, show_log=False)
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['device'] = 'cpu'
        self.rec_model = Predictor(config)

    def deskew(self, cv2_img):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
            angles = [a for a in angles if -45 < a < 45]
            if angles:
                M = cv2.getRotationMatrix2D((cv2_img.shape[1]//2, cv2_img.shape[0]//2), np.median(angles), 1.0)
                return cv2.warpAffine(cv2_img, M, (cv2_img.shape[1], cv2_img.shape[0]), flags=cv2.INTER_CUBIC)
        return cv2_img

    def predict_image(self, cv2_img):
        cv2_img = self.deskew(cv2_img)
        result = self.det_model.ocr(cv2_img, rec=False)
        if not result or result[0] is None: return ""
        
        boxes = sorted(result[0], key=lambda x: np.mean([p[1] for p in x]))
        texts = []
        for box in boxes:
            pts = np.array(box, dtype="int32")
            xmin, ymin = np.min(pts, axis=0)
            xmax, ymax = np.max(pts, axis=0)
            crop = cv2_img[max(0,ymin-2):ymax+2, max(0,xmin-2):xmax+2]
            if crop.size > 0:
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                texts.append(self.rec_model.predict(pil_img))
        return " ".join(texts)

    def process_any_file(self, file_path, poppler_path=None):
        # Tự động nhận diện PDF hay Ảnh
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            # Chỉ quét trang đầu tiên để lấy nhãn (tiết kiệm thời gian)
            pages = convert_from_path(file_path, dpi=200, first_page=1, last_page=1, poppler_path=poppler_path)
            cv2_img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            return self.predict_image(cv2_img)
        else:
            cv2_img = cv2.imread(file_path)
            return self.predict_image(cv2_img)