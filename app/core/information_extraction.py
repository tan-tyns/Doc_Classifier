import re
from typing import Dict, List, Tuple
from datetime import datetime

class DocumentInfoExtractor:
    """
    Trích xuất thông tin cơ bản từ văn bản OCR tiếng Việt:
    - Ngày tháng năm
    - Loại văn bản
    - Số hiệu
    - Trích yếu (tóm tắt)
    - Nội dung chính
    """
    
    # Định nghĩa các loại văn bản
    DOCUMENT_TYPES = {
        "công_văn": ["công văn", "cong van", "cv"],
        "quyết_định": ["quyết định", "quyet dinh", "qđ", "qd"],
        "thông_báo": ["thông báo", "thong bao", "tb"],
        "giấy_mời": ["giấy mời", "giay moi", "gm", "giấy mời họp", "giay moi hop", "thư mời", "thu moi"],
        "tờ_trình": ["tờ trình", "to trinh", "tt"],
        "báo_cáo": ["báo cáo", "bao cao", "bc"],
        "kế_hoạch": ["kế hoạch", "ke hoach", "kh"]
    }

    @staticmethod
    def extract_date(text: str) -> str:
        """
        Trích xuất ngày tháng năm.
        Cập nhật: Bắt thêm lỗi OCR "tháng" -> "thảng", "thạng"...
        """
        lines = text.split('\n')[:15]
        header_text = " ".join(lines)
        
        # Thêm các ký tự dấu vào pattern để bắt được chữ "thảng"
        pattern_vn = r'(?:ngày\s+)?(\d{1,2})\s+th[áaảãạ]\w*\s+(\d{1,2})\s+năm\s+(\d{4})'
        match = re.search(pattern_vn, header_text, re.IGNORECASE)
        if match:
            day, month, year = match.groups()
            return f"{day}/{month}/{year}"
        
        pattern_slash = r'(\d{1,2})/(\d{1,2})/(\d{4})'
        match = re.search(pattern_slash, header_text)
        if match:
            return match.group(0)
            
        pattern_dash = r'(\d{4})-(\d{1,2})-(\d{1,2})'
        match = re.search(pattern_dash, header_text)
        if match:
            year, month, day = match.groups()
            return f"{day}/{month}/{year}"
        
        return "Không xác định"
    
    @staticmethod
    def extract_document_number(text: str) -> str:
        """
        Trích xuất số hiệu văn bản.
        Cập nhật: Nhảy cóc (skip) các từ rác OCR chèn vào giữa thay vì bỏ ngang.
        """
        lines = text.split('\n')[:15]
        
        for line in lines:
            m = re.search(r'(?:^\s*Số\s*[:]?|Số\s*:)\s*(.+)', line, re.IGNORECASE)
            if m:
                raw_str = m.group(1).strip()
                
                # Cắt bỏ từ chữ "ngày", dấu phẩy, hoặc "thảng"
                cut_m = re.search(r'(,|\bngày\b|\bth[áaảãạ]\w*\b|\bnăm\b)', raw_str, re.IGNORECASE)
                if cut_m:
                    raw_str = raw_str[:cut_m.start()].strip()
                
                parts = raw_str.split()
                valid_parts = []
                
                for p in parts:
                    # Bổ sung 'kh' vào danh sách đuôi hợp lệ
                    if re.search(r'[0-9\/\-]', p) or p.isupper() or p.lower() in ['qđ', 'ubnd', 'nđ', 'cp', 'tt', 'ct', 'kh', 'stp']:
                        valid_parts.append(p)
                    else:
                        # Điểm mấu chốt: Bỏ qua từ rác (như "Salling") thay vì break
                        continue
                
                if valid_parts:
                    number_str = " ".join(valid_parts)
                    number_str = re.sub(r'\s*([/\-])\s*', r'\1', number_str)
                    number_str = re.sub(r'\s+', '/', number_str)
                    number_str = number_str.strip('/-.,')
                    
                    if number_str and len(number_str) > 1:
                        return number_str
        
        return "Không xác định"

    @staticmethod
    def detect_document_type(text: str) -> str:
        """
        Phát hiện loại văn bản dựa trên keywords.
        """
        text_lower = text.lower()
        
        for doc_type, keywords in DocumentInfoExtractor.DOCUMENT_TYPES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Ưu tiên các từ xuất hiện gần đầu (thường là tiêu đề)
                    if text_lower.find(keyword) < len(text) * 0.3:
                        return doc_type.replace("_", " ").title()
        
        return "Văn bản khác"

    @staticmethod
    def extract_title(text: str) -> str:
        """
        Trích xuất tiêu đề văn bản.
        Ưu tiên: 1) Dòng bắt đầu với V/v, Về, Chuyên đề, Ban hành (ở đầu dòng)
                 2) Dòng tiếp theo sau loại văn bản (công văn, quyết định, giấy mời, v.v.)
        """
        lines = text.strip().split('\n')
        
        # Cách 1: Tìm dòng **bắt đầu với** V/v, Về, Chuyên đề, Ban hành (ở đầu, không phải giữa câu)
        for line in lines:
            # Match ở đầu dòng (sau whitespace nếu có)
            match = re.search(r'^\s*(?:V/v|Về|Chuyên đề|Ban hành)\s*[:–-]?\s*(.+)', line, re.IGNORECASE)
            if match:
                title = re.sub(r'^0\s+', '', line[match.start():]).strip()
                if title:
                    return title
        
        # Cách 2: Nếu không có V/v, lấy dòng tiếp theo sau loại văn bản (công văn, quyết định, giấy mời, etc.)
        doc_type_idx = -1
        for i, line in enumerate(lines):
            if re.search(r'(công văn|quyết định|thông báo|giấy mời|tờ trình|báo cáo|kế hoạch|giấy mời họp|thư mời)', 
                        line, re.IGNORECASE):
                doc_type_idx = i
                break
        
        if doc_type_idx >= 0:
            # Tìm dòng không rỗng tiếp theo
            for i in range(doc_type_idx + 1, len(lines)):
                title = lines[i].strip()
                # Bỏ số 0 ở đầu nếu có
                title = re.sub(r'^0\s+', '', title)
                
                # Không lấy nếu là dòng chứa "Kính gửi", "Gửi", "Thực hiện" (phần nội dung)
                if title and not re.match(r'^(Kính\s*gửi|Gửi|Thực hiện)', title, re.IGNORECASE):
                    return title
        
        return "Tiêu đề không rõ"

    @staticmethod
    def extract_summary(text: str) -> str:
        """
        Trích xuất trích yếu.
        Cập nhật: Phân biệt "V/v" (có thể ở giữa dòng) và "Về việc" (phải ở đầu dòng).
        Bổ sung chặn các chức danh (GIÁM ĐỐC, CHỦ TỊCH) và lỗi OCR siêu nặng (Cắn cứ, Căn cử).
        """
        lines = [l.strip() for l in text.split('\n')[:30] if l.strip()]

        # ── ƯU TIÊN 1: Tìm cụm "V/v", "Về việc" ──
        for i, line in enumerate(lines):
            # "V/v" là viết tắt đặc trưng, thường an toàn để tìm ở bất cứ đâu (trị lỗi gộp dòng)
            m = re.search(r'\b(V[/\\]v)\s*[:–\-]?\s*(.+)', line, re.IGNORECASE)
            
            # "Về việc", "Ban hành" là từ thông dụng, CHỈ BẮT khi nó đứng ở đầu dòng
            if not m:
                m = re.match(r'^(Về việc|Về|Chuyên đề|Ban hành)\s*[:–\-]?\s*(.+)', line, re.IGNORECASE)
                
            if m:
                summary_lines = [m.group(2).strip()]
                
                # BỘ LỌC DỪNG: Nâng cấp Regex để tóm cả "Cắn cứ", "Căn cử" và chức danh lãnh đạo
                stop_pattern = r'^(C[ăaâắằảãạấầẩẫậ]+n\s+c[ứuưửữự]+|Theo\b|Điều\b|Kính gửi|Gửi\b|1\.|Thực hiện\b|GIÁM ĐỐC|GIẨM ĐỘC|CHỦ TỊCH|TM\.|KT\.|UBND|ỦY BAN)'
                
                for j in range(i + 1, min(i + 7, len(lines))):
                    next_line = lines[j]
                    
                    if re.search(stop_pattern, next_line, re.IGNORECASE):
                        break
                        
                    if re.search(r'(SAO Y|CỘNG HÒA|CỘNG HOÀ|ĐỘC LẬP|Hạnh phúc|CÔNG BẢO)', next_line, re.IGNORECASE):
                        continue
                        
                    if len(next_line) > 2:
                         summary_lines.append(next_line)
                
                return " ".join(summary_lines)

        # ── ƯU TIÊN 2: Nằm sau chữ "QUYẾT ĐỊNH", "THÔNG BÁO"... ──
        doc_type_pattern = r'^(QUYẾT ĐỊNH|THÔNG BÁO|BÁO CÁO|TỜ TRÌNH|KẾ HOẠCH|CHỈ THỊ|GIẤY MỜI|THƯ MỜI)'
        for i, line in enumerate(lines):
            match = re.search(doc_type_pattern, line, re.IGNORECASE)
            if match:
                summary_lines = []
                remainder = line[match.end():].strip()
                remainder = re.sub(r'^[:\-\s]+', '', remainder)
                if remainder:
                    summary_lines.append(remainder)
                
                # Cập nhật bộ lọc dừng y hệt như Priority 1
                stop_pattern = r'^(C[ăaâắằảãạấầẩẫậ]+n\s+c[ứuưửữự]+|Theo\b|Điều\b|Kính gửi|Gửi\b|1\.|Thực hiện\b|GIÁM ĐỐC|GIẨM ĐỘC|CHỦ TỊCH|TM\.|KT\.|UBND|ỦY BAN)'
                
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j]
                    if re.match(stop_pattern, next_line, re.IGNORECASE):
                        break
                    if re.match(r'^ngày\s+\d+\s+th[áa]ng\s+\d+\s+năm', next_line, re.IGNORECASE):
                        continue
                    if len(next_line) > 5:
                         summary_lines.append(next_line)
                
                if summary_lines:
                    return " ".join(summary_lines)

        # ── ƯU TIÊN 3: Fallback (Quét vớt) ──
        for line in lines:
            if re.search(r'(SAO Y|CỘNG HÒA|CỘNG HOÀ|ĐỘC LẬP|Hạnh phúc|CÔNG BẢO|THỦ TƯỚNG|BỘ TƯ PHÁP|Số:|tháng|ngày|năm)', line, re.IGNORECASE):
                continue
            if re.match(r'^(C[ăaâắằảãạấầẩẫậ]+n\s+c[ứuưửữự]+|Theo\b|Điều\b|Kính gửi|Thực hiện|GIÁM ĐỐC|GIẨM ĐỘC)', line, re.IGNORECASE):
                continue
            if len(line) > 30:
                return line.strip()

        return "Trích yếu không rõ"
    
    @staticmethod
    def extract_main_content(text: str) -> str:
        """
        Trích xuất nội dung chính.
        Ưu tiên: Kính gửi → Mục 1. → Thực hiện → từ dòng 6
        """
        lines = text.strip().split('\n')
        content_start = 0
        
        # Cách 1: Từ "Kính gửi"
        for i, line in enumerate(lines):
            if re.match(r'^Kính\s+gửi', line.strip(), re.IGNORECASE):
                content_start = i
                break
        
        # Cách 2: Từ mục "1."
        if content_start == 0:
            for i, line in enumerate(lines):
                if re.match(r'^\s*1[\.\)]\s+', line.strip()):
                    content_start = i
                    break
        
        # Cách 3: Từ "Thực hiện"
        if content_start == 0:
            for i, line in enumerate(lines):
                if re.match(r'^Thực hiện', line.strip(), re.IGNORECASE):
                    content_start = i
                    break
        
        # Cách 4: Fallback — dòng 6
        if content_start == 0:
            content_start = min(6, len(lines))
        
        # Tìm điểm kết thúc (ký tên/footer)
        content_end = len(lines)
        for i in range(len(lines) - 1, max(content_start, 0), -1):
            line_lower = lines[i].lower().strip()
            if re.search(r'^(tm\s|ký\s|chủ tịch|giám đốc|trưởng|phó|người ký|ủy quyền|nơi nhận|lưu|tl\.|fax|điện thoại)', line_lower):
                content_end = i
                break
        
        # Lấy nội dung
        content_lines = []
        for i in range(content_start, content_end):
            line = lines[i].strip()
            if line:
                content_lines.append(line)
        
        content = "\n".join(content_lines).strip()
        
        # Loại bỏ header thừa
        content = re.sub(r'^Số\s*[:=]?[^\n]*\n+', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^(CÔNG VĂN|QUYẾT ĐỊNH|THÔNG BÁO|GIẤY MỜI|TỜ TRÌNH|BÁO CÁO|KỂ HOẠCH)[^\n]*\n+', '', content, flags=re.IGNORECASE)
        
        if len(content) > 2000:
            content = content[:2000] + "..."
        
        return content.strip() if content.strip() else "Nội dung không rõ"

    @staticmethod
    def extract_city(text: str) -> str:
        """
        Trích xuất thành phố/nơi ban hành từ văn bản.
        Pattern: "Số: ... Hà Nội, ngày ..." hoặc "Hà Nội, ngày ..."
        """
        # Danh sách thành phố/tỉnh chính ở Việt Nam
        cities = [
            "Hà Nội", "Hồ Chí Minh", "TP HCM", "Sài Gòn",
            "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Bình Dương",
            "Đồng Nai", "Long An", "Tiền Giang", "Bến Tre",
            "Vĩnh Long", "An Giang", "Kiên Giang", "Cà Mau",
            "Thái Nguyên", "Bắc Kạn", "Cao Bằng", "Lạng Sơn",
            "Tuyên Quang", "Yên Bái", "Sơn La", "Điện Biên",
            "Lai Châu", "Hà Giang", "Phú Thọ", "Vĩnh Phúc",
            "Bắc Ninh", "Bắc Giang", "Hưng Yên", "Hải Dương",
            "Thái Bình", "Nam Định", "Ninh Bình", "Thanh Hóa",
            "Nghệ An", "Hà Tĩnh", "Quảng Bình", "Quảng Trị",
            "Thừa Thiên Huế", "Quảng Nam", "Quảng Ngãi", "Bình Định",
            "Phú Yên", "Khánh Hòa", "Ninh Thuận", "Bình Thuận",
            "Đắk Nông", "Đắk Lắk", "Lâm Đồng", "Tây Ninh",
            "Bình Phước", "Đồng Tháp", "Vũng Tàu", "Bà Rịa",
            "Hậu Giang", "Sóc Trăng", "Bạc Liêu", "Trà Vinh"
        ]
        
        # Tìm dòng chứa "Số:" hoặc dòng có ngày
        lines = text.split('\n')
        for line in lines:
            # Pattern: "Số: ... Hà Nội, ngày ..."
            for city in cities:
                # Tìm thành phố theo pattern: "Số: ... <City>, ngày"
                if re.search(rf'\b{re.escape(city)}\s*,\s*ngày', line, re.IGNORECASE):
                    return city
                # Hoặc chỉ tìm thành phố trong dòng có "Số:"
                if "Số:" in line and re.search(rf'\b{re.escape(city)}\b', line, re.IGNORECASE):
                    return city
        
        # Nếu không tìm được từ "Số:", tìm ở bất kỳ đâu trước "ngày"
        for line in lines:
            for city in cities:
                if re.search(rf'{re.escape(city)}\s*,\s*ngày', line, re.IGNORECASE):
                    return city
        
        return "Không xác định"

    @staticmethod
    def extract_all_info(text: str) -> Dict[str, str]:
        """
        Trích xuất tất cả thông tin từ văn bản OCR.
        """
        return {
            "ngay_thang_nam": DocumentInfoExtractor.extract_date(text),
            "loai_van_ban": DocumentInfoExtractor.detect_document_type(text),
            "so_hieu": DocumentInfoExtractor.extract_document_number(text),
            "tieu_de": DocumentInfoExtractor.extract_title(text),
            "trich_yeu": DocumentInfoExtractor.extract_summary(text),
            "noi_dung": DocumentInfoExtractor.extract_main_content(text),
            "thanh_pho": DocumentInfoExtractor.extract_city(text)
        }


# Khởi tạo extractor
info_extractor = DocumentInfoExtractor()