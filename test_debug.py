#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug script để in OCR text và trích xuất từng bước"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from app.core.information_extraction import info_extractor

# Tạo OCR text từ dữ liệu người dùng cung cấp
ocr_text = """THƯ MỜI
Viết bài tham gia Hội thảo khoa học: "Hóa học vì sự phát triển bền vững"

Các giải pháp nâng cao chất lượng đào tạo và nghiên cứu về Hóa học, Trường Đại học Thủ Dầu Một 
tổ chức Hội thảo khoa học với chủ đề: "Hóa học vì sự phát triển bền vững"

Kính gửi: Quý Thầy/Cô giáo,

Hội thảo khoa học "Hóa học vì sự phát triển bền vững" sẽ diễn ra...
"""

print("=" * 80)
print("📋 OCR TEXT:")
print("=" * 80)
print(ocr_text)

print("\n" + "=" * 80)
print("🔍 DEBUG EXTRACTION:")
print("=" * 80)

lines = ocr_text.strip().split('\n')
print(f"\n📍 Lines in OCR text:")
for i, line in enumerate(lines):
    print(f"   {i}: {repr(line)}")

print("\n" + "-" * 80)
print("🎯 EXTRACT_TITLE() DEBUG:")
print("-" * 80)

import re

# Cách 1: Tìm V/v, Về, etc.
print("\n[Cách 1] Tìm V/v, Về, Chuyên đề, Ban hành:")
found_way1 = False
for i, line in enumerate(lines):
    match = re.search(r'(?:V/v|Về|Chuyên đề|Ban hành)\s*[:–-]?\s*(.+)', line, re.IGNORECASE)
    if match:
        print(f"   ✓ Found at line {i}: {repr(line[:100])}")
        found_way1 = True
        break
if not found_way1:
    print("   ✗ Not found")

# Cách 2: Tìm loại văn bản
print("\n[Cách 2] Tìm loại văn bản rồi lấy dòng tiếp theo:")
doc_type_idx = -1
for i, line in enumerate(lines):
    if re.search(r'(công văn|quyết định|thông báo|giấy mời|tờ trình|báo cáo|kế hoạch|giấy mời họp)', 
                line, re.IGNORECASE):
        print(f"   ✓ Found doc_type at line {i}: {repr(line[:80])}")
        doc_type_idx = i
        break

if doc_type_idx >= 0:
    print(f"   Looking for title starting from line {doc_type_idx + 1}:")
    for i in range(doc_type_idx + 1, len(lines)):
        title = lines[i].strip()
        print(f"      Line {i}: {repr(title[:80])}")
        if title and not re.match(r'^(Kính\s*gửi|Gửi|Thực hiện)', title, re.IGNORECASE):
            print(f"      → Selected as title: {repr(title)}")
            break

# Actual extraction
print("\n" + "-" * 80)
print("✨ ACTUAL EXTRACTION RESULT:")
print("-" * 80)

info = info_extractor.extract_all_info(ocr_text)
for key, val in info.items():
    print(f"  {key:20} : {val}")
