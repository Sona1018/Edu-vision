import os
import re
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

# Try importing pytesseract
try:
    import pytesseract
    # Common windows paths if not in PATH
    default_win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe")
    ]
    for p in default_win_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

PAGE_IMAGES_BASE = Path("d:/IntelliPlace-AI-Answer-Grading-System-main/storage/page_images")
PAGE_IMAGES_BASE.mkdir(parents=True, exist_ok=True)

def clean_ocr_text(raw_text: str) -> str:
    """Cleans typical OCR noise, standardizes spaces, and preserves question numbers."""
    if not raw_text:
        return ""
    # Fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', raw_text)
    # Replace non-breaking spaces and tabs with normal space
    text = text.replace('\u00a0', ' ').replace('\t', ' ')
    # Normalize multiple newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text.strip()

def process_student_document(file_path: str, file_id: str) -> dict:
    """
    Pipeline 1:
    Processes student answer sheet (PDF or image).
    Renders pages as images for handwriting verification.
    Extracts text per page using PyMuPDF + Tesseract OCR.
    Splits into questions.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    output_dir = PAGE_IMAGES_BASE / file_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pages_data = []
    full_text_blocks = []
    
    if ext == ".pdf":
        doc = fitz.open(str(path))
        num_pages = len(doc)
        
        for idx in range(num_pages):
            page_num = idx + 1
            page = doc[idx]
            
            # 1. Render high-res image for handwriting verification
            pix = page.get_pixmap(dpi=150)
            img_rel_path = f"page_images/{file_id}/page_{page_num}.png"
            img_abs_path = output_dir / f"page_{page_num}.png"
            pix.save(str(img_abs_path))
            
            # 2. Extract text (PDF text layer first)
            page_text = page.get_text("text").strip()
            
            # 3. If PDF has no text layer or very short, try Tesseract OCR on rendered image
            if len(page_text.split()) < 10 and HAS_TESSERACT:
                try:
                    img = Image.open(str(img_abs_path))
                    ocr_text = pytesseract.image_to_string(img)
                    if len(ocr_text.strip()) > len(page_text):
                        page_text = ocr_text.strip()
                except Exception as e:
                    pass
            
            page_text_clean = clean_ocr_text(page_text)
            
            # Find which questions appear on this page
            qs_on_page = detect_questions_in_text(page_text_clean)
            
            pages_data.append({
                "page_number": page_num,
                "image_url": f"/storage/page_images/{file_id}/page_{page_num}.png",
                "extracted_text": page_text_clean,
                "word_count": len(page_text_clean.split()),
                "questions_detected": qs_on_page
            })
            
            full_text_blocks.append(f"==================== PAGE {page_num} ====================\n{page_text_clean}")
            
        doc.close()
    elif ext in [".png", ".jpg", ".jpeg"]:
        # Single image
        page_num = 1
        img_abs_path = output_dir / f"page_1.png"
        img = Image.open(str(path)).convert("RGB")
        img.save(str(img_abs_path), "PNG")
        
        page_text = ""
        if HAS_TESSERACT:
            try:
                page_text = pytesseract.image_to_string(img)
            except Exception:
                page_text = ""
                
        page_text_clean = clean_ocr_text(page_text)
        qs_on_page = detect_questions_in_text(page_text_clean)
        
        pages_data.append({
            "page_number": 1,
            "image_url": f"/storage/page_images/{file_id}/page_1.png",
            "extracted_text": page_text_clean,
            "word_count": len(page_text_clean.split()),
            "questions_detected": qs_on_page
        })
        full_text_blocks.append(f"==================== PAGE 1 ====================\n{page_text_clean}")
    else:
        # Fallback text file
        with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        page_text_clean = clean_ocr_text(raw)
        pages_data.append({
            "page_number": 1,
            "image_url": None,
            "extracted_text": page_text_clean,
            "word_count": len(page_text_clean.split()),
            "questions_detected": detect_questions_in_text(page_text_clean)
        })
        full_text_blocks.append(page_text_clean)

    combined_full_text = "\n\n".join(full_text_blocks)
    
    # Split entire student text by questions
    question_answers = split_student_questions(combined_full_text)
    
    # Calculate confidence indicator based on length and word characteristics
    total_words = sum(p["word_count"] for p in pages_data)
    if total_words > 200 and len(question_answers) >= 3:
        confidence = "High"
    elif total_words > 50:
        confidence = "Medium"
    else:
        confidence = "Low (Manual Review Recommended)"

    return {
        "file_id": file_id,
        "total_pages": len(pages_data),
        "total_words": total_words,
        "confidence": confidence,
        "pages": pages_data,
        "full_text": combined_full_text,
        "questions_parsed": question_answers,
        "extracted_answers": question_answers
    }

def detect_questions_in_text(text: str) -> list:
    """Finds Q1, Q2, Ques-3, Question 4, etc. on a specific page."""
    qs = set()
    patterns = [
        r'(?:^|\n|\s)(?:Q(?:ues)?\.?\s*[-:]?\s*(\d+))',
        r'(?:^|\n|\s)(?:Question\s*[-:]?\s*(\d+))',
        r'(?:^|\n|\s)(\d+)\s*[.)]\s*(?:Define|What|Explain|Differentiate|Discuss)'
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            qnum = m.group(1)
            if qnum and 1 <= int(qnum) <= 20:
                qs.add(f"Q{int(qnum)}")
    return sorted(list(qs), key=lambda x: int(x[1:]))

def split_student_questions(text: str) -> dict:
    """Splits full extracted answer text into Question-wise dictionary."""
    markers = []
    
    # Pattern 1: Q1, Q.1, Q-1, Ques-1, Ques1, Question 1
    for m in re.finditer(r'(?:^|\n)\s*(?:Q(?:ues)?\.?\s*[-:)]?\s*(\d+)|Question\s*[-:)]?\s*(\d+))', text, re.IGNORECASE | re.MULTILINE):
        qnum = m.group(1) or m.group(2)
        if qnum and 1 <= int(qnum) <= 20:
            markers.append((int(qnum), m.start()))
            
    # Pattern 2: "1. Define...", "2. What...", etc.
    for m in re.finditer(r'(?:^|\n)\s*(\d+)\s*[.\\)]\s*(?:Define|What|Explain|Differentiate|Discuss)', text, re.IGNORECASE | re.MULTILINE):
        qnum = m.group(1)
        if qnum and 1 <= int(qnum) <= 20:
            if int(qnum) not in [q for q, _ in markers]:
                markers.append((int(qnum), m.start()))
                
    # Pattern 3: Fallback keyword search for Q1-Q5
    if len(markers) < 2:
        q_keyword_map = [
            (1, r'(?:^|\n)\s*(?:Q\s+)?(?:Define|Definition of)\s+(?:market|Marketing)'),
            (2, r'(?:^|\n)\s*(?:Q\s+)?(?:What\s+all\s+are\s+)?(?:scope|importance)\s+of\s+marketing'),
            (3, r'(?:^|\n)\s*(?:Q\s+)?(?:Differentiate|needs?\s+and\s+wants?)'),
            (4, r'(?:^|\n)\s*(?:Q\s+)?(?:Explain\s+)?(?:marketing\s+environment|internal\s+and\s+external)'),
            (5, r'(?:^|\n)\s*(?:Q\s+)?(?:Explain\s+)?(?:micro\s+and\s+macro|external\s+environmental)')
        ]
        for qnum, kw_pat in q_keyword_map:
            m = re.search(kw_pat, text, re.IGNORECASE)
            if m and qnum not in [q for q, _ in markers]:
                markers.append((qnum, m.start()))

    markers.sort(key=lambda x: x[1])
    
    # Remove duplicates
    seen = set()
    unique_markers = []
    for qnum, pos in markers:
        if qnum not in seen:
            seen.add(qnum)
            unique_markers.append((qnum, pos))
    markers = unique_markers
    
    answers = {}
    if not markers:
        # If no question markers identified, store entire text as Q1
        answers["Q1"] = text.strip()
        return answers
        
    for i, (qnum, start) in enumerate(markers):
        end = markers[i + 1][1] if i + 1 < len(markers) else len(text)
        ans_content = text[start:end].strip()
        # Clean out leading page boundary artifacts
        ans_content = re.sub(r'={10,}\s*PAGE\s*\d+\s*={10,}', '', ans_content).strip()
        answers[f"Q{qnum}"] = ans_content
        
    return answers
