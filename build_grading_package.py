"""
Master script to build the complete grading dataset package.
Reads all 24 student .txt files (some may be empty/missing),
generates rubric, annotated texts, CSV, Excel, Python helper, and ZIP.
"""
import re
import os
import json
import csv
import zipfile
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
UPLOAD_DIR = r"C:\Users\Dell\.gemini\antigravity\brain\63fde9ca-f083-42fb-8c6f-3b13ede6c873\.user_uploaded"
WORK_DIR = r"d:\IntelliPlace-AI-Answer-Grading-System-main\grading_package"
STUDENT_DIR = os.path.join(WORK_DIR, "annotated_student_texts")
os.makedirs(STUDENT_DIR, exist_ok=True)

TOTAL_PAPER_MARKS_DEFAULT = 100

# Map: label -> actual uploaded filename
FILE_MAP = {
    "1.txt": "media_1795111441483_913a592a.txt",
    "2.txt": "media_1795111441483_f5333143.txt",
    "3.txt": "media_1795111441483_dc3b4fea.txt",
    "4.txt": "media_1795111441483_4743f754.txt",
    "5.txt": "media_1795111441483_480a29a4.txt",
    "6.txt": "media_1795111441483_b9051155.txt",
    "7.txt": "media_1795111441483_77b5db29.txt",
    "8.txt": "media_1795111441483_84cf74b1.txt",
    "9.txt": "media_1795111441486_67d0bc78.txt",
    "10.txt": "media_1795111441488_4b7482ab.txt",
    "11.txt": "media_1795111441489_ef940a61.txt",
    "12.txt": "media_1795111441492_3a45ee43.txt",
    "13.txt": "media_1795111441490_9ec942ba.txt",
    "14.txt": "media_1795111804016_f2826dc6.txt",
    "15.txt": "media_1795111804006_4e57704f.txt",
    "16.txt": "media_1795111804054_89cf3dad.txt",
    "17.txt": "media_1795111804055_ada85150.txt",
    "18.txt": "media_1795111804055_3f528e2c.txt",
    "19.txt": "media_1795111804105_35473996.txt",
    "20.txt": "media_1795111804056_29f04491.txt",
    "21.txt": "media_1795111804056_3372c904.txt",
    "23.txt": "media_1795111804057_8cb18126.txt",
    "24.txt": "media_1795111804054_41512c02.txt",
    "25.txt": "media_1795111804056_2b67f0bf.txt",
}

# ============================================================
# STEP 1: Read all files
# ============================================================
print("=" * 60)
print("STEP 1: Reading all student files...")
print("=" * 60)

student_texts = {}
for label in sorted(FILE_MAP.keys(), key=lambda x: int(x.split('.')[0])):
    fname = FILE_MAP[label]
    src = os.path.join(UPLOAD_DIR, fname)
    try:
        with open(src, 'r', encoding='utf-8') as f:
            text = f.read()
    except:
        try:
            with open(src, 'r', encoding='latin-1') as f:
                text = f.read()
        except:
            text = ""
    student_texts[label] = text
    status = f"{len(text.split())} words" if text.strip() else "EMPTY"
    print(f"  {label}: {status}")

# ============================================================
# STEP 1b: Extract student info and split answers per question
# ============================================================
def extract_student_info(text, label):
    """Extract name, enrollment, batch from text."""
    info = {"file": label, "name": "Unknown", "enrollment": "", "batch": "CSE-3"}
    
    name_match = re.search(r'Name[:\s-]*([A-Za-z\s]+)', text, re.IGNORECASE)
    if name_match:
        info["name"] = name_match.group(1).strip()
    else:
        # Try first few lines
        lines = text.strip().split('\n')
        for line in lines[:6]:
            line = line.strip()
            # Skip lines that are question text, assignment headers, numbers
            if re.match(r'^(\d+|Q\d|Ques|Assignment|Marketing|MM |UNIT|BTECH|CSE|HMC|Date|Branch|ॐ|$)', line, re.IGNORECASE):
                continue
            if re.match(r'^[A-Za-z][A-Za-z\s]{2,30}$', line):
                info["name"] = line.strip()
                break
    
    enroll_match = re.search(r'(?:Enroll(?:ment)?\s*(?:No\.?)?|Roll\s*No\.?)[:\s-]*(\d+)', text, re.IGNORECASE)
    if enroll_match:
        info["enrollment"] = enroll_match.group(1).strip()
    else:
        # Look for standalone enrollment number pattern
        enroll_match2 = re.search(r'\b(\d{11,13})\b', text[:500])
        if enroll_match2:
            info["enrollment"] = enroll_match2.group(1)
    
    return info

def split_into_questions(text):
    """Split student text into individual question answers."""
    markers = []
    
    # Pattern 1: Q1, Q.1, Q-1, Ques-1, Ques1, Q 1, etc.
    for m in re.finditer(r'(?:^|\n)\s*(?:Q(?:ues)?\.?\s*[-:)]?\s*(\d+))', text, re.IGNORECASE | re.MULTILINE):
        qnum = m.group(1)
        if qnum and 1 <= int(qnum) <= 10:
            markers.append((int(qnum), m.start()))
    
    # Pattern 2: "1. Define..." or "1\. Define..."
    for m in re.finditer(r'(?:^|\n)\s*(\d+)\s*[.\\)]\s*(?:Define|What|Explain|Differentiate|Discuss)', text, re.IGNORECASE | re.MULTILINE):
        qnum = m.group(1)
        if qnum and 1 <= int(qnum) <= 10:
            if int(qnum) not in [q for q, _ in markers]:
                markers.append((int(qnum), m.start()))
    
    # Pattern 3: "Q- Define..." or "Q Define..." (no number) — assign Q1-Q5 by order
    if len(markers) == 0:
        q_dashes = list(re.finditer(r'(?:^|\n)\s*Q[-\s]+(?:Define|What|Explain|Differentiate|Discuss)', text, re.IGNORECASE | re.MULTILINE))
        if q_dashes:
            # Map known question keywords to question numbers
            q_keyword_map = [
                (1, r'Define\s+(?:market|Marketing)'),
                (2, r'(?:scope|importance)\s+of\s+marketing'),
                (3, r'(?:Differentiate|needs?\s+and\s+wants?)'),
                (4, r'(?:marketing\s+environment|internal\s+and\s+external)'),
                (5, r'(?:micro\s+and\s+macro|external\s+environmental)'),
            ]
            for m in q_dashes:
                snippet = text[m.start():min(m.start() + 120, len(text))]
                assigned = False
                for qnum, kw_pat in q_keyword_map:
                    if re.search(kw_pat, snippet, re.IGNORECASE):
                        if qnum not in [q for q, _ in markers]:
                            markers.append((qnum, m.start()))
                            assigned = True
                            break
                if not assigned:
                    # Sequential fallback
                    next_q = len(markers) + 1
                    if next_q <= 5:
                        markers.append((next_q, m.start()))
    
    # Pattern 4: "Q Define" without dash (single Q)
    if not any(q == 1 for q, _ in markers):
        m = re.search(r'(?:^|\n)\s*Q\s+Define', text, re.IGNORECASE | re.MULTILINE)
        if m:
            markers.append((1, m.start()))
    
    # Also check for "Q What" for Q2
    if not any(q == 2 for q, _ in markers):
        m = re.search(r'(?:^|\n)\s*Q[-\s]*What\s+(?:all\s+)?(?:is|are)\s+(?:the\s+)?scope', text, re.IGNORECASE | re.MULTILINE)
        if m:
            markers.append((2, m.start()))
    
    markers.sort(key=lambda x: x[1])
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_markers = []
    for qnum, pos in markers:
        if qnum not in seen:
            seen.add(qnum)
            unique_markers.append((qnum, pos))
    markers = unique_markers
    
    answers = {}
    for i, (qnum, start) in enumerate(markers):
        end = markers[i + 1][1] if i + 1 < len(markers) else len(text)
        answer_text = text[start:end].strip()
        answers[qnum] = answer_text
    
    return answers

students = []
for label in sorted(student_texts.keys(), key=lambda x: int(x.split('.')[0])):
    text = student_texts[label]
    if not text.strip():
        continue
    info = extract_student_info(text, label)
    answers = split_into_questions(text)
    info["questions_attempted"] = sorted(answers.keys())
    info["answers"] = answers
    info["full_text"] = text
    students.append(info)

print(f"\nValid students parsed: {len(students)}")
for s in students:
    print(f"  {s['file']}: {s['name'][:25]:<25} Enroll: {s['enrollment']:<15} Qs: {s['questions_attempted']}")

# ============================================================
# STEP 2: Create Dynamic Reference Rubric
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Creating Dynamic Reference Rubric...")
print("=" * 60)

RUBRIC = {
    "TOTAL_PAPER_MARKS": 100,
    "subject": "Marketing Management",
    "unit": "Unit-1",
    "questions": [
        {
            "question_id": "Q1",
            "question_text": "Define Marketing. Explain the nature of marketing along with examples.",
            "question_type": "Factual_Definition",
            "weight_pct": 20.0,
            "scaled_max_marks_100": 20.0,
            "scaled_max_marks_50": 10.0,
            "model_reference_answer": (
                "Marketing is the social and managerial process by which individuals and organizations obtain what they need and want "
                "through identifying, creating, communicating, delivering, and exchanging offerings of value with others. According to "
                "the American Marketing Association (AMA), marketing is the activity, set of institutions, and processes for creating, "
                "communicating, delivering, and exchanging offerings that have value for customers, clients, partners, and society at large.\n\n"
                "Marketing can be expressed as: Marketing = Understanding Customer Needs + Creating Value + Delivering Satisfaction.\n\n"
                "Nature of Marketing:\n"
                "1. Customer-Oriented: Modern marketing begins with understanding consumer behaviour, demand patterns and expectations. "
                "Organizations design products based on customer insights. Example: Netflix uses data analytics to recommend personalized content.\n"
                "2. Continuous and Dynamic Process: Marketing is an ongoing activity involving research, development, promotion, distribution "
                "and post-sale service. It evolves with technology and consumer preferences.\n"
                "3. Value-Centered: Marketing focuses on creating and delivering superior value through the marketing mix (Product, Price, Place, Promotion). "
                "Example: Apple creates premium value through design, ecosystem integration and brand experience.\n"
                "4. Exchange-Based: Marketing operates on the principle of voluntary exchange where both buyer and seller gain value. "
                "It ensures mutual benefit and long-term relationship development.\n"
                "5. Integrated Activity: Marketing coordinates with production, finance, R&D and HR to achieve organizational objectives. "
                "It is a cross-functional activity requiring alignment across departments.\n"
                "6. Goal-Oriented: Marketing activities are directed towards achieving specific organizational goals such as profitability, "
                "market share, and brand recognition.\n"
                "7. Dynamic in Nature: Marketing strategies must adapt to changes in technology, competition, regulations, and consumer "
                "behaviour. Example: Companies shifted to digital marketing during COVID-19."
            ),
            "mandatory_key_concepts": [
                "Definition of Marketing (AMA or Kotler definition)",
                "Marketing equation: Understanding + Creating + Delivering",
                "Customer-Oriented / Customer-Centric approach",
                "Continuous and Dynamic Process",
                "Value Creation and Delivery",
                "Exchange-Based principle (mutual benefit)",
                "Integrated / Cross-functional Activity",
                "Goal-Oriented nature",
                "Real-world examples (Netflix, Apple, Zomato, etc.)"
            ],
            "sub_concepts": [
                {"concept": "Definition of Marketing", "weight_within_q": 0.15},
                {"concept": "Customer-Oriented nature", "weight_within_q": 0.15},
                {"concept": "Continuous and Dynamic Process", "weight_within_q": 0.10},
                {"concept": "Value-Centered", "weight_within_q": 0.15},
                {"concept": "Exchange-Based", "weight_within_q": 0.10},
                {"concept": "Integrated Activity", "weight_within_q": 0.10},
                {"concept": "Goal-Oriented", "weight_within_q": 0.10},
                {"concept": "Real-world Examples", "weight_within_q": 0.15},
            ]
        },
        {
            "question_id": "Q2",
            "question_text": "What all are the scope of marketing and importance of marketing?",
            "question_type": "Analytical_Explanation",
            "weight_pct": 20.0,
            "scaled_max_marks_100": 20.0,
            "scaled_max_marks_50": 10.0,
            "model_reference_answer": (
                "Scope of Marketing:\n"
                "Marketing applies to a wide range of entities and activities:\n"
                "1. Marketing of Goods & Services: Physical products (electronics, FMCG) and intangible services (banking, healthcare, education).\n"
                "2. Marketing of Ideas & Social Causes: Social campaigns like Swachh Bharat Abhiyan, road safety awareness, anti-smoking drives.\n"
                "3. Marketing of Experiences: Companies sell experiences, not just products. Examples: theme parks (Disneyland), luxury hotels, concerts.\n"
                "4. Place Marketing: Tourism promotion of cities, countries, states. Example: 'Incredible India' campaign.\n"
                "5. Person Marketing: Building personal brands - celebrities, influencers, politicians. Example: Personal branding on LinkedIn.\n"
                "6. Digital and Online Marketing: Social media marketing, content marketing, SEO, email marketing, influencer marketing.\n"
                "7. Market Research: Collecting and analysing data about customers, competitors and market trends before product launch.\n"
                "8. Product Planning & Development: Deciding product design, features, packaging, branding and quality.\n"
                "9. Pricing: Fixing product price based on cost, competition and customer demand.\n"
                "10. Promotion: Communicating product information through advertising, sales promotion, social media.\n"
                "11. Distribution (Place): Ensuring product reaches customers through proper channels.\n\n"
                "Importance of Marketing:\n"
                "A. Importance to Business:\n"
                "  - Helps in sales growth and revenue generation\n"
                "  - Creates and maintains brand image\n"
                "  - Facilitates market expansion\n"
                "  - Creates competitive advantage\n"
                "B. Importance to Consumers:\n"
                "  - Satisfies needs and wants\n"
                "  - Provides better quality products and more choices\n"
                "  - Improves standard of living\n"
                "  - Creates time, place and possession utility\n"
                "C. Importance to Society:\n"
                "  - Generates employment opportunities\n"
                "  - Encourages innovation and technological progress\n"
                "  - Supports economic growth and development\n"
                "  - Promotes social welfare through CSR activities"
            ),
            "mandatory_key_concepts": [
                "Marketing of Goods, Services, Ideas, Experiences, Places, Persons",
                "Market Research as scope",
                "Product Planning & Development",
                "Pricing strategies",
                "Promotion (advertising, social media, sales promotion)",
                "Distribution / Place channels",
                "Digital / Online Marketing",
                "Importance to Business (sales, brand, competition)",
                "Importance to Consumers (satisfaction, quality, choices)",
                "Importance to Society (employment, innovation, economic growth)"
            ],
            "sub_concepts": [
                {"concept": "Scope - Various marketing applications", "weight_within_q": 0.25},
                {"concept": "Scope - Marketing Mix functions (4Ps)", "weight_within_q": 0.25},
                {"concept": "Importance to Business", "weight_within_q": 0.20},
                {"concept": "Importance to Consumers", "weight_within_q": 0.15},
                {"concept": "Importance to Society", "weight_within_q": 0.15},
            ]
        },
        {
            "question_id": "Q3",
            "question_text": "Differentiate between Needs and Wants. Explain how marketing influences both.",
            "question_type": "Analytical_Explanation",
            "weight_pct": 20.0,
            "scaled_max_marks_100": 20.0,
            "scaled_max_marks_50": 10.0,
            "model_reference_answer": (
                "Difference between Needs and Wants:\n\n"
                "| Basis | Needs | Wants |\n"
                "|-------|-------|-------|\n"
                "| Definition | Basic human requirements essential for survival | Specific desires shaped by culture, personality, lifestyle |\n"
                "| Nature | Limited and universal | Unlimited and vary by individual |\n"
                "| Origin | Biological and psychological | Social, cultural and personal preferences |\n"
                "| Examples | Food, water, shelter, clothing, communication | Pizza, iPhone, BMW, branded clothes, luxury items |\n"
                "| Marketing Role | Cannot be created by marketing | Shaped and influenced by marketing |\n"
                "| Urgency | Immediate and essential | Can be deferred or substituted |\n\n"
                "How Marketing Influences Needs and Wants:\n\n"
                "1. Marketing Does NOT Create Needs: Needs like food, safety, shelter and communication already exist as fundamental "
                "human requirements. Marketing cannot create these needs.\n\n"
                "2. Marketing Shapes and Channels Wants:\n"
                "   - Through Advertising: Creates awareness and desire for specific products. TV ads, social media campaigns influence "
                "     which brand consumers prefer.\n"
                "   - Through Branding: Strong brand identity makes consumers prefer one product over another. Example: Nike vs local shoes.\n"
                "   - Through Lifestyle Positioning: Marketing positions products as part of a desirable lifestyle. Example: Apple "
                "     positions its products as symbols of innovation and status.\n"
                "   - Through Attractive Packaging: Visual appeal and packaging design influence purchase decisions.\n\n"
                "3. Marketing Creates Demand Through Awareness:\n"
                "   - Informative advertising makes consumers aware of new solutions to existing needs.\n"
                "   - Example: Need = communication → Want = iPhone (influenced by Apple's marketing).\n"
                "   - Example: Need = transportation → Want = BMW (influenced by brand positioning)."
            ),
            "mandatory_key_concepts": [
                "Definition of Needs (basic, essential, survival)",
                "Definition of Wants (desires, culture-shaped, unlimited)",
                "Tabular/point-wise comparison (at least 4 bases)",
                "Marketing does NOT create needs",
                "Marketing shapes/channels wants",
                "Influence through Advertising",
                "Influence through Branding",
                "Influence through Lifestyle Positioning / Brand Image",
                "Influence through Packaging",
                "Real-world examples (iPhone, BMW, Nike, etc.)"
            ],
            "sub_concepts": [
                {"concept": "Difference Table/Comparison", "weight_within_q": 0.35},
                {"concept": "Marketing does not create needs", "weight_within_q": 0.10},
                {"concept": "Marketing shapes wants (4 mechanisms)", "weight_within_q": 0.35},
                {"concept": "Real-world examples", "weight_within_q": 0.20},
            ]
        },
        {
            "question_id": "Q4",
            "question_text": "Explain Marketing Environment. Define Internal and External Marketing Environment.",
            "question_type": "Analytical_Explanation",
            "weight_pct": 20.0,
            "scaled_max_marks_100": 20.0,
            "scaled_max_marks_50": 10.0,
            "model_reference_answer": (
                "Marketing Environment:\n"
                "Marketing environment refers to the sum total of all internal and external factors/forces that influence a company's "
                "marketing decisions, strategies and performance. It affects pricing, product design, promotional strategy and distribution.\n\n"
                "Characteristics of Marketing Environment:\n"
                "1. Dynamic in Nature: Constantly changing due to technological, social and economic shifts.\n"
                "2. Complex and Interrelated: Different factors influence each other. Example: Technological change affects social behaviour.\n"
                "3. Uncontrollable (External): Organizations cannot control political, economic or demographic conditions; they can only adapt.\n"
                "4. Requires Continuous Monitoring: Environmental scanning helps identify opportunities and threats.\n\n"
                "Types of Marketing Environment:\n\n"
                "A. Internal Marketing Environment:\n"
                "Forces within the organization that are controllable by management:\n"
                "- Top Management: Policies, vision, strategic direction\n"
                "- Employees: Skills, motivation, productivity\n"
                "- Organizational Structure: Hierarchy, authority, responsibility\n"
                "- Company Culture & Values: Influence brand image and customer relationships\n"
                "- Financial Resources: Budget allocation for marketing activities\n"
                "- R&D Capabilities: Innovation and product development capacity\n"
                "- Company Objectives: Marketing strategies must align with business goals\n\n"
                "B. External Marketing Environment:\n"
                "Forces outside the organization, divided into:\n\n"
                "i. Micro Environment (Task Environment):\n"
                "Forces close to the company directly affecting operations:\n"
                "- Customers: Target market segments, their preferences and demand\n"
                "- Suppliers: Raw materials, components, reliability affects production\n"
                "- Competitors: Rival firms influencing pricing, quality, innovation\n"
                "- Marketing Intermediaries: Distributors, wholesalers, retailers\n"
                "- Publics: Media, financial institutions, government, local communities\n\n"
                "ii. Macro Environment (PESTLE):\n"
                "Broader societal forces affecting all organizations:\n"
                "- Political: Government policies, taxation, trade regulations\n"
                "- Economic: Inflation, GDP, interest rates, unemployment\n"
                "- Social/Cultural: Demographics, values, lifestyle changes\n"
                "- Technological: Innovation, AI, digital transformation\n"
                "- Legal: Consumer protection, labor laws, environmental regulations\n"
                "- Environmental: Climate change, sustainability, green policies"
            ),
            "mandatory_key_concepts": [
                "Definition of Marketing Environment",
                "Characteristics (Dynamic, Complex, Uncontrollable, Continuous Monitoring)",
                "Internal Environment - definition and components",
                "Top Management, Employees, Structure, Culture, Resources",
                "External Environment - Micro (task environment)",
                "Micro: Customers, Suppliers, Competitors, Intermediaries, Publics",
                "External Environment - Macro (PESTLE)",
                "Macro: Political, Economic, Social, Technological, Legal, Environmental"
            ],
            "sub_concepts": [
                {"concept": "Definition & Characteristics", "weight_within_q": 0.20},
                {"concept": "Internal Environment components", "weight_within_q": 0.25},
                {"concept": "Micro External Environment", "weight_within_q": 0.25},
                {"concept": "Macro External Environment (PESTLE)", "weight_within_q": 0.30},
            ]
        },
        {
            "question_id": "Q5",
            "question_text": "Explain Micro and Macro External Environmental Factors.",
            "question_type": "Analytical_Explanation",
            "weight_pct": 20.0,
            "scaled_max_marks_100": 20.0,
            "scaled_max_marks_50": 10.0,
            "model_reference_answer": (
                "Micro External Environmental Factors:\n"
                "These are forces close to the company that directly affect its day-to-day operations and marketing decisions.\n\n"
                "Components:\n"
                "1. Customers: Target market segments who purchase products/services. Their preferences, buying behaviour and "
                "satisfaction directly affect demand and revenue. Companies must continuously understand changing customer needs.\n"
                "2. Competitors: Rival firms offering similar products/services. They influence pricing strategies, quality standards, "
                "and push innovation. Example: Coca-Cola vs Pepsi continuously innovate to stay competitive.\n"
                "3. Suppliers: Provide raw materials, components or services needed for production. Their reliability, pricing and "
                "quality directly affect production cost and product quality.\n"
                "4. Marketing Intermediaries: Distributors, wholesalers, retailers, logistics companies that help in product "
                "distribution and market reach. Example: Amazon, Flipkart as online intermediaries.\n"
                "5. Publics: Any group that has actual or potential interest in the company - media, financial institutions, "
                "government agencies, citizen action groups, local communities.\n\n"
                "Macro External Environmental Factors:\n"
                "These are broader uncontrollable societal forces that affect all industries and organizations.\n\n"
                "Commonly analyzed using PESTLE framework:\n"
                "1. Political Factors: Government policies, taxation, trade regulations, political stability, foreign trade policies.\n"
                "2. Economic Factors: Inflation rate, GDP growth, interest rates, unemployment levels, exchange rates, purchasing power.\n"
                "3. Social/Cultural Factors: Demographics, cultural values, lifestyle changes, education levels, population trends, "
                "consumer attitudes.\n"
                "4. Technological Factors: Innovation, automation, artificial intelligence, digital transformation, R&D investment, "
                "technology adoption rate.\n"
                "5. Legal Factors: Consumer protection laws, labor laws, intellectual property rights, advertising standards, "
                "data protection regulations (GDPR).\n"
                "6. Environmental Factors: Climate change, sustainability requirements, environmental regulations, green marketing, "
                "CSR initiatives."
            ),
            "mandatory_key_concepts": [
                "Micro Environment definition (close to company, direct impact)",
                "Customers - preferences, buying behaviour",
                "Competitors - rivalry, pricing influence",
                "Suppliers - raw materials, reliability",
                "Marketing Intermediaries - distributors, retailers",
                "Publics - media, government, communities",
                "Macro Environment definition (broader, uncontrollable)",
                "PESTLE framework mention",
                "Political Factors",
                "Economic Factors",
                "Social/Cultural Factors",
                "Technological Factors",
                "Legal Factors",
                "Environmental Factors"
            ],
            "sub_concepts": [
                {"concept": "Micro definition + Customers", "weight_within_q": 0.15},
                {"concept": "Competitors + Suppliers", "weight_within_q": 0.15},
                {"concept": "Intermediaries + Publics", "weight_within_q": 0.10},
                {"concept": "Macro definition + PESTLE intro", "weight_within_q": 0.10},
                {"concept": "Political + Economic factors", "weight_within_q": 0.15},
                {"concept": "Social + Cultural factors", "weight_within_q": 0.10},
                {"concept": "Technological factors", "weight_within_q": 0.10},
                {"concept": "Legal + Environmental factors", "weight_within_q": 0.15},
            ]
        },
    ]
}

# Save rubric as JSON
rubric_json_path = os.path.join(WORK_DIR, "master_reference_rubric.json")
with open(rubric_json_path, 'w', encoding='utf-8') as f:
    json.dump(RUBRIC, f, indent=2, ensure_ascii=False)
print(f"  Saved: {rubric_json_path}")

# Save rubric as TXT
rubric_txt_path = os.path.join(WORK_DIR, "master_reference_rubric.txt")
with open(rubric_txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("INTELLIPLACE — MASTER REFERENCE RUBRIC & ANSWER KEY\n")
    f.write(f"Subject: {RUBRIC['subject']} | Unit: {RUBRIC['unit']}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"TOTAL_PAPER_MARKS (configurable): {RUBRIC['TOTAL_PAPER_MARKS']}\n")
    f.write("NOTE: All marks scale dynamically. If TOTAL_PAPER_MARKS=50, divide all by 2.\n\n")
    
    for q in RUBRIC["questions"]:
        f.write("-" * 80 + "\n")
        f.write(f"Question_ID: {q['question_id']}\n")
        f.write(f"Question_Type: {q['question_type']}\n")
        f.write(f"Question: {q['question_text']}\n")
        f.write(f"Weight_Percentage: {q['weight_pct']}%\n")
        f.write(f"Scaled_Max_Marks (100-scale): {q['scaled_max_marks_100']}\n")
        f.write(f"Scaled_Max_Marks (50-scale): {q['scaled_max_marks_50']}\n\n")
        f.write("MODEL REFERENCE ANSWER:\n")
        f.write(q['model_reference_answer'] + "\n\n")
        f.write("MANDATORY KEY CONCEPTS:\n")
        for kc in q['mandatory_key_concepts']:
            f.write(f"  • {kc}\n")
        f.write("\nSUB-CONCEPT WEIGHTS (within question):\n")
        for sc in q['sub_concepts']:
            f.write(f"  - {sc['concept']}: {sc['weight_within_q']*100:.0f}%\n")
        f.write("\n")
    
print(f"  Saved: {rubric_txt_path}")

# ============================================================
# STEP 3: Score estimation helper
# ============================================================
def estimate_score(answer_text, question):
    """Estimate a realistic baseline score (0-100 pct) for a student answer."""
    if not answer_text or len(answer_text.split()) < 10:
        return 5.0
    
    text_lower = answer_text.lower()
    word_count = len(answer_text.split())
    
    # Check key concepts coverage
    key_concepts = question["mandatory_key_concepts"]
    concepts_found = 0
    for concept in key_concepts:
        # Split concept into keywords and check if at least some appear
        keywords = [w.lower() for w in concept.split() if len(w) > 3]
        if not keywords:
            continue
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches >= len(keywords) * 0.4:  # At least 40% of keywords present
            concepts_found += 1
    
    coverage_pct = (concepts_found / max(1, len(key_concepts))) * 100
    
    # Length factor
    if word_count < 30:
        length_factor = 0.4
    elif word_count < 80:
        length_factor = 0.6
    elif word_count < 150:
        length_factor = 0.8
    elif word_count < 300:
        length_factor = 0.95
    else:
        length_factor = 1.0
    
    # Structure bonus (has numbered points, bullets, examples)
    structure_bonus = 0
    if re.search(r'\d+\.\s', answer_text):
        structure_bonus += 5
    if re.search(r'[•\-→]\s', answer_text):
        structure_bonus += 3
    if re.search(r'example|e\.g\.|for instance', text_lower):
        structure_bonus += 5
    
    # Combine
    raw_score = coverage_pct * length_factor + structure_bonus
    return round(min(95.0, max(5.0, raw_score)), 1)

# ============================================================
# STEP 3: Annotate student texts
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Creating annotated student texts...")
print("=" * 60)

q_type_map = {q["question_id"]: q["question_type"] for q in RUBRIC["questions"]}
q_weight_map = {q["question_id"]: q["weight_pct"] for q in RUBRIC["questions"]}

for student in students:
    label = student["file"]
    text = student["full_text"]
    
    # Build metadata header
    qs_attempted = ",".join(f"Q{q}" for q in student["questions_attempted"])
    
    # Compute scale scores
    n_qs = len(student["questions_attempted"])
    total_weight_attempted = sum(q_weight_map.get(f"Q{q}", 0) for q in student["questions_attempted"])
    max_100 = total_weight_attempted
    max_50 = total_weight_attempted / 2
    
    header = (
        f"[METADATA]\n"
        f"Student_ID: {student['enrollment'] or label.replace('.txt','')}\n"
        f"Student_Name: {student['name']}\n"
        f"File: {label}\n"
        f"Paper_Scale_100: {max_100}\n"
        f"Paper_Scale_50: {max_50}\n"
        f"Questions_Attempted: {qs_attempted}\n"
        f"Total_Questions_Attempted: {n_qs}\n"
        f"[/METADATA]\n\n"
    )
    
    # Insert Q_START tags
    annotated = header
    
    # Re-split and tag
    answers = student["answers"]
    # Sort by position in original text
    full_text = student["full_text"]
    
    for qnum in sorted(answers.keys()):
        qid = f"Q{qnum}"
        qtype = q_type_map.get(qid, "Unknown")
        qweight = q_weight_map.get(qid, 0)
        
        annotated += f"[Q_START: {qid} | Type: {qtype} | Weight: {qweight}%]\n"
        annotated += answers[qnum] + "\n"
        annotated += f"[Q_END: {qid}]\n\n"
    
    # Save annotated file
    out_path = os.path.join(STUDENT_DIR, label)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(annotated)
    print(f"  Annotated: {label}")

print(f"  Total annotated files: {len(students)}")

# ============================================================
# STEP 4: Generate CSV dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Generating grading benchmark dataset...")
print("=" * 60)

rows = []
for student in students:
    for qnum in student["questions_attempted"]:
        qid = f"Q{qnum}"
        q_data = None
        for q in RUBRIC["questions"]:
            if q["question_id"] == qid:
                q_data = q
                break
        if not q_data:
            continue
        
        answer_text = student["answers"].get(qnum, "")
        estimated_pct = estimate_score(answer_text, q_data)
        
        max_100 = q_data["scaled_max_marks_100"]
        max_50 = q_data["scaled_max_marks_50"]
        expected_100 = round(estimated_pct / 100 * max_100, 2)
        expected_50 = round(estimated_pct / 100 * max_50, 2)
        
        rows.append({
            "Student_ID": student["enrollment"] or student["file"].replace(".txt", ""),
            "Student_Name": student["name"],
            "File_Name": student["file"],
            "Question_ID": qid,
            "Question_Type": q_data["question_type"],
            "Weight_Pct": q_data["weight_pct"],
            "Max_Marks_100_Scale": max_100,
            "Max_Marks_50_Scale": max_50,
            "Student_Extracted_Answer": answer_text[:2000],  # Truncate for CSV
            "Reference_Answer": q_data["model_reference_answer"][:2000],
            "Estimated_Score_Pct": estimated_pct,
            "Expected_Score_100_Scale": expected_100,
            "Expected_Score_50_Scale": expected_50,
        })

# Save CSV
csv_path = os.path.join(WORK_DIR, "grading_benchmark_dataset.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    if rows:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
print(f"  Saved CSV: {csv_path} ({len(rows)} rows)")

# ============================================================
# STEP 4b: Generate Excel
# ============================================================
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("  WARNING: openpyxl not installed. Excel file will be skipped.")

if HAS_OPENPYXL:
    wb = openpyxl.Workbook()
    
    # --- Tab 1: Summary Scores ---
    ws1 = wb.active
    ws1.title = "Summary_Scores"
    
    # Header row with TOTAL_PAPER_MARKS config cell
    ws1['A1'] = "TOTAL_PAPER_MARKS:"
    ws1['B1'] = 100
    ws1['A1'].font = Font(bold=True, color="FF0000")
    ws1['B1'].font = Font(bold=True, color="FF0000", size=14)
    ws1['B1'].number_format = '0'
    ws1['C1'] = "← Change this to 50 or 100 to auto-scale all marks"
    ws1['C1'].font = Font(italic=True, color="808080")
    
    # Column headers
    headers = ["Student_ID", "Student_Name", "File_Name", 
               "Questions_Attempted", "Total_Max_Marks", "Total_Expected_Score",
               "Percentage"]
    for col, h in enumerate(headers, 1):
        cell = ws1.cell(row=3, column=col, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")
    
    # Data rows
    student_summary = {}
    for row_data in rows:
        sid = row_data["Student_ID"]
        if sid not in student_summary:
            student_summary[sid] = {
                "name": row_data["Student_Name"],
                "file": row_data["File_Name"],
                "questions": [],
                "total_weight": 0,
                "total_expected": 0,
            }
        student_summary[sid]["questions"].append(row_data["Question_ID"])
        student_summary[sid]["total_weight"] += row_data["Weight_Pct"]
        student_summary[sid]["total_expected"] += row_data["Estimated_Score_Pct"] / 100 * row_data["Weight_Pct"]
    
    for r_idx, (sid, info) in enumerate(student_summary.items(), 4):
        ws1.cell(row=r_idx, column=1, value=sid)
        ws1.cell(row=r_idx, column=2, value=info["name"])
        ws1.cell(row=r_idx, column=3, value=info["file"])
        ws1.cell(row=r_idx, column=4, value=", ".join(info["questions"]))
        # Dynamic formula: total_weight% * B1 / 100
        ws1.cell(row=r_idx, column=5).value = f'={info["total_weight"]}*$B$1/100'
        ws1.cell(row=r_idx, column=5).number_format = '0.00'
        # Expected score scaled
        ws1.cell(row=r_idx, column=6).value = f'={round(info["total_expected"],2)}*$B$1/100'
        ws1.cell(row=r_idx, column=6).number_format = '0.00'
        # Percentage
        ws1.cell(row=r_idx, column=7).value = f'=IF(E{r_idx}>0, F{r_idx}/E{r_idx}*100, 0)'
        ws1.cell(row=r_idx, column=7).number_format = '0.0"%"'
    
    # --- Tab 2: Question_Wise_Breakdown ---
    ws2 = wb.create_sheet("Question_Wise_Breakdown")
    ws2['A1'] = "TOTAL_PAPER_MARKS:"
    ws2['B1'] = '=$Summary_Scores.$B$1'
    ws2['A1'].font = Font(bold=True, color="FF0000")
    
    q_headers = ["Student_ID", "Student_Name", "File_Name", "Question_ID", 
                 "Question_Type", "Weight_Pct", "Max_Marks", "Student_Answer_Preview",
                 "Estimated_Score_Pct", "Expected_Marks"]
    for col, h in enumerate(q_headers, 1):
        cell = ws2.cell(row=3, column=col, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="548235", end_color="548235", fill_type="solid")
    
    for r_idx, row_data in enumerate(rows, 4):
        ws2.cell(row=r_idx, column=1, value=row_data["Student_ID"])
        ws2.cell(row=r_idx, column=2, value=row_data["Student_Name"])
        ws2.cell(row=r_idx, column=3, value=row_data["File_Name"])
        ws2.cell(row=r_idx, column=4, value=row_data["Question_ID"])
        ws2.cell(row=r_idx, column=5, value=row_data["Question_Type"])
        ws2.cell(row=r_idx, column=6, value=row_data["Weight_Pct"])
        # Max marks formula: weight_pct * B1 / 100
        ws2.cell(row=r_idx, column=7).value = f'=F{r_idx}*$B$1/100'
        ws2.cell(row=r_idx, column=7).number_format = '0.00'
        ws2.cell(row=r_idx, column=8, value=row_data["Student_Extracted_Answer"][:300])
        ws2.cell(row=r_idx, column=9, value=row_data["Estimated_Score_Pct"])
        # Expected marks formula: estimated_pct/100 * max_marks
        ws2.cell(row=r_idx, column=10).value = f'=I{r_idx}/100*G{r_idx}'
        ws2.cell(row=r_idx, column=10).number_format = '0.00'
    
    # --- Tab 3: Dynamic_Rubric_Key ---
    ws3 = wb.create_sheet("Dynamic_Rubric_Key")
    ws3['A1'] = "TOTAL_PAPER_MARKS:"
    ws3['B1'] = '=$Summary_Scores.$B$1'
    ws3['A1'].font = Font(bold=True, color="FF0000")
    
    rub_headers = ["Question_ID", "Question_Type", "Weight_Pct", 
                   "Max_Marks_Scaled", "Question_Text", "Mandatory_Key_Concepts",
                   "Reference_Answer"]
    for col, h in enumerate(rub_headers, 1):
        cell = ws3.cell(row=3, column=col, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="BF8F00", end_color="BF8F00", fill_type="solid")
    
    for r_idx, q in enumerate(RUBRIC["questions"], 4):
        ws3.cell(row=r_idx, column=1, value=q["question_id"])
        ws3.cell(row=r_idx, column=2, value=q["question_type"])
        ws3.cell(row=r_idx, column=3, value=q["weight_pct"])
        ws3.cell(row=r_idx, column=4).value = f'=C{r_idx}*$B$1/100'
        ws3.cell(row=r_idx, column=4).number_format = '0.00'
        ws3.cell(row=r_idx, column=5, value=q["question_text"])
        ws3.cell(row=r_idx, column=6, value="\n".join(f"• {kc}" for kc in q["mandatory_key_concepts"]))
        ws3.cell(row=r_idx, column=6).alignment = Alignment(wrap_text=True)
        ws3.cell(row=r_idx, column=7, value=q["model_reference_answer"][:2000])
        ws3.cell(row=r_idx, column=7).alignment = Alignment(wrap_text=True)
    
    # Adjust column widths
    for ws in [ws1, ws2, ws3]:
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                if cell.value:
                    max_len = max(max_len, min(50, len(str(cell.value))))
            ws.column_dimensions[col_letter].width = min(55, max(12, max_len + 2))
    
    xlsx_path = os.path.join(WORK_DIR, "grading_benchmark_dataset.xlsx")
    wb.save(xlsx_path)
    print(f"  Saved Excel: {xlsx_path}")

# ============================================================
# STEP 5: Python helper script
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Creating scale_marks.py helper...")
print("=" * 60)

helper_script = '''#!/usr/bin/env python3
"""
scale_marks.py — Dynamic Marks Scaler for IntelliPlace Grading Dataset
----------------------------------------------------------------------
Usage:
    python scale_marks.py --scale 100   # Full 100-mark paper
    python scale_marks.py --scale 50    # Half (50-mark) paper
    python scale_marks.py --scale 75    # Custom scale

This reads grading_benchmark_dataset.csv, recalculates all marks
based on the target scale, and saves a new file.
"""
import argparse
import csv
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Scale marks in grading dataset")
    parser.add_argument("--scale", type=float, default=100, help="Target total paper marks (default: 100)")
    parser.add_argument("--input", default="grading_benchmark_dataset.csv", help="Input CSV file")
    parser.add_argument("--output", default=None, help="Output CSV file (default: auto-named)")
    parser.add_argument("--rubric", default="master_reference_rubric.json", help="Rubric JSON file")
    args = parser.parse_args()
    
    scale = args.scale
    if args.output is None:
        args.output = f"grading_dataset_scale_{int(scale)}.csv"
    
    # Load rubric for weight info
    with open(args.rubric, 'r', encoding='utf-8') as f:
        rubric = json.load(f)
    
    weight_map = {q["question_id"]: q["weight_pct"] for q in rubric["questions"]}
    
    # Read and rescale
    with open(args.input, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
    
    for row in rows:
        qid = row["Question_ID"]
        weight = float(weight_map.get(qid, row.get("Weight_Pct", 20)))
        max_marks = weight * scale / 100
        est_pct = float(row.get("Estimated_Score_Pct", 0))
        
        row["Max_Marks_Scaled"] = round(max_marks, 2)
        row["Expected_Score_Scaled"] = round(est_pct / 100 * max_marks, 2)
        row["Paper_Scale"] = int(scale)
    
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Scaled to {int(scale)}-mark paper")
    print(f"   Input:  {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Rows:   {len(rows)}")

if __name__ == "__main__":
    main()
'''

helper_path = os.path.join(WORK_DIR, "scale_marks.py")
with open(helper_path, 'w', encoding='utf-8') as f:
    f.write(helper_script)
print(f"  Saved: {helper_path}")

# ============================================================
# STEP 6: Create ZIP package
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Creating ZIP package...")
print("=" * 60)

zip_path = os.path.join(WORK_DIR, "complete_grading_dataset.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Rubric files
    zf.write(rubric_txt_path, "master_reference_rubric.txt")
    zf.write(rubric_json_path, "master_reference_rubric.json")
    
    # Dataset files
    zf.write(csv_path, "grading_benchmark_dataset.csv")
    if HAS_OPENPYXL:
        zf.write(xlsx_path, "grading_benchmark_dataset.xlsx")
    
    # Helper script
    zf.write(helper_path, "scale_marks.py")
    
    # Annotated student texts
    for fname in os.listdir(STUDENT_DIR):
        fpath = os.path.join(STUDENT_DIR, fname)
        zf.write(fpath, f"annotated_student_texts/{fname}")

print(f"  Saved ZIP: {zip_path}")

# Final summary
print("\n" + "=" * 60)
print("COMPLETE PACKAGE SUMMARY")
print("=" * 60)
print(f"  Students processed: {len(students)}")
print(f"  Questions in rubric: {len(RUBRIC['questions'])}")
print(f"  Total (Student × Question) rows: {len(rows)}")
print(f"  Files in ZIP:")
print(f"    - master_reference_rubric.txt")
print(f"    - master_reference_rubric.json")
print(f"    - grading_benchmark_dataset.csv")
if HAS_OPENPYXL:
    print(f"    - grading_benchmark_dataset.xlsx")
print(f"    - scale_marks.py")
print(f"    - annotated_student_texts/ ({len(students)} files)")
print(f"\n  ZIP location: {zip_path}")
