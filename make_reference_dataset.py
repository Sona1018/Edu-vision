import json
import os
import re

# Load benchmark csv created earlier
import csv

CSV_PATH = r"d:\IntelliPlace-AI-Answer-Grading-System-main\grading_package\grading_benchmark_dataset.csv"
RUBRIC_PATH = r"d:\IntelliPlace-AI-Answer-Grading-System-main\grading_package\master_reference_rubric.json"
OUT_PATH = r"d:\IntelliPlace-AI-Answer-Grading-System-main\dataset\marketing_reference_dataset.json"

with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
    rubric_data = json.load(f)

# Read answers from CSV
answers_by_q = {1: [], 2: [], 3: [], 4: [], 5: []}
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        qid_str = row["Question_ID"].replace("Q", "")
        if qid_str.isdigit():
            qn = int(qid_str)
            ans_text = row["Student_Extracted_Answer"].strip()
            # Clean OCR question headers if any
            ans_text = re.sub(r'^(?:Q\d|Ques\d|Question\s*\d)[^\n]*\n', '', ans_text, flags=re.IGNORECASE).strip()
            pct = float(row.get("Estimated_Score_Pct", 50.0))
            score_10 = round(pct / 10.0, 1) # out of 10 scale
            if len(ans_text) > 30:
                answers_by_q[qn].append({
                    "text": ans_text,
                    "score": score_10
                })

# Build exact IntelliPlace dataset schema
final_dataset = []

# Question Rubrics structured
q_rubrics = {
    1: [
        {
            "concept_id": "MM_Q01_C1",
            "concept_name": "Definition of Marketing",
            "reference_explanation": "Marketing is the social and managerial process of identifying customer needs, creating products or services, communicating their value, and delivering satisfaction to customers while achieving organizational goals.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q01_C2",
            "concept_name": "Customer-Oriented Nature",
            "reference_explanation": "Marketing starts and ends with customer satisfaction and understanding consumer demand rather than just selling.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q01_C3",
            "concept_name": "Continuous & Dynamic Process",
            "reference_explanation": "Marketing is an ongoing process that evolves with changing consumer preferences, competition, and technology.",
            "weight": 0.15
        },
        {
            "concept_id": "MM_Q01_C4",
            "concept_name": "Value Creation & Exchange-Based",
            "reference_explanation": "It creates superior value through marketing mix and operates on voluntary exchange where both buyer and seller benefit.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q01_C5",
            "concept_name": "Integrated Activity & Examples",
            "reference_explanation": "Coordinates 4Ps (Product, Price, Place, Promotion) with real-world examples like Apple, Zomato, Amazon, Netflix.",
            "weight": 0.25
        }
    ],
    2: [
        {
            "concept_id": "MM_Q02_C1",
            "concept_name": "Scope - Entities Marketed",
            "reference_explanation": "Marketing applies to physical goods, services, experiences, places (tourism), persons (celebrities), ideas and causes.",
            "weight": 0.25
        },
        {
            "concept_id": "MM_Q02_C2",
            "concept_name": "Scope - Functional Activities",
            "reference_explanation": "Includes market research, product planning, branding, pricing, promotion, distribution channels, and digital marketing.",
            "weight": 0.25
        },
        {
            "concept_id": "MM_Q02_C3",
            "concept_name": "Importance to Business",
            "reference_explanation": "Drives sales growth, generates revenue, builds brand image, customer loyalty, and creates competitive advantage.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q02_C4",
            "concept_name": "Importance to Consumers",
            "reference_explanation": "Satisfies needs and wants, provides wider choices, better quality, convenience, and improves living standards.",
            "weight": 0.15
        },
        {
            "concept_id": "MM_Q02_C5",
            "concept_name": "Importance to Society",
            "reference_explanation": "Generates employment opportunities, encourages innovation, and supports national economic development.",
            "weight": 0.15
        }
    ],
    3: [
        {
            "concept_id": "MM_Q03_C1",
            "concept_name": "Concept of Needs",
            "reference_explanation": "Needs are basic human requirements essential for survival such as food, water, clothing, shelter, and communication.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q03_C2",
            "concept_name": "Concept of Wants",
            "reference_explanation": "Wants are desires for specific goods or services that satisfy needs, shaped by culture, personality, and society.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q03_C3",
            "concept_name": "Differences Between Needs and Wants",
            "reference_explanation": "Needs are biological, universal, limited, and essential for survival; wants are psychological, unlimited, cultural, and optional.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q03_C4",
            "concept_name": "Marketing Does Not Create Needs",
            "reference_explanation": "Fundamental human needs already exist; marketing identifies and provides solutions to fulfill them.",
            "weight": 0.15
        },
        {
            "concept_id": "MM_Q03_C5",
            "concept_name": "How Marketing Shapes Wants",
            "reference_explanation": "Marketing transforms needs into wants through advertising, branding, emotional appeal, packaging, and lifestyle positioning.",
            "weight": 0.25
        }
    ],
    4: [
        {
            "concept_id": "MM_Q04_C1",
            "concept_name": "Definition of Marketing Environment",
            "reference_explanation": "The sum total of internal and external forces and factors that influence a firm's ability to develop customer relationships.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q04_C2",
            "concept_name": "Characteristics of Marketing Environment",
            "reference_explanation": "Dynamic in nature, complex, interrelated, uncertain, uncontrollable external factors, and requiring continuous monitoring.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q04_C3",
            "concept_name": "Internal Environment",
            "reference_explanation": "Controllable factors within organization: top management, organizational structure, culture, finance, R&D, employees.",
            "weight": 0.30
        },
        {
            "concept_id": "MM_Q04_C4",
            "concept_name": "External Environment Overview",
            "reference_explanation": "Uncontrollable forces outside organization divided into Micro environment (task) and Macro environment (societal).",
            "weight": 0.30
        }
    ],
    5: [
        {
            "concept_id": "MM_Q05_C1",
            "concept_name": "Micro Environment Concept & Customers",
            "reference_explanation": "Forces close to company directly affecting operations: customers are the central focus (consumer, business, government markets).",
            "weight": 0.25
        },
        {
            "concept_id": "MM_Q05_C2",
            "concept_name": "Micro Factors - Suppliers, Competitors, Intermediaries, Publics",
            "reference_explanation": "Suppliers provide inputs; competitors offer substitutes; intermediaries help distribute; publics include media and community.",
            "weight": 0.25
        },
        {
            "concept_id": "MM_Q05_C3",
            "concept_name": "Macro Environment Concept & PESTLE",
            "reference_explanation": "Broad uncontrollable societal forces analyzed through PESTLE framework affecting the entire industry.",
            "weight": 0.20
        },
        {
            "concept_id": "MM_Q05_C4",
            "concept_name": "Macro Factors - Demographic, Economic, Socio-Cultural",
            "reference_explanation": "Demographics (age, population), economic (inflation, income), and socio-cultural factors (lifestyle, customs, health awareness).",
            "weight": 0.15
        },
        {
            "concept_id": "MM_Q05_C5",
            "concept_name": "Macro Factors - Technological, Political, Legal, Natural",
            "reference_explanation": "Technology (AI, digital payments), political/legal (tax, laws), and natural environmental factors (sustainability, green trends).",
            "weight": 0.15
        }
    ]
}

questions_meta = [
    {
        "q_num": 1,
        "question_id": "MM_Q01",
        "question": "Define Marketing. Explain the nature of marketing along with examples."
    },
    {
        "q_num": 2,
        "question_id": "MM_Q02",
        "question": "What all are the scope of marketing and importance of marketing?"
    },
    {
        "q_num": 3,
        "question_id": "MM_Q03",
        "question": "Differentiate between Needs and Wants. Explain how marketing influences both."
    },
    {
        "q_num": 4,
        "question_id": "MM_Q04",
        "question": "Explain Marketing Environment. Define Internal and External Marketing Environment."
    },
    {
        "q_num": 5,
        "question_id": "MM_Q05",
        "question": "Explain Micro and Macro External Environmental Factors."
    }
]

for q_meta in questions_meta:
    qn = q_meta["q_num"]
    item = {
        "question_id": q_meta["question_id"],
        "subject": "Marketing Management",
        "question": q_meta["question"],
        "rubric": q_rubrics[qn],
        "answers": answers_by_q[qn]
    }
    final_dataset.append(item)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)

print(f"SUCCESS: Saved reference test dataset to {OUT_PATH}")
print(f"Total Questions: {len(final_dataset)}")
for q in final_dataset:
    print(f"  {q['question_id']}: {len(q['rubric'])} concepts, {len(q['answers'])} student test answers")
