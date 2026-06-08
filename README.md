# 🏥 Healthcare AI Agent — RAG System

Domain-specific AI assistant for Healthcare. Every conversation is stored in Snowflake, embedded with `all-MiniLM-L6-v2`, indexed in ChromaDB, and retrieved via RAG for increasingly accurate answers.

---

## System Architecture

```
User Query
    │
    ▼
HealthcareAgent (Groq llama-3.3-70b)
    │
    ├── RAGPipeline.retrieve()
    │       ├── Embed query → all-MiniLM-L6-v2
    │       └── ChromaDB semantic search → top-5 context chunks
    │
    ├── LangChain chain (prompt | llm | parser)
    │
    └── Response → SnowflakeStore.save_interaction()
                        │
                        └── HEALTHCARE_INTERACTIONS table

[Periodically / On-demand]
KnowledgeBase.build()
    ├── Fetch all rows from Snowflake
    ├── Clean + chunk text
    ├── Embed with all-MiniLM-L6-v2
    └── Upsert into ChromaDB
```

---

## Setup

```bash
pip install -r requirements.txt
python app.py
```

You will be prompted for:
- `GROQ_API_KEY`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- `SNOWFLAKE_WAREHOUSE`

---

## Snowflake Table Schema

```sql
CREATE TABLE HEALTHCARE_INTERACTIONS (
    QUERY_ID            VARCHAR(64)   PRIMARY KEY,
    USER_QUESTION       TEXT,
    AI_RESPONSE         TEXT,
    DOMAIN_CATEGORY     VARCHAR(100),
    DOMAIN_SUBCATEGORY  VARCHAR(100),
    TIMESTAMP           TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
    SESSION_ID          VARCHAR(64)
);
```

---

## Domain Categories

| Category    | Subcategories                                      |
|-------------|---------------------------------------------------|
| Conditions  | Cancer, Diabetes, Obesity, Hypertension, Asthma   |
| Medications | Paracetamol, Ibuprofen, Aspirin, Antibiotics      |
| Admissions  | Emergency, Urgent, Elective, Discharge             |
| Diagnostics | Blood Tests, Imaging, Test Results, Lab Work       |
| Insurance   | Coverage, Billing, Claims, Providers               |
| Symptoms    | Pain, Fever, Fatigue, Respiratory                  |
| General     | Wellness, Prevention, Nutrition                    |

---

## Files

| File                 | Purpose                                              |
|----------------------|------------------------------------------------------|
| `app.py`             | Gradio UI — main entry point                         |
| `agent.py`           | HealthcareAgent — Groq LLM + category classifier     |
| `snowflake_store.py` | Save/fetch interactions from Snowflake               |
| `knowledge_base.py`  | Build ChromaDB from Snowflake data                   |
| `rag_pipeline.py`    | Embed query → retrieve top-K context from ChromaDB   |
| `requirements.txt`   | Dependencies                                         |

---

## How to Use

1. **Ask questions** in the chat → agent answers + saves to Snowflake
2. **Rebuild KB** button → pulls all Snowflake rows, embeds, stores in ChromaDB
3. **Future queries** now use RAG — answers grounded in accumulated knowledge
4. Knowledge base **grows** with every session

---

## Dataset Used

`healthcare_dataset.csv` — 55,000+ patient records with:
- Medical Conditions (Cancer, Diabetes, Obesity, Hypertension, Asthma)
- Medications (Paracetamol, Ibuprofen, Aspirin)
- Admission Types (Emergency, Urgent, Elective)
- Test Results (Normal, Abnormal, Inconclusive)
- Insurance Providers, Billing Amounts, Doctors, Hospitals
