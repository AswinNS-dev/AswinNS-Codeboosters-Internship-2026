import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = """You are an expert Healthcare AI Assistant. You ONLY answer questions related to:
- Medical conditions and diseases
- Symptoms and diagnosis
- Medications and treatments
- Hospital admissions and procedures
- Health insurance and billing
- Diagnostic tests and results
- Patient care and wellness

If a question is outside the healthcare domain, politely decline and redirect.

Use the following retrieved context from the knowledge base when available.
Prioritize the context over general knowledge, but supplement with your expertise if needed.

Retrieved Context:
{context}

Always be accurate, empathetic, and clear. Do not give personal medical advice — recommend consulting a doctor for specific cases.
"""

CATEGORY_PROMPT = """Classify this healthcare question into one of these categories and subcategories.
Return ONLY a JSON object like: {{"category": "...", "subcategory": "..."}}

Categories and subcategories:
- Conditions: Cancer, Diabetes, Obesity, Hypertension, Asthma, Other
- Medications: Paracetamol, Ibuprofen, Aspirin, Antibiotics, Other
- Admissions: Emergency, Urgent, Elective, Discharge
- Diagnostics: Blood Tests, Imaging, Test Results, Lab Work
- Insurance: Coverage, Billing, Claims, Providers
- Symptoms: Pain, Fever, Fatigue, Respiratory, Other
- General: Wellness, Prevention, Nutrition, Other

Question: {question}

Return only valid JSON, no extra text."""

class HealthcareAgent:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.environ["GROQ_API_KEY"],
        )
        self.parser = StrOutputParser()

        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        self.category_prompt = ChatPromptTemplate.from_messages([
            ("human", CATEGORY_PROMPT),
        ])

        self.chain = self.chat_prompt | self.llm | self.parser
        self.category_chain = self.category_prompt | self.llm | self.parser

    def _classify(self, question):
        try:
            import json
            raw = self.category_chain.invoke({"question": question})
            raw = raw.strip().replace("```json", "").replace("```", "")
            result = json.loads(raw)
            return result.get("category", "General"), result.get("subcategory", "Other")
        except Exception:
            return "General", "Other"

    def answer(self, question, history=None):
        context = self.rag.retrieve(question)
        response = self.chain.invoke({"question": question, "context": context})
        category, subcategory = self._classify(question)
        return response, category, subcategory
