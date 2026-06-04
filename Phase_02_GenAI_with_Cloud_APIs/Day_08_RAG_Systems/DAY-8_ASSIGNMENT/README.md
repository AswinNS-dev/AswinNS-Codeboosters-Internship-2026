# 🚀 Social Media Content Generator

A clean, minimal, and fully vertical AI-powered social media generator built with **LangChain**, **Groq (Llama 3.3)**, and **Gradio**.

## ✨ Features

- **Vertical Top-Down UI**: A streamlined, simple interface with zero animations for a distraction-free experience.
- **Multi-Platform Support**: Optimized prompts for LinkedIn, Instagram, X (Twitter), Facebook, Threads, and YouTube.
- **Smart Logic**:
    - Accurate Word Count enforcement (not character count).
    - Customizable Hashtag inclusion and count.
    - Emoji toggle (Include/Exclude).
    - Engagement Strategy & Scoring for each post.
- **Privacy Focus**: Groq API Key is provided via the UI and kept in session (not hardcoded).

## 🛠️ Tech Stack

- **Python 3.13+**
- **Gradio**: UI Framework.
- **LangChain**: LLM Orchestration & Prompt Templatization.
- **Groq API**: Powered by the `llama-3.3-70b-versatile` model.

## 🚀 How to Run

1.  **Clone the project** or download the `.ipynb` file.
2.  **Install dependencies**:
    ```bash
    pip install langchain langchain-groq langchain-core gradio
    ```
3.  **Open the Notebook**: Run all cells in `AI_Social_Media_Content_Generator_Gradio.ipynb`.
4.  **Launch the App**: The Gradio interface will open at a local URL (e.g., `http://127.0.0.1:7860`).

## 📝 Usage

1.  Enter your **Groq API Key**.
2.  Type your **Topic** (e.g., "The importance of mindfulness").
3.  Select your **Platform**.
4.  Set your **Target Word Count**.
5.  Toggle **Emojis/Hashtags** and hit **Generate**.

---
*Created as part of the DAY-8 ASSIGNMENT.*
