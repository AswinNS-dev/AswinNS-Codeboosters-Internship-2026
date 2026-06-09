import os
import uuid
import gradio as gr
from agent import HealthcareAgent
from snowflake_store import SnowflakeStore
from knowledge_base import KnowledgeBase
from rag_pipeline import RAGPipeline

# ── credentials (only what's actually needed) ─────────────────────────────────
os.environ["GROQ_API_KEY"]       = input("Enter GROQ API KEY: ")
os.environ["SNOWFLAKE_USER"]     = input("Enter Snowflake Username: ")
os.environ["SNOWFLAKE_PASSWORD"] = input("Enter Snowflake Password: ")
os.environ["SNOWFLAKE_ACCOUNT"]  = input("Enter Snowflake Account Identifier: ")

# ── init components ───────────────────────────────────────────────────────────
store = SnowflakeStore()
store.init_table()          # creates DB + schema + table if they don't exist

kb    = KnowledgeBase(store)
rag   = RAGPipeline(kb)
agent = HealthcareAgent(rag)

SESSION_ID = str(uuid.uuid4())

# ── handlers ──────────────────────────────────────────────────────────────────
def chat(user_message, history):
    if not user_message.strip():
        return history, ""

    response, category, subcategory = agent.answer(user_message, history)

    store.save_interaction(
        query_id=str(uuid.uuid4()),
        user_question=user_message,
        ai_response=response,
        domain_category=category,
        domain_subcategory=subcategory,
        session_id=SESSION_ID,
    )

    history = history or []
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return history, ""

def rebuild_kb():
    count = kb.build()
    return f"✅ Knowledge base rebuilt — {count} chunks indexed into ChromaDB."

def get_stats():
    stats = store.get_stats()
    lines = "\n".join(f"  • {k}: {v}" for k, v in stats["by_category"].items())
    return f"📊 **Total interactions:** {stats['total']}\n\n🗂️ **By category:**\n{lines}"

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Healthcare AI Agent") as demo:

    gr.HTML("""
    <div id="header">
      <h1>🏥 Healthcare AI Agent</h1>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Healthcare Assistant",
                height=480,
                layout="bubble",
                avatar_images=(None, "https://img.icons8.com/emoji/96/stethoscope-emoji.png"),
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask anything about healthcare, symptoms, medications, conditions…",
                    show_label=False,
                    scale=5,
                    lines=1,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "What are the common symptoms of Type 2 Diabetes?",
                    "Explain the difference between urgent and emergency admission.",
                    "What does an abnormal test result typically indicate?",
                    "Which medications are commonly used for cancer pain management?",
                    "What insurance providers typically cover obesity treatment?",
                ],
                inputs=msg_box,
                label="💡 Try these questions",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Knowledge Base")
            rebuild_btn = gr.Button("🔄 Rebuild KB from Snowflake", variant="secondary")
            rebuild_out = gr.Markdown("")

            gr.Markdown("### 📈 Session Stats")
            stats_btn = gr.Button("📊 Get Stats", variant="secondary")
            stats_out = gr.Markdown("")

        

    send_btn.click(chat, [msg_box, chatbot], [chatbot, msg_box])
    msg_box.submit(chat, [msg_box, chatbot], [chatbot, msg_box])
    rebuild_btn.click(rebuild_kb, outputs=rebuild_out)
    stats_btn.click(get_stats, outputs=stats_out)

demo.launch(
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("DM Sans"),
    ),
)