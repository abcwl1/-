# === Streamlit + LCEL + RAG + 上下文记忆 ===

# === 1️⃣ 导入必要库 ===
from langchain_community.embeddings import HuggingFaceEmbedding
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

from medicalrag import PubMedFetcher, MedicalRAGBuilder, MedicalQASystem

# ---------------------------
# 1️⃣ 初始化向量库和 QA 系统
# ---------------------------

# 向量库路径
VECTORSTORE_PATH = ""

# 初始化 RAG
rag = MedicalRAGBuilder()
if os.path.exists(VECTORSTORE_PATH):
    rag.load_vectorstore(persist_directory=VECTORSTORE_PATH)

# 初始化 QA 系统
qa_system = MedicalQASystem(rag.vectorstore)

# ---------------------------
# 2️⃣ Streamlit 前端
# ---------------------------

st.markdown("#### 🦜🔗 医学文献问答 RAG 系统（带对话记忆）")

# 对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 用户输入
if prompt := st.chat_input("请输入问题"):
    st.session_state.messages.append(("Human", prompt))

    # 调用 QA 系统
    answer_data = qa_system.ask(prompt)

    # 显示用户输入
    with st.chat_message("human"):
        st.write(prompt)

    # 显示 AI 回答
    with st.chat_message("ai"):
        st.write(answer_data['answer'])

    # 显示参考文献
    st.markdown("**参考文献:**")
    for i, meta in enumerate(answer_data['sources'], 1):
        title = meta.get("title", "Unknown title")
        pmid = meta.get("pmid", "Unknown PMID")
        source = meta.get("source", "Unknown source")
        st.markdown(f"{i}. {title} (PMID: {pmid})\n来源: {source}")

    # 保存 AI 回答到聊天历史
    st.session_state.messages.append(("AI", answer_data['answer']))

# 可选：显示完整聊天历史
if st.checkbox("显示聊天历史"):
    st.markdown("### 聊天历史")
    for role, msg in st.session_state.messages:
        st.markdown(f"**{role}:** {msg}")


    

