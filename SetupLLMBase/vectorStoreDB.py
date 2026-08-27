# SetupLLMBase/vectorStoreDB.py

import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_core.documents import Document

from SetupLLMBase.analysisRecord import format_analysis_record


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
RAG_PATH = Path("SetupLLMBase/rag_store")
RAG_PATH.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# LAZY EMBEDDING LOADER
# -------------------------------------------------
def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. "
            "Set it in your environment or .env file."
        )

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )


# -------------------------------------------------
# SAFE LOAD OR CREATE STORE
# -------------------------------------------------
def load_rag_store():
    embeddings = get_embeddings()

    if (RAG_PATH / "index.faiss").exists():
        try:
            return FAISS.load_local(
                str(RAG_PATH),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print("⚠️ FAISS load failed, rebuilding:", e)

    # Proper empty index initialization
    dummy_doc = Document(page_content="init", metadata={})
    store = FAISS.from_documents([dummy_doc], embeddings)
    store.delete([store.index_to_docstore_id[0]])
    store.save_local(str(RAG_PATH))
    return store


# -------------------------------------------------
# SAVE ANALYSIS
# -------------------------------------------------
def save_analysis_to_rag(
    ticker: str,
    analysis_type: str,
    analysis_text: str,
    confidence: str = "medium",
):
    if not analysis_text:
        return

    store = load_rag_store()

    record = format_analysis_record(
        ticker=ticker,
        analysis_type=analysis_type,
        content=analysis_text,
        confidence=confidence,
    )

    doc = Document(
        page_content=record,
        metadata={
            "ticker": ticker,
            "analysis_type": analysis_type,
            "confidence": confidence,
        },
    )

    store.add_documents([doc])
    store.save_local(str(RAG_PATH))


# -------------------------------------------------
# RETRIEVE PAST ANALYSIS
# -------------------------------------------------
def get_past_analysis(ticker: str, k: int = 5):
    store = load_rag_store()

    retriever = store.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"ticker": ticker},
        }
    )

    return retriever.invoke(ticker)
