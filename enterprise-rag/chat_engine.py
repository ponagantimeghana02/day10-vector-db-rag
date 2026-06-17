# chat_engine.py

import os
import json
from datetime import datetime

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    CHAT_HISTORY_FILE
)

from retriever import (
    retrieve_context,
    build_context
)

from cache_manager import (
    get_cached_response,
    cache_response
)

from access_control import (
    get_metadata_filter
)

# ==================================
# Load Environment
# ==================================

load_dotenv()

# ==================================
# LLM
# ==================================

llm = ChatGroq(
    model=LLM_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0
)

# ==================================
# Prompt
# ==================================

prompt = ChatPromptTemplate.from_template(
"""
You are an Enterprise Knowledge Assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:

"I don't have enough information in the available documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = (
    prompt
    | llm
    | StrOutputParser()
)

# ==================================
# Ensure History Folder Exists
# ==================================

os.makedirs(
    os.path.dirname(
        CHAT_HISTORY_FILE
    ),
    exist_ok=True
)

# ==================================
# Load History
# ==================================

def load_history():

    if not os.path.exists(
        CHAT_HISTORY_FILE
    ):

        return []

    try:

        with open(
            CHAT_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except:

        return []

# ==================================
# Save History
# ==================================

def save_history(history):

    with open(
        CHAT_HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

# ==================================
# Add Chat Entry
# ==================================

def add_chat_entry(
    username,
    question,
    answer,
    sources
):

    history = load_history()

    history.append({

        "timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "user":
        username,

        "question":
        question,

        "answer":
        answer,

        "sources":
        sources
    })

    save_history(history)

# ==================================
# Ask Question
# ==================================

def ask_question(
    username,
    role,
    question
):

    # ------------------------
    # Cache Check
    # ------------------------

    cached = get_cached_response(
        question
    )

    if cached:

        return {

            "answer":
            cached["answer"],

            "sources":
            cached["sources"],

            "cached":
            True
        }

    # ------------------------
    # Access Filter
    # ------------------------

    metadata_filter = (
        get_metadata_filter(
            role
        )
    )

    # ------------------------
    # Retrieve Documents
    # ------------------------

    retrieved_docs = (
        retrieve_context(
            query=question,
            metadata_filter=
            metadata_filter
        )
    )

    if not retrieved_docs:

        return {

            "answer":
            "No relevant documents found.",

            "sources":
            [],

            "cached":
            False
        }

    # ------------------------
    # Build Context
    # ------------------------

    context, sources = (
        build_context(
            retrieved_docs
        )
    )

    # ------------------------
    # Generate Answer
    # ------------------------

    answer = chain.invoke({

        "context":
        context,

        "question":
        question
    })

    # ------------------------
    # Save Cache
    # ------------------------

    cache_response(

        question,

        answer,

        sources

    )

    # ------------------------
    # Save History
    # ------------------------

    add_chat_entry(

        username,

        question,

        answer,

        sources

    )

    return {

        "answer":
        answer,

        "sources":
        sources,

        "cached":
        False
    }

# ==================================
# View Chat History
# ==================================

def show_chat_history():

    history = load_history()

    if not history:

        print(
            "\nNo history found."
        )

        return

    print("\nChat History")

    print("-" * 80)

    for item in history:

        print(
            f"\n[{item['timestamp']}]"
        )

        print(
            f"User: {item['user']}"
        )

        print(
            f"Question: {item['question']}"
        )

        print(
            f"Answer: {item['answer']}"
        )

        print(
            f"Sources: {item['sources']}"
        )

        print("-" * 80)

# ==================================
# Test
# ==================================

if __name__ == "__main__":

    result = ask_question(

        username="admin",

        role="admin",

        question="What is AI?"

    )

    print(result)