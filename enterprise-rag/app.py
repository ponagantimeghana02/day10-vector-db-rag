# app.py

from ingestion import ingest_document

from vector_store import (
    store_chunks,
    document_count
)

from access_control import (
    login
)

from chat_engine import (
    ask_question,
    show_chat_history
)

from cache_manager import (
    get_cache_stats,
    clear_cache
)


# ==================================
# Upload Document
# ==================================

def upload_document():

    source = input(
        "\nEnter file path or URL: "
    )

    try:

        chunks = ingest_document(
            source
        )

        store_chunks(
            chunks
        )

        print(
            "\nDocument indexed successfully."
        )

    except Exception as e:

        print(
            "\nError:",
            e
        )


# ==================================
# Ask Question
# ==================================

def ask_user_question(
    username,
    role
):

    question = input(
        "\nAsk a question: "
    )

    result = ask_question(

        username=username,

        role=role,

        question=question

    )

    print("\nAnswer")

    print("-" * 80)

    print(
        result["answer"]
    )

    print("\nSources")

    print("-" * 80)

    unique_sources = list(
        set(
            result["sources"]
        )
    )

    for source in unique_sources:

        print(
            f"- {source}"
        )

    print("\nCached:")

    print(
        result["cached"]
    )


# ==================================
# Show Statistics
# ==================================

def show_statistics():

    print("\nSystem Statistics")

    print("-" * 80)

    print(
        "Indexed Chunks:",
        document_count()
    )

    print()

    stats = get_cache_stats()

    for key, value in stats.items():

        print(
            f"{key}: {value}"
        )


# ==================================
# Main Menu
# ==================================

def menu():

    print("\nEnterprise RAG System")

    print("-" * 80)

    print("1. Upload Document")

    print("2. Ask Question")

    print("3. View Chat History")

    print("4. System Statistics")

    print("5. Clear Cache")

    print("6. Exit")


# ==================================
# Main Application
# ==================================

def main():

    print(
        "\nEnterprise Knowledge Assistant"
    )

    print("=" * 80)

    user = login()

    if not user:

        return

    username = input(
        "\nEnter username again for session: "
    )

    role = user["role"]

    while True:

        menu()

        choice = input(
            "\nChoose option: "
        )

        # ----------------------
        # Upload
        # ----------------------

        if choice == "1":

            upload_document()

        # ----------------------
        # Ask
        # ----------------------

        elif choice == "2":

            ask_user_question(
                username,
                role
            )

        # ----------------------
        # History
        # ----------------------

        elif choice == "3":

            show_chat_history()

        # ----------------------
        # Statistics
        # ----------------------

        elif choice == "4":

            show_statistics()

        # ----------------------
        # Clear Cache
        # ----------------------

        elif choice == "5":

            clear_cache()

        # ----------------------
        # Exit
        # ----------------------

        elif choice == "6":

            print(
                "\nGoodbye."
            )

            break

        else:

            print(
                "\nInvalid option."
            )


# ==================================
# Run
# ==================================

if __name__ == "__main__":

    main()