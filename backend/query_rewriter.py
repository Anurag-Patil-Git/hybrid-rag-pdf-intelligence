
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def rewrite_query(
    question: str,
    chat_history: list,
    llm
) -> str:
    """
    Convert a follow-up question into a standalone question
    using previous conversation messages.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not chat_history:
        return question.strip()

    # Keep the latest 6 messages to limit prompt size.
    recent_history = chat_history[-6:]

    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_history
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a search query rewriting assistant.

Your job is to rewrite the user's latest question into a
standalone question using the conversation history.

Rules:
1. Resolve references such as it, they, their, this and that.
2. Preserve the original meaning and intent.
3. Do not answer the question.
4. Do not introduce facts that are not in the conversation.
5. If the question is already standalone, return it unchanged.
6. Return only the rewritten question.
            """
        ),
        (
            "human",
            """
Conversation history:
{chat_history}

Latest user question:
{question}

Standalone search question:
            """
        )
    ])

    chain = prompt | llm | StrOutputParser()

    rewritten_question = chain.invoke({
        "chat_history": history_text,
        "question": question
    })

    return rewritten_question.strip().strip('"')

