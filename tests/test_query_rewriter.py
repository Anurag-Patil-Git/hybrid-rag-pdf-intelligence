
from backend.llm import get_llm
from backend.query_rewriter import rewrite_query


llm = get_llm()

# Simulated conversation
chat_history = [
    {
        "role": "user",
        "content": "What are LangChain indexes?"
    },
    {
        "role": "assistant",
        "content": (
            "LangChain indexes connect LLM applications "
            "with external knowledge sources."
        )
    }
]

# Follow-up question
question = "What are their advantages?"

rewritten_question = rewrite_query(
    question=question,
    chat_history=chat_history,
    llm=llm
)

print("\nOriginal question:")
print(question)

print("\nRewritten standalone question:")
print(rewritten_question)
