from langchain_openai import ChatOpenAI


def build_llm():
    return ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.7,
    )