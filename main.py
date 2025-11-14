import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from graph import ConversationState, build_graph

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")

def main():
    app = build_graph()

    state: ConversationState = {"messages": []}

    print("Chat (q to quit):")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"q", "quit", "exit"}:
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = app.invoke(state)

        last_msg = state["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        print(f"AI: {last_msg.content}\n")


if __name__ == "__main__":
    main()