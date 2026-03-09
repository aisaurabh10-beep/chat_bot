import uuid
from typing import List

import streamlit as st

from main_user import ChatMessage, Source, chatbot, chat_logger


st.set_page_config(page_title="BharathaTechno Chatbot", page_icon="💬", layout="wide")


@st.cache_resource
def get_chatbot():
    """Load heavy resources once per Streamlit server process."""
    if not chatbot.model:
        chatbot.load_resources()
    return chatbot


def to_chat_history(messages: List[dict]) -> List[ChatMessage]:
    history: List[ChatMessage] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content:
            history.append(ChatMessage(role=role, content=content))
    return history


def extract_sources(context: List[dict]) -> List[Source]:
    return [
        Source(
            title=chunk.get("metadata", {}).get("title", "N/A"),
            url=chunk.get("metadata", {}).get("url", "N/A"),
        )
        for chunk in context[:3]
    ]


def main():
    st.title("BharathaTechno Chatbot")
    st.caption("Streamlit test app using logic from main_user.py")

    with st.sidebar:
        st.subheader("Session")
        if "user_id" not in st.session_state:
            st.session_state.user_id = f"user-{uuid.uuid4().hex[:8]}"
        st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
        st.info("To access from same network, run Streamlit on 0.0.0.0.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi, I am Anna. Ask me anything about BharathaTechno.",
            }
        ]

    try:
        bot = get_chatbot()
    except Exception as exc:
        st.error(f"Failed to load chatbot resources: {exc}")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Type your message...")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Exclude current user message from history to avoid duplication.
                history_models = to_chat_history(st.session_state.messages[:-1])
                context = bot.retrieve_context(user_prompt)
                answer = bot.generate_response(user_prompt, context, history_models)
                sources = extract_sources(context)
                chat_logger.log_interaction(
                    st.session_state.user_id, user_prompt, answer, sources
                )
            except Exception as exc:
                answer = (
                    f"Sorry, I hit an error while generating a response.\n\nDetails: `{exc}`"
                )
                sources = []

        st.markdown(answer)

        valid_sources = [s for s in sources if s.url and s.url != "N/A"]
        if valid_sources:
            with st.expander("Sources"):
                for src in valid_sources:
                    st.markdown(f"- [{src.title}]({src.url})")

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
