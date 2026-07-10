import os
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


load_dotenv()


st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="✍️",
    layout="centered",
)


# ---------- Secret Handling ----------
try:
    for key in ["GROQ_API_KEY", "MISTRAL_API_KEY", "TAVILY_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #2563eb);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.22);
    }

    .hero h1 {
        font-size: 2.4rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-bottom: 0;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 16px;
        background: white;
        color: #111827;
        border: 1px solid #e5e7eb;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
        line-height: 1.7;
        font-size: 1.02rem;
        font-weight: 500;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #0f172a);
        color: white;
        border: none;
    }

    div.stButton > button:hover {
        color: white;
        border: none;
        filter: brightness(1.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- State ----------
class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# ---------- Prompts ----------
WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim with short paragraphs, around "
    "150-200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim using short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>"
)


# ---------- LangGraph App ----------
@st.cache_resource
def build_app():
    search_tool = TavilySearch(max_results=3)
    tools = [search_tool]

    writer_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
    )

    reviewer_llm = ChatMistralAI(
        model="mistral-medium-3-5",
        temperature=0.2,
    )

    writer_with_web = writer_llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def writer_node(state: State) -> dict:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        is_returning_from_tool = getattr(last_message, "type", None) == "tool"

        if is_returning_from_tool:
            attempt = state.get("attempt", 1)

            response = writer_with_web.invoke(
                [("system", WRITER_SYSTEM_PROMPT)] + messages
            )

            return {
                "messages": [response],
                "attempt": attempt,
            }

        attempt = state.get("attempt", 0) + 1

        if attempt > 5:
            return {
                "attempt": 5,
            }

        topic = state["topic"]
        previous_feedback = state.get("review_feedback", "")

        if attempt == 1:
            user_message = (
                f"Write a LinkedIn post on this topic: {topic}. "
                "If current information is useful, search the web first."
            )
        else:
            user_message = (
                f"The previous draft on '{topic}' was rejected.\n\n"
                f"Reviewer feedback:\n{previous_feedback}\n\n"
                "Write a stronger version that fixes every issue."
            )

        response = writer_with_web.invoke(
            [
                ("system", WRITER_SYSTEM_PROMPT),
                ("human", user_message),
            ]
        )

        return {
            "messages": [("human", user_message), response],
            "attempt": attempt,
        }

    def extract_draft_node(state: State) -> dict:
        last_message = state["messages"][-1]
        return {"draft": last_message.content}

    def reviewer_node(state: State) -> dict:
        draft = state["draft"]

        response = reviewer_llm.invoke(
            [
                ("system", REVIEWER_SYSTEM_PROMPT),
                ("human", f"Review this LinkedIn post draft:\n\n{draft}"),
            ]
        )

        review_text = response.content.strip()
        verdict_area = review_text.upper().split("FEEDBACK")[0]
        is_approved = "APPROVED" in verdict_area

        if "FEEDBACK:" in review_text:
            feedback = review_text.split("FEEDBACK:", 1)[1].strip()
        else:
            feedback = review_text

        return {
            "review_feedback": feedback,
            "is_approved": is_approved,
        }

    def should_use_tool(state: State):
        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"

        return "extract_draft"

    def should_stop_looping(state: State):
        if state["is_approved"]:
            return END

        if state["attempt"] >= 5:
            return END

        return "writer"

    graph = StateGraph(State)

    graph.add_node("writer", writer_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "writer")
    graph.add_conditional_edges("writer", should_use_tool)
    graph.add_edge("tools", "writer")
    graph.add_edge("extract_draft", "reviewer")
    graph.add_conditional_edges("reviewer", should_stop_looping)

    return graph.compile()


# ---------- UI ----------
st.markdown(
    """
    <div class="hero">
        <h1>LinkedIn Post Generator</h1>
        <p>Create, review, and improve a professional LinkedIn post automatically.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

topic = st.text_area(
    "What topic do you want a LinkedIn post about?",
    placeholder="Example: How AI agents are changing customer support teams",
    height=120,
)

col1, col2 = st.columns([2, 1])

with col1:
    generate = st.button("Generate LinkedIn Post")

with col2:
    st.markdown(
        "<div class='small-muted'>The app can revise up to 5 times.</div>",
        unsafe_allow_html=True,
    )

if generate:
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    missing_keys = [
        key
        for key in ["GROQ_API_KEY", "MISTRAL_API_KEY", "TAVILY_API_KEY"]
        if not os.environ.get(key)
    ]

    if missing_keys:
        st.error(
            "Missing required API keys. Add them to your .env file or Streamlit secrets."
        )
        st.stop()

    initial_state = {
        "topic": topic.strip(),
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    app = build_app()

    with st.status("Creating your LinkedIn post...", expanded=True) as status:
        st.write("Writing and reviewing your post...")
        final_state = app.invoke(initial_state)

        if final_state["is_approved"]:
            status.update(
                label="Post approved and ready.",
                state="complete",
                expanded=False,
            )
        else:
            status.update(
                label="Finished with the best available draft.",
                state="complete",
                expanded=False,
            )

    st.markdown("### Final LinkedIn Post")

    safe_draft = final_state["draft"].replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="result-box">
            {safe_draft}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Revision Attempts", min(final_state["attempt"], 5))

    with metric2:
        st.metric(
            "Review Result",
            "Approved" if final_state["is_approved"] else "Needs Review",
        )

    with st.expander("Reviewer Feedback"):
        st.write(final_state["review_feedback"])

    st.download_button(
        label="Download Post",
        data=final_state["draft"],
        file_name="linkedin_post.txt",
        mime="text/plain",
    )

