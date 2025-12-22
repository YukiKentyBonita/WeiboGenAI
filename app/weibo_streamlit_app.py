import streamlit as st
from backend.weiboQA import load_faiss_vectorstore, answer_question

# ---------- Cache the vectorstore so it's not reloaded every time ----------
@st.cache_resource
def get_vectorstore():
    return load_faiss_vectorstore()

# ---------- Streamlit app UI ----------
def main():
    st.set_page_config(page_title="Weibo GenAI QA", page_icon="🐣", layout="wide")
    st.title("🐣 Weibo GenAI QA (Bilingual)")

    st.markdown(
        "Ask questions about the actor's Weibo posts in **Chinese or English**.\n\n"
        "The assistant answers using only the scraped Weibo posts (Chinese + English translation)."
    )

    # Load vectorstore once
    vectorstore = get_vectorstore()

    # User input
    question = st.text_area(
        "Your question (可以用中文或英文提问):",
        value="罗云熙最近在微博上有提到他的工作计划吗？",
        height=80,
    )

    k = st.slider("Max number of posts used in the answer:", min_value=1, max_value=10, value=5)

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                answer, docs = answer_question(question, vectorstore, k=k)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Show model context (retrieved posts)", expanded=False):
                if not docs:
                    st.write("No posts were retrieved for this question.")
                else:
                    for i, d in enumerate(docs, start=1):
                        m = d.metadata or {}
                        created_at = m.get("created_at", "Unknown time")
                        likes = m.get("like_num", "N/A")
                        comments = m.get("comment_num", "N/A")
                        reposts = m.get("repost_num", "N/A")
                        raw_zh = m.get("raw_zn") or ""
                        raw_en = m.get("raw_en") or ""

                        st.markdown(
                            f"**Post {i}**  |  time: `{created_at}`  |  👍 {likes}  💬 {comments}  🔁 {reposts}"
                        )

                        # Show original Chinese and English separately if available
                        if raw_zh or raw_en:
                            if raw_zh:
                                st.markdown("**Chinese (原文):**")
                                st.write(raw_zh)
                            if raw_en:
                                st.markdown("**English (translation):**")
                                st.write(raw_en)
                        else:
                            # fallback to combined text
                            st.write(d.page_content)

                        st.markdown("---")


if __name__ == "__main__":
    main()