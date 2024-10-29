import time
import utils
import config
import streamlit as st

from elasticsearch import Elasticsearch

# Elasticsearch client
client = Elasticsearch(hosts=[config.HOST_URL])

# Page setup
st.set_page_config(page_title="Comm100 agent-assist-enhancement",
                   page_icon="🔍")
st.title("POC for agent-assist enhancement")

# st.session_state.update(st.session_state)

# Use a text input to get user query
if 'retrieved_docs' not in st.session_state:
    st.session_state.retrieved_docs = None


def on_click_callback() -> None:
    """
    Callback fucntion 
    """
    user_query = st.session_state.user_query
    st.session_state.retrieved_docs = None

    if not user_query:
        alert = st.error('Please put in a search query', icon="🚨")
        time.sleep(2)
        alert.empty()
        return None

    st.session_state.retrieved_docs = utils.get_ranked_docs_from_query(client, user_query)
    # st.session_state.retrieved_docs = utils.books


query_placeholder = st.form("chat-form")

with query_placeholder:
    st.markdown("_Press enter to submit your **query**_")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "query",
        label_visibility="collapsed",
        key="user_query",
    )
    submit_button = cols[1].form_submit_button(
        "Submit",
        type="primary"
    )

    if submit_button:
        on_click_callback()

if st.session_state.retrieved_docs:
    st.write("<br>", unsafe_allow_html=True)
    text = '<p style="font-size: 30px;"><strong>Result sorted by semantic scores</strong></p>'
    st.markdown(text, unsafe_allow_html=True)
    st.write("<hr>", unsafe_allow_html=True)
    container = st.container()

    with container:
        rows = [st.columns(3) for _ in range(3)]

        for idx, col in enumerate(sum(rows, [])):
            if idx < len(st.session_state.retrieved_docs):
                with col:
                    st.markdown(f"**Document {idx + 1}**")
                    st.write(st.session_state.retrieved_docs[idx])
