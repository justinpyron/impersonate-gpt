import requests
import streamlit as st

URL = "https://impersonate-gpt-623148967155.us-central1.run.app"
WRITERS = [
    "GPT2",
    "Darwin",
    "Dostoevsky",
    "Fitzgerald",
    "Twain",
]


def ping_api(
    endpoint: str,
    text: str,
    temperature: float = 1,
    num_tokens: float = 70,
) -> str:
    response = requests.post(
        url=f"{URL}/{endpoint}",
        headers={"Content-Type": "application/json"},
        json={
            "text": text,
            "temperature": temperature,
            "num_tokens": num_tokens,
        },
    )
    return response.json()["generated_text"]


st.set_page_config(page_title="ImpersonateGPT", layout="centered", page_icon="🥸")

st.title("ImpersonateGPT 🥸")
with st.expander("What is this app?"):
    st.markdown("TODO")

with st.form("inputs", enter_to_submit=False, border=False):
    text_seed = st.text_area("Enter seed text", "")
    selected_writers = st.segmented_control(
        "Writers",
        options=WRITERS,
        selection_mode="multi",
        help="The writers whose voice to mimic in the generated output",
    )
    selected_writers = [w for w in WRITERS if w in selected_writers]  # Preserve order
    submitted = st.form_submit_button("Generate text", use_container_width=True)

if submitted and len(selected_writers) > 0:
    columns = st.columns(len(selected_writers))
    for col, writer in zip(columns, selected_writers):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {writer} says...")
                with st.spinner("Computing..."):
                    generated_text = ping_api(writer.lower(), text_seed)
                st.markdown(generated_text)
