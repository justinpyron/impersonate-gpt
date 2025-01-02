import requests
import streamlit as st

WRITERS = [
    "GPT2",
    "Darwin",
    "Dostoevsky",
    "Fitzgerald",
    "Twain",
]
URL = "https://impersonate-gpt-623148967155.us-central1.run.app"


def ping_api(
    endpoint: str,
    text: str,
    temperature: float,
    num_tokens: float,
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
    text_seed = st.text_area("Enter some text", "")
    selected_writers = st.segmented_control(
        "Writers",
        options=WRITERS,
        selection_mode="multi",
        help="The writers whose voice to mimic in the generated output",
    )
    selected_writers = [w for w in WRITERS if w in selected_writers]  # Preserve order
    submitted = st.form_submit_button("Submit", use_container_width=True)

if submitted and len(selected_writers) > 0:
    columns = st.columns(len(selected_writers))
    for col, writer in zip(columns, selected_writers):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {writer} says...")
                with st.spinner("Computing..."):
                    out = ping_api(writer.lower(), text_seed, 1, 80)
                # sample_text = "Lorem ipsum odor amet, consectetuer adipiscing elit. Diam leo blandit ipsum blandit lacus dictum duis. Porta adipiscing efficitur arcu volutpat sodales tempus imperdiet ridiculus per. Cras in malesuada feugiat magna; ipsum nullam erat auctor. Ad sed habitant dignissim finibus dapibus metus erat fames euismod? Habitant consectetur magnis eget adipiscing litora magna commodo velit. Vestibulum senectus porttitor placerat, sapien rhoncus quis sit curae. Facilisi nam elit fusce donec augue."
                st.markdown(out)
