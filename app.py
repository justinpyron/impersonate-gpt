import requests
import streamlit as st

WRITERS = [
    "GPT2",
    "Darwin",
    "Dostoevsky",
    "Fitzgerald",
    "Twain",
]

st.set_page_config(page_title="ImpersonateGPT", layout="centered", page_icon="🥸")

st.title("ImpersonateGPT 🥸")
with st.expander("What is this app?"):
    st.markdown("TODO")

text_seed = st.text_area("Enter some text", "")

sample_text = """
Lorem ipsum odor amet, consectetuer adipiscing elit. Diam leo blandit ipsum blandit lacus dictum duis. Porta adipiscing efficitur arcu volutpat sodales tempus imperdiet ridiculus per. Cras in malesuada feugiat magna; ipsum nullam erat auctor. Ad sed habitant dignissim finibus dapibus metus erat fames euismod? Habitant consectetur magnis eget adipiscing litora magna commodo velit. Vestibulum senectus porttitor placerat, sapien rhoncus quis sit curae. Facilisi nam elit fusce donec augue.
"""
selected_writers = st.segmented_control(
    "Writers",
    options=WRITERS,
    selection_mode="multi",
    help="The writers whose voice to mimic in the generated output",
)
selected_writers = [w for w in WRITERS if w in selected_writers]  # Preserve order
num_writers = len(selected_writers)
if num_writers > 0:
    columns = st.columns(num_writers)
    for col, writer in zip(columns, selected_writers):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {writer} says...")
                st.markdown(sample_text)

if st.button(label="Execute"):
    response = requests.post(
        url="https://simple-translate-623148967155.us-central1.run.app/translate",
        headers={"Content-Type": "application/json"},
        json={
            "text": text_seed,
            "temperature": 0.2,
        },
    )
    st.write(response.json()["translation"])
