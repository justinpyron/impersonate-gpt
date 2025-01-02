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
WHAT_IS_THIS_APP = """
This app demos [GPT2 models](https://huggingface.co/openai-community/gpt2) fine-tuned to mimic four famous writers with distinctive voices:
1. Charles Darwin
2. Fyodor Dostoevsky
3. F. Scott Fitzgerald
4. Mark Twain

For each author, I obtained a dataset of four books from [Project Gutenberg](https://www.gutenberg.org). Each model was fine-tuned on these books for 10 epochs on an Nvidia L4 GPU on a Google Compute Engine VM.

Source code 👉 [GitHub](https://github.com/justinpyron/impersonate-gpt)
"""


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
    st.markdown(WHAT_IS_THIS_APP)
footer_html = """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            height: 35px;
            color: white;
            background-color: #262730;
            text-align: center;
            padding: 2px;
            opacity: 1;
        }
    </style>
    <div class="footer">
        <p>⚠️ It may take up to two minutes for the inference server to start after the first 'Generate text' button click.</p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

with st.form("inputs", enter_to_submit=False, border=False):
    text_seed = st.text_area(
        "Enter seed text",
        "",
        help="The app will generate text starting from what you enter here",
    )
    col1, col2 = st.columns([3, 2])
    with col1:
        selected_writers = st.segmented_control(
            "Writers to mimic",
            options=WRITERS,
            selection_mode="multi",
            help="The writers whose voice to emulate in the generated output",
        )
        selected_writers = [
            w for w in WRITERS if w in selected_writers
        ]  # Preserve order
    with col2:
        with st.expander("Settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                value=1.0,
                help="Controls randomness of generated text. Lower values are less random.",
            )
            num_tokens = st.slider(
                "Number of tokens",
                min_value=10,
                max_value=200,
                step=10,
                value=80,
                help="The number of tokens to generate",
            )
    submitted = st.form_submit_button(
        "Generate text", use_container_width=True, type="primary"
    )

if submitted and len(selected_writers) > 0:
    columns = st.columns(len(selected_writers))
    for col, writer in zip(columns, selected_writers):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {writer} says...")
                with st.spinner("Computing..."):
                    generated_text = ping_api(
                        writer.lower(),
                        text_seed,
                        temperature,
                        num_tokens,
                    )
                st.markdown(generated_text)
st.markdown(footer_html, unsafe_allow_html=True)
