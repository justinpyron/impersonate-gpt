import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL")
WRITERS = [
    "Base",
    "Dostoevsky",
    "Fitzgerald",
    "Twain",
]
WHAT_IS_THIS_APP = """
This app demos LoRA adapters fine-tuned on a base language model to mimic three famous writers with distinctive voices:
1. Fyodor Dostoevsky
2. F. Scott Fitzgerald
3. Mark Twain

For each author, I obtained a dataset of books from [Project Gutenberg](https://www.gutenberg.org) and fine-tuned LoRA adapters using supervised fine-tuning.

You can also select "Base" to generate text using the base LLM without author-specific tuning.

Source code 👉 [GitHub](https://github.com/justinpyron/impersonate-gpt)
"""


def stream_api(
    adapter_name: str,
    text: str,
    temperature: float = 1,
    num_tokens: float = 70,
):
    """
    Stream generated text from the backend as token chunks.

    Args:
        adapter_name: The adapter to use (base, darwin, dostoevsky, twain)
        text: The seed text to generate from
        temperature: Sampling temperature
        num_tokens: Number of tokens to generate

    Yields:
        Token chunks as strings
    """
    if not BACKEND_URL:
        raise ValueError("BACKEND_URL environment variable is not set")

    with httpx.stream(
        "POST",
        f"{BACKEND_URL}/generate/{adapter_name}",
        json={
            "text": text,
            "temperature": temperature,
            "num_tokens": int(num_tokens),
        },
        timeout=300.0,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            yield chunk


st.set_page_config(page_title="ImpersonateGPT", layout="wide", page_icon="🥸")

st.title("ImpersonateGPT 🥸")
with st.expander("What is this app?"):
    st.markdown(WHAT_IS_THIS_APP)

with st.form("inputs", enter_to_submit=False, border=False):
    text_seed = st.text_area(
        "Enter seed text",
        "",
        help="The app will generate text starting from what you enter here",
    )
    col1, col2, col3 = st.columns([2.5, 1, 1])
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
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            value=1.0,
            help="Controls randomness of generated text. Lower values are less random.",
        )
    with col3:
        num_tokens = st.slider(
            "Number of tokens",
            min_value=10,
            max_value=200,
            step=10,
            value=80,
            help="The number of tokens to generate",
        )
    submitted = st.form_submit_button(
        "Generate", use_container_width=True, type="primary"
    )

if submitted and len(selected_writers) > 0:
    columns = st.columns(len(selected_writers))
    for col, writer in zip(columns, selected_writers):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {writer} says...")
                placeholder = st.empty()
                placeholder.markdown("⏳ Waiting for server...")

                def stream_with_status(generator, placeholder):
                    for i, chunk in enumerate(generator):
                        if i == 0:
                            placeholder.empty()
                        yield chunk

                st.write_stream(
                    stream_with_status(
                        stream_api(
                            writer.lower(),
                            text_seed,
                            temperature,
                            num_tokens,
                        ),
                        placeholder,
                    )
                )
