"""Modal-based inference server for text generation."""

import modal

# =============================================================================
# Global Configuration Parameters
# =============================================================================

MODAL_VOLUME_NAME = "impersonate-gpt"
MODEL_FOLDER_PATH = "/data/model"
SCALEDOWN_WINDOW_SECONDS = 600

# =============================================================================
# Modal Setup
# =============================================================================

app = modal.App("impersonate-gpt")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1",
    "transformers==4.47.1",
)

volume = modal.Volume.from_name(MODAL_VOLUME_NAME)


# =============================================================================
# Server Class
# =============================================================================


@app.cls(
    image=image,
    volumes={"/data": volume},
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
class Server:
    """Modal class for serving model inference."""

    @modal.enter()
    def load_model(self):
        """Load model and tokenizer on container startup."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer and model from volume
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_FOLDER_PATH)
        self.model.eval()

    def generate(
        self,
        text: str,
        temperature: float,
        num_tokens: int,
    ) -> str:
        """
        Generate text continuation for a single input.

        Args:
            text: The seed text to continue
            temperature: Temperature for sampling (higher = more random)
            num_tokens: Number of new tokens to generate

        Returns:
            The full generated text (seed + continuation)
        """
        # Tokenize input
        input_tokens = self.tokenizer(text, return_tensors="pt")

        # Generate continuation
        output_tokens = self.model.generate(
            **input_tokens,
            max_new_tokens=num_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode and return full text
        generated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True
        )
        return generated_text

    @modal.asgi_app()
    def fastapi_server(self):
        """Create and configure the FastAPI application."""
        from fastapi import FastAPI

        server = FastAPI(title="ImpersonateGPT API")

        @server.post("/generate")
        def generate_endpoint(
            text: str,
            temperature: float,
            num_tokens: int,
        ):
            """
            Generate text continuation from seed text.

            Args:
                text: The seed text to continue
                temperature: Temperature for sampling (higher = more random)
                num_tokens: Number of new tokens to generate

            Returns:
                JSON response with the full generated text
            """
            generated_text = self.generate(
                text=text,
                temperature=temperature,
                num_tokens=num_tokens,
            )
            return {"generated_text": generated_text}

        @server.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        return server
