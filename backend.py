"""Modal-based inference server for text generation."""

import modal

# =============================================================================
# Global Configuration Parameters
# =============================================================================

VOLUME_NAME = "impersonate-gpt"
VOLUME_MOUNT_PATH = "/data"
MODEL_FOLDER_PATH = "gemma-3-270m"
SCALEDOWN_WINDOW_SECONDS = 60

# =============================================================================
# Modal Setup
# =============================================================================

app = modal.App("impersonate-gpt")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.10.0",
    "transformers==4.55.4",
    "accelerate==1.12.0",
    "fastapi==0.128.0",
    "pydantic==2.12.5",
)

volume = modal.Volume.from_name(VOLUME_NAME)


# =============================================================================
# Server Class
# =============================================================================


@app.cls(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    gpu="T4",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
class Server:
    """Modal class for serving model inference."""

    @modal.enter()
    def load_model_and_tokenizer(self):
        """Load model and tokenizer on container startup."""
        import time

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_total = time.time()

        model_path = f"{VOLUME_MOUNT_PATH}/{MODEL_FOLDER_PATH}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[TIMING] (Loading) Device: {self.device}")

        # Load tokenizer
        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"[TIMING] (Loading) Tokenizer loaded in {time.time() - start:.2f}s")

        # Load model directly to GPU
        start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        print(f"[TIMING] (Loading) Model loaded in {time.time() - start:.2f}s")

        # Set to eval mode
        self.model.eval()

        print(
            f"[TIMING] (Loading) Total model loading: {time.time() - start_total:.2f}s"
        )

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
        import time

        start_total = time.time()

        # Tokenize input and move to GPU
        start = time.time()
        input_tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        print(f"[TIMING] (Inference) Tokenization: {time.time() - start:.2f}s")

        # Generate continuation (output stays on GPU)
        start = time.time()
        output_tokens = self.model.generate(
            **input_tokens,
            max_new_tokens=num_tokens,
            temperature=temperature,
        )
        print(f"[TIMING] (Inference) Generation: {time.time() - start:.2f}s")

        # Decode and return full text (decoder handles device transfer internally)
        start = time.time()
        generated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True
        )
        print(f"[TIMING] (Inference) Decoding: {time.time() - start:.2f}s")

        print(
            f"[TIMING] (Inference) Total generate call: {time.time() - start_total:.2f}s"
        )

        return generated_text

    @modal.asgi_app()
    def fastapi_server(self):
        """Create and configure the FastAPI application."""
        from fastapi import FastAPI
        from pydantic import BaseModel

        class GenerateRequest(BaseModel):
            text: str
            temperature: float
            num_tokens: int

        class GenerateResponse(BaseModel):
            generated_text: str

        server = FastAPI(title="ImpersonateGPT API")

        @server.post("/generate", response_model=GenerateResponse)
        def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
            """
            Generate text continuation from seed text.

            Args:
                request: Request containing seed text and generation parameters

            Returns:
                Response with the full generated text
            """
            generated_text = self.generate(
                text=request.text,
                temperature=request.temperature,
                num_tokens=request.num_tokens,
            )
            return GenerateResponse(generated_text=generated_text)

        @server.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        return server
