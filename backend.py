"""Modal-based inference server for text generation."""

import modal

# =============================================================================
# Global Configuration Parameters
# =============================================================================

VOLUME_NAME = "impersonate-gpt"
VOLUME_MOUNT_PATH = "/data"
MODEL_FOLDER_PATH = "gemma-3-270m"
SCALEDOWN_WINDOW_SECONDS = 60
ADAPTER_PATHS = {
    "darwin": "PLACEHOLDER_PATH",
    "dostoevsky": "PLACEHOLDER_PATH",
    "twain": "PLACEHOLDER_PATH",
}

# =============================================================================
# Modal Setup
# =============================================================================

app = modal.App("impersonate-gpt")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.10.0",  # TODO: Remove
    "transformers==4.55.4",
    "peft==0.14.0",
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
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
class Server:
    """Modal class for serving model inference."""

    @modal.enter()
    def load_model_and_tokenizer(self):
        """Load model and tokenizer on container startup."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = f"{VOLUME_MOUNT_PATH}/{MODEL_FOLDER_PATH}"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_path)
        base_model.eval()

        # Load first adapter
        adapter_names = list(ADAPTER_PATHS.keys())
        first_adapter_name = adapter_names[0]
        first_adapter_path = f"{VOLUME_MOUNT_PATH}/{ADAPTER_PATHS[first_adapter_name]}"
        self.model = PeftModel.from_pretrained(
            base_model, first_adapter_path, adapter_name=first_adapter_name
        )

        # Load remaining adapters
        for adapter_name in adapter_names[1:]:
            adapter_path = f"{VOLUME_MOUNT_PATH}/{ADAPTER_PATHS[adapter_name]}"
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)

    def generate(
        self,
        text: str,
        temperature: float,
        num_tokens: int,
        adapter_name: str,
    ) -> str:
        """
        Generate text continuation for a single input.

        Args:
            text: The seed text to continue
            temperature: Temperature for sampling (higher = more random)
            num_tokens: Number of new tokens to generate
            adapter_name: Name of the LoRA adapter to use

        Returns:
            The full generated text (seed + continuation)
        """
        # Switch to the requested adapter
        self.model.set_adapter(adapter_name)

        # Tokenize input
        input_tokens = self.tokenizer(text, return_tensors="pt")

        # Generate continuation
        output_tokens = self.model.generate(
            **input_tokens,
            max_new_tokens=num_tokens,
            temperature=temperature,
        )

        # Decode and return full text
        generated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True
        )

        return generated_text

    @modal.asgi_app()
    def fastapi_server(self):
        """Create and configure the FastAPI application."""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        class GenerateRequest(BaseModel):
            text: str
            temperature: float
            num_tokens: int

        class GenerateResponse(BaseModel):
            generated_text: str

        server = FastAPI(title="ImpersonateGPT API")

        @server.post("/generate/{adapter_name}", response_model=GenerateResponse)
        def generate_endpoint(
            adapter_name: str, request: GenerateRequest
        ) -> GenerateResponse:
            """
            Generate text continuation from seed text using specified adapter.

            Args:
                adapter_name: Name of the LoRA adapter to use (darwin, dostoevsky, twain)
                request: Request containing seed text and generation parameters

            Returns:
                Response with the full generated text
            """
            if adapter_name not in ADAPTER_PATHS:
                raise HTTPException(
                    status_code=404,
                    detail=f"Adapter '{adapter_name}' not found. Available adapters: {list(ADAPTER_PATHS.keys())}",
                )

            generated_text = self.generate(
                text=request.text,
                temperature=request.temperature,
                num_tokens=request.num_tokens,
                adapter_name=adapter_name,
            )
            return GenerateResponse(generated_text=generated_text)

        @server.get("/adapters")
        def list_adapters():
            """List all available adapters."""
            return {"adapters": list(ADAPTER_PATHS.keys())}

        @server.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        return server
