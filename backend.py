"""Modal-based inference server for text generation."""

import modal

# =============================================================================
# Global Configuration Parameters
# =============================================================================

VOLUME_NAME = "impersonate-gpt"
VOLUME_MOUNT_PATH = "/data"
MODEL_FOLDER_PATH = "gemma-3-270m"
ADAPTERS = {
    "darwin": "weights_sft/darwin_20260206T162059Z",
    "dickens": "weights_sft/dickens_20260206T162433Z",
    "dostoevsky": "weights_sft/dostoevsky_20260206T162221Z",
    "doyle": "weights_sft/doyle_20260206T162518Z",
    "fitzgerald": "weights_sft/fitzgerald_20260206T162358Z",
    "twain": "weights_sft/twain_20260206T145643Z",
}
SCALEDOWN_WINDOW_SECONDS = 600

# =============================================================================
# Modal Setup
# =============================================================================

app = modal.App("impersonate-gpt")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers",
    "peft",
    "fastapi",
    "pydantic",
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
        adapter_names = sorted(list(ADAPTERS.keys()))
        first_adapter_name = adapter_names[0]
        first_adapter_path = f"{VOLUME_MOUNT_PATH}/{ADAPTERS[first_adapter_name]}"
        self.model = PeftModel.from_pretrained(
            base_model, first_adapter_path, adapter_name=first_adapter_name
        )

        # Load remaining adapters
        for adapter_name in adapter_names[1:]:
            adapter_path = f"{VOLUME_MOUNT_PATH}/{ADAPTERS[adapter_name]}"
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)

    def generate(
        self,
        adapter_name: str,
        text: str,
        temperature: float,
        num_tokens: int,
    ):
        """
        Generate text continuation, yielding token chunks as they are produced.

        Args:
            text: The seed text to continue
            temperature: Temperature for sampling (higher = more random)
            num_tokens: Number of new tokens to generate
            adapter_name: Name of the LoRA adapter to use, or "base" for base model

        Yields:
            Token chunks as strings
        """
        import threading

        from transformers import TextIteratorStreamer

        # Tokenize input
        input_tokens = self.tokenizer(text, return_tensors="pt")

        # Set up streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **input_tokens,
            streamer=streamer,
            max_new_tokens=num_tokens,
            temperature=temperature,
        )

        # Run generation in background thread (model.generate blocks)
        def _run():
            if adapter_name == "base":
                with self.model.disable_adapter():
                    self.model.generate(**generation_kwargs)
            else:
                self.model.set_adapter(adapter_name)
                self.model.generate(**generation_kwargs)

        thread = threading.Thread(target=_run)
        thread.start()

        # Yield token chunks as they arrive
        for chunk in streamer:
            yield chunk

        thread.join()

    @modal.asgi_app()
    def fastapi_server(self):
        """Create and configure the FastAPI application."""
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel

        class GenerateRequest(BaseModel):
            text: str
            temperature: float
            num_tokens: int

        server = FastAPI(title="ImpersonateGPT API")

        @server.post("/generate/{adapter_name}")
        def generate_endpoint(
            adapter_name: str, request: GenerateRequest
        ) -> StreamingResponse:
            """
            Stream generated text as plain text token chunks.

            Args:
                adapter_name: Name of the LoRA adapter or "base"
                request: Request containing seed text and generation parameters

            Returns:
                Streaming plain text response of token chunks
            """
            if adapter_name != "base" and adapter_name not in ADAPTERS:
                raise HTTPException(
                    status_code=404,
                    detail=f"Adapter '{adapter_name}' not found. Available: {sorted(list(ADAPTERS.keys()))} or 'base'.",
                )

            return StreamingResponse(
                self.generate(
                    adapter_name=adapter_name,
                    text=request.text,
                    temperature=request.temperature,
                    num_tokens=request.num_tokens,
                ),
                media_type="text/plain",
            )

        @server.get("/adapters")
        def list_adapters():
            """List all available adapters."""
            return {"adapters": sorted(list(ADAPTERS.keys()))}

        @server.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        return server
