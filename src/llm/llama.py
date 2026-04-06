from __future__ import annotations

from typing import Optional


class LlamaModel:
    """Wrapper for LLaMA-3.1 models via Ollama or compatible API."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        api_url: str = "http://localhost:11434/api/generate",
        request_timeout_seconds: int = 180,
    ):
        """Initialize LLaMA model.
        
        Args:
            model_name: Model identifier (e.g., "llama2", "llama2-7b").
            api_url: Ollama or compatible API endpoint.
            request_timeout_seconds: HTTP read timeout for generation requests.
        """
        self.model_name = model_name
        self.api_url = api_url
        self.request_timeout_seconds = request_timeout_seconds

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate text using LLaMA model.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0-1.0).
        
        Returns:
            Generated text.
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Requests library required for API calls: pip install requests"
            )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=self.request_timeout_seconds)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.api_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        except requests.exceptions.ReadTimeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.request_timeout_seconds}s for model '{self.model_name}'. "
                "Try a smaller model, reduce generation size, or increase timeout in app settings."
            )
        except requests.exceptions.HTTPError as exc:
            body = ""
            try:
                body = exc.response.text if exc.response is not None else ""
            except Exception:
                body = ""

            if exc.response is not None and exc.response.status_code == 404:
                if "model" in body.lower() and "not found" in body.lower():
                    raise RuntimeError(
                        f"Model '{self.model_name}' was not found in Ollama. "
                        f"Pull it first with: ollama pull {self.model_name}"
                    )
                raise RuntimeError(
                    f"Ollama endpoint not found at {self.api_url}. "
                    "Use the generate endpoint, e.g. http://localhost:11434/api/generate"
                )

            raise RuntimeError(
                f"Ollama request failed ({exc.response.status_code if exc.response is not None else 'unknown status'}): {body or str(exc)}"
            )

    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of prompts.
            max_tokens: Max tokens per generation.
            temperature: Sampling temperature.
        
        Returns:
            List of generated texts.
        """
        return [
            self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            for prompt in prompts
        ]
