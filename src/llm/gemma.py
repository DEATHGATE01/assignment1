from __future__ import annotations

from typing import Optional


class GemmaModel:
    """Wrapper for Gemma models via Ollama or compatible API."""

    def __init__(
        self,
        model_name: str = "gemma:7b",
        api_url: str = "http://localhost:11434/api/generate",
    ):
        """Initialize Gemma model.
        
        Args:
            model_name: Model identifier (e.g., "gemma:7b", "gemma:13b").
            api_url: Ollama or compatible API endpoint.
        """
        self.model_name = model_name
        self.api_url = api_url

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate text using Gemma model.
        
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
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.api_url}. "
                "Make sure Ollama is running: ollama serve"
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
