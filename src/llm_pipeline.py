"""
Optional LLM-enhanced text preprocessing via Ollama's REST API.

This module provides text enhancement and dialogue detection for TTS
narration pipelines using a locally-running Ollama instance. It uses
only Python stdlib (urllib.request, json) for HTTP -- no extra
dependencies required.

IMPORTANT: This module is OPTIONAL. It requires:
    1. Ollama installed: https://ollama.com/download
    2. Ollama running:   ollama serve
    3. A model pulled:   ollama pull llama3.2

Without Ollama, the main pipeline works fine -- this module adds
optional preprocessing that can improve narration quality by fixing
OCR errors, expanding abbreviations, and detecting dialogue.

Usage:
    from llm_pipeline import LLMPipeline

    pipeline = LLMPipeline(model='llama3.2')
    cleaned = pipeline.process(raw_text, mode='enhance')
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'llama3.2'
DEFAULT_URL = 'http://localhost:11434'

ENHANCE_PROMPT = (
    "You are a text preprocessor for TTS narration. "
    "Fix OCR errors, expand unusual abbreviations, improve readability. "
    "Return ONLY the modified text with no commentary. "
    "Preserve all original content."
)

DIALOGUE_PROMPT = (
    "Analyze the following text and identify all dialogue with their speakers. "
    "Return a JSON array where each element has 'speaker' (string) and 'text' (string). "
    "If the speaker is unknown, use 'narrator'. "
    "Return ONLY the JSON array, no other text or formatting."
)


def check_ollama(url: str = DEFAULT_URL) -> bool:
    """Check if Ollama is running and accessible.

    Pings the /api/tags endpoint to verify the Ollama server is up.

    Args:
        url: Base URL for the Ollama server.

    Returns:
        True if Ollama is running and responsive, False otherwise.
    """
    try:
        req = urllib.request.Request(f'{url}/api/tags', method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def query_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_URL,
    timeout: int = 60,
) -> str:
    """Send a prompt to Ollama and return the response text.

    Uses the /api/generate endpoint with streaming disabled for a
    single complete response.

    Args:
        prompt: The prompt text to send to the model.
        model: Ollama model name (must be already pulled).
        url: Base URL for the Ollama server.
        timeout: Request timeout in seconds.

    Returns:
        The model's response text.

    Raises:
        RuntimeError: If Ollama is not running, the model is not available,
                      or the request fails for any reason.
    """
    endpoint = f'{url}/api/generate'

    payload = json.dumps({
        'model': model,
        'prompt': prompt,
        'stream': False,
    }).encode('utf-8')

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode('utf-8'))
            return body.get('response', '')
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace') if e.fp else ''
        raise RuntimeError(
            f"Ollama request failed (HTTP {e.code}): {error_body}\n"
            f"Ensure model '{model}' is pulled: ollama pull {model}"
        )
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}: {e.reason}\n"
            "Ensure Ollama is installed and running:\n"
            "  Install: https://ollama.com/download\n"
            "  Start:   ollama serve"
        )
    except TimeoutError:
        raise RuntimeError(
            f"Ollama request timed out after {timeout}s. "
            f"The text may be too long for model '{model}' to process."
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse Ollama response as JSON: {e}")


def enhance_for_narration(text: str, model: str = DEFAULT_MODEL) -> str:
    """Enhance text for TTS narration using an LLM.

    Sends the text through the LLM with instructions to fix OCR errors,
    expand abbreviations, and improve readability while preserving all
    original content.

    Args:
        text: Raw text to enhance.
        model: Ollama model name.

    Returns:
        Enhanced text suitable for TTS narration.

    Raises:
        RuntimeError: If Ollama is not available or the request fails.
    """
    if not text or not text.strip():
        return text

    full_prompt = f"{ENHANCE_PROMPT}\n\n---\n\n{text}"
    logger.info("Enhancing text for narration (%d chars) with model '%s'", len(text), model)

    result = query_ollama(full_prompt, model=model)

    if not result or not result.strip():
        logger.warning("LLM returned empty response; returning original text")
        return text

    return result.strip()


def detect_dialogue(text: str, model: str = DEFAULT_MODEL) -> list[dict]:
    """Detect dialogue and speakers in text using an LLM.

    Asks the LLM to identify dialogue passages and attribute them to
    speakers, returning structured data suitable for multi-voice TTS.

    Args:
        text: Text containing dialogue to analyze.
        model: Ollama model name.

    Returns:
        List of dicts with 'speaker' and 'text' keys. Returns an empty
        list if the LLM response cannot be parsed as valid JSON.

    Raises:
        RuntimeError: If Ollama is not available or the request fails
                      (network/connection errors only -- JSON parse
                      failures return an empty list).
    """
    if not text or not text.strip():
        return []

    full_prompt = f"{DIALOGUE_PROMPT}\n\n---\n\n{text}"
    logger.info("Detecting dialogue (%d chars) with model '%s'", len(text), model)

    raw_response = query_ollama(full_prompt, model=model)

    # Strip markdown code fences if the model wraps its response
    cleaned = raw_response.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith('```')]
        cleaned = '\n'.join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse dialogue detection response as JSON. "
            "Raw response (first 200 chars): %s",
            raw_response[:200],
        )
        return []

    if not isinstance(parsed, list):
        logger.warning("Dialogue detection returned non-list JSON: %s", type(parsed).__name__)
        return []

    # Validate and normalize entries
    result = []
    for entry in parsed:
        if isinstance(entry, dict) and 'speaker' in entry and 'text' in entry:
            result.append({
                'speaker': str(entry['speaker']),
                'text': str(entry['text']),
            })
        else:
            logger.debug("Skipping malformed dialogue entry: %s", entry)

    logger.info("Detected %d dialogue segments", len(result))
    return result


class LLMPipeline:
    """High-level interface for LLM-enhanced text processing.

    Wraps the module's functions into a stateful pipeline that checks
    Ollama availability on initialization and provides a unified
    process() method.

    This class is OPTIONAL. The main audiblez pipeline works without it.
    It requires Ollama to be installed and running locally.

    Example:
        pipeline = LLMPipeline(model='llama3.2')
        enhanced = pipeline.process(raw_text, mode='enhance')
        dialogue = pipeline.process(chapter_text, mode='dialogue_detect')

    Attributes:
        model: The Ollama model name to use for all queries.
        url: The base URL for the Ollama server.
        available: Whether Ollama was reachable at initialization time.
    """

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL):
        """Initialize the LLM pipeline.

        Checks Ollama availability immediately. If Ollama is not running,
        the pipeline is still created but self.available is set to False.
        Calling process() when unavailable will raise RuntimeError.

        Args:
            model: Ollama model name (must be already pulled).
            url: Base URL for the Ollama server.
        """
        self.model = model
        self.url = url
        self.available = check_ollama(url)

        if self.available:
            logger.info("LLM pipeline initialized: model='%s', url='%s'", model, url)
        else:
            logger.warning(
                "Ollama is not available at %s. LLM pipeline features are disabled. "
                "Install and start Ollama to enable: https://ollama.com/download",
                url,
            )

    def process(self, text: str, mode: str = 'enhance') -> str:
        """Process text through the LLM pipeline.

        Args:
            text: Input text to process.
            mode: Processing mode. Supported modes:
                - 'enhance': Fix OCR errors, expand abbreviations, improve
                  readability for TTS narration. Returns enhanced text.
                - 'dialogue_detect': Identify dialogue and speakers. Returns
                  JSON string of the detected dialogue array.

        Returns:
            Processed text. For 'enhance' mode, returns the enhanced text
            string. For 'dialogue_detect' mode, returns a JSON string of
            the dialogue array (use json.loads() to parse).

        Raises:
            RuntimeError: If Ollama is not available.
            ValueError: If an unsupported mode is specified.
        """
        if not self.available:
            # Re-check in case Ollama was started after init
            self.available = check_ollama(self.url)
            if not self.available:
                raise RuntimeError(
                    f"Ollama is not available at {self.url}. "
                    "Start Ollama with: ollama serve"
                )

        if mode == 'enhance':
            return enhance_for_narration(text, model=self.model)
        elif mode == 'dialogue_detect':
            dialogue = detect_dialogue(text, model=self.model)
            return json.dumps(dialogue, ensure_ascii=False, indent=2)
        else:
            raise ValueError(
                f"Unsupported processing mode: '{mode}'. "
                f"Supported modes: 'enhance', 'dialogue_detect'"
            )

    def __repr__(self) -> str:
        status = 'available' if self.available else 'unavailable'
        return f"LLMPipeline(model='{self.model}', url='{self.url}', {status})"
