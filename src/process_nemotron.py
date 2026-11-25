#!/usr/bin/env python3
"""
SCOPE: Nemotron Persona Processing Pipeline

Processes personas from the NVIDIA Nemotron-Personas-USA dataset
and augments them with SCOPE questionnaire responses.

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA

Usage:
    python process_nemotron.py --output nemotron_augmented.jsonl --model gpt-4o --limit 100

Author: Pranav Narayanan Venkit, Yu Li, Yada Pruksachatkun, Chien-Sheng Wu
Paper: https://arxiv.org/abs/2601.07110
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import aiofiles
import openai
from datasets import load_dataset

try:
    from .utils import (
        DEFAULT_FACET_SECTIONS,
        build_answer_format_prompt,
        extract_question_numbers,
        parse_facet_responses,
        parse_questionnaire_sections,
    )
except ImportError:
    from utils import (
        DEFAULT_FACET_SECTIONS,
        build_answer_format_prompt,
        extract_question_numbers,
        parse_facet_responses,
        parse_questionnaire_sections,
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _resolve_llm_backend(
    model: str,
    api_key_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Determine which LLM backend to use for a given model name.

    Decision logic
    --------------
    1. Read OPENAI_MODELS from the environment.  This should be a JSON array
       string, e.g. ``'["poligpt"]'``.  Shell-style values such as
       ``["poligpt"]`` (without quotes around the array elements) are also
       handled via a fallback CSV parser.
    2. If `model` is in that list -> use the **openai** backend with
       OPENAI_API_KEY / OPENAI_BASE_URL.
    3. Otherwise -> use the **litellm** backend with LLM_API_KEY / LLM_BASE_URL.
       Ollama does not require a real API key; we default to the string
       ``"ollama"`` so that openai.AsyncOpenAI (used internally by litellm)
       does not reject the call for a missing key.

    CLI overrides (`--api-key`, `--base-url`) always take precedence over
    environment variables.

    Args:
        model:             Model identifier (e.g. ``"poligpt"`` or ``"llama3"``).
        api_key_override:  Explicit API key from ``--api-key`` CLI flag.
        base_url_override: Explicit base URL from ``--base-url`` CLI flag.

    Returns:
        A 3-tuple ``(backend, api_key, base_url)`` where *backend* is either
        ``"openai"`` or ``"litellm"``.

    Edge cases:
        - OPENAI_MODELS not set -> treated as ``[]`` (no TypeError on ``in``).
        - OPENAI_MODELS is not valid JSON -> parsed as comma-separated values so
          shell-style ``["poligpt"]`` still works.
        - LLM_API_KEY not set -> defaults to ``"ollama"`` for litellm path.
        - api_key_override / base_url_override always win.
    """
    import json

    openai_models_raw = os.environ.get("OPENAI_MODELS", "[]")
    try:
        openai_models: list = json.loads(openai_models_raw)
    except (json.JSONDecodeError, ValueError):
        openai_models = [
            m.strip().strip('"').strip("'")
            for m in openai_models_raw.strip("[] ").split(",")
            if m.strip().strip('"').strip("'")
        ]

    if model in openai_models:
        api_key = api_key_override or os.environ.get("OPENAI_API_KEY")
        base_url = base_url_override or os.environ.get("OPENAI_BASE_URL")
        return "openai", api_key, base_url
    else:
        api_key = api_key_override or os.environ.get("LLM_API_KEY") or "ollama"
        base_url = base_url_override or os.environ.get("LLM_BASE_URL")
        return "litellm", api_key, base_url


class NemotronProcessor:
    """Processes Nemotron personas and augments them with SCOPE responses."""

    FACET_SECTIONS = DEFAULT_FACET_SECTIONS

    # Key Nemotron persona attributes to include in prompts
    NEMOTRON_ATTRIBUTES = [
        "persona_age",
        "persona_gender",
        "persona_main",
        "professional_persona",
        "interests_persona",
        "family_persona",
        "political_persona",
        "economic_persona",
        "community_persona",
        "health_persona",
        "character_persona"
    ]

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        questionnaire_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the Nemotron processor.

        Args:
            model: LLM model identifier
            api_key: API key (defaults to OPENAI_API_KEY or LLM_API_KEY env var)
            base_url: API base URL (defaults to OPENAI_BASE_URL or LLM_BASE_URL env var)
            temperature: LLM temperature for response generation
            max_tokens: Maximum tokens per response
            questionnaire_path: Optional path to questionnaire markdown file
                (defaults to data/questionnaire.md)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # -- Backend routing ---------------------------------------------------
        # Same logic as augment_persona.py — see _resolve_llm_backend docstring.
        self.backend, resolved_api_key, resolved_base_url = _resolve_llm_backend(
            model, api_key_override=api_key, base_url_override=base_url
        )
        self.api_key = resolved_api_key

        if not resolved_api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY or LLM_API_KEY environment variable, "
                "or pass --api-key / api_key parameter."
            )

        if self.backend == "openai":
            client_kwargs: dict = {"api_key": resolved_api_key}
            if resolved_base_url:
                client_kwargs["base_url"] = resolved_base_url
            self.client = openai.AsyncOpenAI(**client_kwargs)
        else:
            self.client = None

        self.base_url = resolved_base_url

        # Load questionnaire
        if questionnaire_path is not None:
            if not str(questionnaire_path).strip():
                raise ValueError("Questionnaire path cannot be empty.")
            self.questionnaire_path = Path(questionnaire_path)
        else:
            self.questionnaire_path = Path(__file__).parent.parent / "data" / "questionnaire.md"
        self.facet_questions = self._parse_questionnaire()

    def _parse_questionnaire(self) -> Dict[str, str]:
        """Parse questionnaire markdown file and extract content for each facet section."""
        if not self.questionnaire_path.is_file():
            raise FileNotFoundError(f"Questionnaire file not found: {self.questionnaire_path}")

        with open(self.questionnaire_path, 'r', encoding='utf-8') as f:
            content = f.read()

        facet_content = parse_questionnaire_sections(content, self.FACET_SECTIONS)
        logger.info(f"Parsed {len([f for f in facet_content.values() if f])} facet sections")
        return facet_content

    def _format_nemotron_persona(self, row: Dict) -> str:
        """
        Format Nemotron persona data into a structured prompt.

        Args:
            row: Nemotron dataset row

        Returns:
            Formatted persona string
        """
        lines = [
            "NEMOTRON PERSONA PROFILE",
            "=" * 50
        ]

        # Extract key attributes
        for attr in self.NEMOTRON_ATTRIBUTES:
            if attr in row and row[attr]:
                # Clean attribute name for display
                display_name = attr.replace('_', ' ').replace('persona', '').strip().title()
                if display_name:
                    lines.append(f"\n### {display_name}")
                    lines.append(str(row[attr]))

        # Add any additional fields not in the standard list
        for key, value in row.items():
            if key not in self.NEMOTRON_ATTRIBUTES and key not in ('uuid', 'index'):
                if value and str(value).strip():
                    display_name = key.replace('_', ' ').title()
                    lines.append(f"\n### {display_name}")
                    lines.append(str(value))

        return "\n".join(lines)

    def _create_prompt(
        self,
        persona_content: str,
        facet_name: str,
        facet_questions: str
    ) -> Tuple[str, str]:
        """
        Create system and user prompts for generating facet responses.

        Args:
            persona_content: Formatted Nemotron persona
            facet_name: Name of the facet being evaluated
            facet_questions: Questions for this facet

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = (
            "You are an expert at embodying personas and authentically answering "
            "questionnaires as that person would. You provide complete, thoughtful responses."
        )

        answer_format = build_answer_format_prompt(extract_question_numbers(facet_questions))

        user_prompt = f"""You are embodying the following Nemotron persona. Answer the questionnaire questions exactly as this person would, based on the comprehensive persona profile provided.

{persona_content}

---

FACET: {facet_name}

{facet_questions}

{answer_format}

IMPORTANT INSTRUCTIONS:
1. Answer EVERY question in the section
2. Answer as the persona, not as an AI
3. Draw from all aspects of the Nemotron persona profile
4. For multiple choice questions, provide ONLY the option text (e.g., "Bachelor's" not "Option C")
5. For numeric scales (1-5), provide ONLY the number
6. For open-ended questions, provide thoughtful responses (2-4 sentences) reflecting the persona's voice
7. Stay true to the persona's background, values, personality, and life circumstances
8. Use first-person perspective (I, me, my)
9. Be consistent with all aspects of the persona profile

CRITICAL: You MUST choose from the exact options provided. Do not paraphrase or use similar words."""

        return system_prompt, user_prompt

    async def _call_llm(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """
        Make async LLM API call with retry logic.

        Dispatches to the correct backend based on self.backend:
          - "openai"  -> openai.AsyncOpenAI (self.client)
          - "litellm" -> litellm.acompletion (lazy import)

        See _resolve_llm_backend() for backend selection logic.

        Edge cases:
          - Transient errors (rate limit, timeout, connection) are retried with
            exponential back-off for both backends.
          - LiteLLM exceptions are caught via the generic Exception handler and
            classified as transient by inspecting the error message.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        for attempt in range(max_retries + 1):
            try:
                if self.backend == "openai":
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    from litellm import acompletion  # noqa: PLC0415
                    response = await acompletion(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        api_base=self.base_url,
                        api_key=self.api_key,
                    )

                return response.choices[0].message.content

            except (openai.RateLimitError, openai.APITimeoutError,
                    openai.APIConnectionError, openai.InternalServerError) as e:
                if attempt == max_retries:
                    logger.error(f"API call failed after {max_retries + 1} attempts: {e}")
                    raise

                backoff_time = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff_time}s...")
                await asyncio.sleep(backoff_time)

            except Exception as e:
                err_msg = str(e).lower()
                is_transient = any(k in err_msg for k in ("rate limit", "timeout", "connection", "overloaded"))
                if is_transient and attempt < max_retries:
                    backoff_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed (transient): {e}. Retrying in {backoff_time}s...")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Non-retryable API error: {e}")
                    raise

    def _parse_responses(
        self,
        llm_output: str,
        expected_questions: Optional[List[int]] = None
    ) -> Dict[str, str]:
        """Parse LLM output into question-answer pairs."""
        return parse_facet_responses(llm_output, expected_questions=expected_questions)

    async def _stream_to_file(self, result: Dict, output_file: str):
        """Stream individual result to file in JSON Lines format."""
        async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(result) + '\n')

    async def _process_single_persona(
        self,
        row: Dict,
        index: int,
        facets: Optional[List[str]],
        output_file: str
    ) -> bool:
        """
        Process a single Nemotron persona.

        Args:
            row: Nemotron dataset row
            index: Row index
            facets: List of facets to generate (None for all)
            output_file: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store UUID and remove from row for formatting
            original_uuid = row.get("uuid", f"persona_{index}")

            # Format persona
            persona_content = self._format_nemotron_persona(row)
            facet_responses = {}

            # Determine which facets to process
            facets_to_process = facets or list(self.FACET_SECTIONS.keys())

            for facet_key in facets_to_process:
                if facet_key not in self.facet_questions:
                    continue

                questions = self.facet_questions[facet_key]
                if not questions:
                    continue

                facet_name = self.FACET_SECTIONS[facet_key]
                system_prompt, user_prompt = self._create_prompt(
                    persona_content, facet_name, questions
                )

                response = await self._call_llm(system_prompt, user_prompt)
                parsed = self._parse_responses(
                    response, expected_questions=extract_question_numbers(questions)
                )
                facet_responses[facet_key] = parsed

            # Build result
            result = {
                "index": index,
                "uuid": original_uuid,
                "nemotron_attributes": {
                    attr: row.get(attr) for attr in self.NEMOTRON_ATTRIBUTES if row.get(attr)
                },
                "facet_responses": facet_responses
            }

            # Stream to file
            await self._stream_to_file(result, output_file)
            logger.info(f"Processed persona {index + 1}: {original_uuid}")
            return True

        except Exception as e:
            logger.error(f"Error processing persona {index}: {e}")
            return False

    async def process_dataset(
        self,
        output_file: str,
        facets: Optional[List[str]] = None,
        limit: Optional[int] = None,
        start_idx: int = 0,
        batch_size: int = 50
    ) -> int:
        """
        Process Nemotron dataset with streaming output.

        Args:
            output_file: Output JSONL file path
            facets: List of facets to generate (None for all)
            limit: Maximum personas to process
            start_idx: Starting index in dataset
            batch_size: Concurrent requests per batch

        Returns:
            Number of personas successfully processed
        """
        logger.info("Loading Nemotron-Personas-USA dataset...")

        try:
            dataset = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
            logger.info(f"Dataset loaded: {len(dataset)} personas")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Clear output file
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            pass

        # Determine processing range
        end_idx = min(start_idx + limit, len(dataset)) if limit else len(dataset)
        logger.info(f"Processing personas {start_idx} to {end_idx - 1}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Batch size: {batch_size}")

        processed_count = 0

        # Process in batches
        for batch_start in range(start_idx, end_idx, batch_size):
            batch_end = min(batch_start + batch_size, end_idx)
            logger.info(f"Processing batch: personas {batch_start} to {batch_end - 1}")

            # Create tasks
            tasks = []
            for i in range(batch_start, batch_end):
                row = dict(dataset[i])
                task = self._process_single_persona(row, i, facets, output_file)
                tasks.append(task)

            # Execute batch
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes
            batch_successes = sum(1 for r in results if r is True)
            processed_count += batch_successes

            logger.info(f"Batch complete: {batch_successes}/{len(tasks)} succeeded")

        logger.info(f"Processing complete. {processed_count} personas written to {output_file}")
        return processed_count


def parse_facets(facets_str: str) -> Optional[List[str]]:
    """Parse facets argument from command line."""
    if facets_str.lower() == 'all':
        return None
    return [f.strip().lower() for f in facets_str.split(',')]


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SCOPE: Process Nemotron personas with sociopsychological questionnaire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_nemotron.py --output nemotron_augmented.jsonl
  python process_nemotron.py --output out.jsonl --limit 100 --facets values,personality
  python process_nemotron.py --output out.jsonl --model gpt-4o-mini --batch-size 100

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA
Paper: https://arxiv.org/abs/2601.07110
        """
    )

    parser.add_argument(
        '--output', '-o',
        default='nemotron_augmented.jsonl',
        help='Output JSONL file path (default: nemotron_augmented.jsonl)'
    )
    parser.add_argument(
        '--model', '-m',
        default='gpt-4o',
        help=(
            'LLM model to use (default: gpt-4o). '
            'Models listed in the OPENAI_MODELS env var use the OpenAI backend '
            '(OPENAI_API_KEY / OPENAI_BASE_URL); all other models use the '
            'LiteLLM backend via LLM_BASE_URL.'
        )
    )
    parser.add_argument(
        '--facets', '-f',
        default='all',
        help='Facets to generate, comma-separated or "all" (default: all)'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.3,
        help='LLM temperature (default: 0.3)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=50,
        help='Concurrent requests per batch (default: 50)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of personas to process'
    )
    parser.add_argument(
        '--start-idx', '-s',
        type=int,
        default=0,
        help='Starting index in dataset (default: 0)'
    )
    parser.add_argument(
        '--questionaire', '--questionnaire', '-q',
        dest='questionnaire',
        default=None,
        help='Path to questionnaire markdown file (default: data/questionnaire.md)'
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='API key (default: from OPENAI_API_KEY or LLM_API_KEY env var)'
    )
    parser.add_argument(
        '--base-url',
        default=None,
        help='API base URL (default: from OPENAI_BASE_URL or LLM_BASE_URL env var)'
    )

    args = parser.parse_args()

    # Validate custom questionnaire file early if provided
    if args.questionnaire is not None and (
        not args.questionnaire.strip() or not Path(args.questionnaire).is_file()
    ):
        logger.error(f"Questionnaire file not found: {args.questionnaire}")
        return

    # Parse facets
    facets = parse_facets(args.facets)

    # Initialize processor
    try:
        processor = NemotronProcessor(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            questionnaire_path=args.questionnaire
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        return

    # Process dataset
    logger.info("Starting Nemotron persona processing")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Backend: {processor.backend} ({processor.base_url})")
    logger.info(f"  Questionnaire: {processor.questionnaire_path}")
    logger.info(f"  Facets: {facets or 'all'}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Start index: {args.start_idx}")
    if args.limit:
        logger.info(f"  Limit: {args.limit}")

    count = await processor.process_dataset(
        output_file=args.output,
        facets=facets,
        limit=args.limit,
        start_idx=args.start_idx,
        batch_size=args.batch_size
    )

    logger.info(f"Done. Processed {count} Nemotron personas.")


if __name__ == "__main__":
    asyncio.run(main())
