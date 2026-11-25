#!/usr/bin/env python3
"""
SCOPE: Generic Persona Augmentation Pipeline

Augments any persona with responses to the SCOPE questionnaire protocol.
Supports any OpenAI-compatible API endpoint.

Usage:
    python augment_persona.py --input personas.jsonl --output augmented.jsonl --model gpt-4o

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


class PersonaAugmenter:
    """Augments personas with SCOPE questionnaire responses using an LLM."""

    FACET_SECTIONS = DEFAULT_FACET_SECTIONS

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
        Initialize the persona augmenter.

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

        # Resolve API credentials from environment if not provided
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")

        if not resolved_api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY or LLM_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client
        client_kwargs = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.client = openai.AsyncOpenAI(**client_kwargs)

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

    def _create_prompt(self, persona_content: str, facet_name: str, facet_questions: str) -> Tuple[str, str]:
        """
        Create system and user prompts for generating facet responses.

        Args:
            persona_content: Formatted persona information
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

        user_prompt = f"""You are embodying the following persona. Answer the questionnaire questions exactly as this person would, based on the persona information provided.

{persona_content}

---

FACET: {facet_name}

{facet_questions}

{answer_format}

IMPORTANT INSTRUCTIONS:
1. Answer EVERY question in the section
2. Answer as the persona, not as an AI
3. Use only information from the persona above
4. For multiple choice questions, provide ONLY the option text (e.g., "Bachelor's" not "Option C")
5. For numeric scales (1-5), provide ONLY the number
6. For open-ended questions, provide thoughtful responses (2-4 sentences)
7. Stay true to the persona's background, values, and personality
8. Use first-person perspective (I, me, my)
9. Be consistent with the persona's characteristics

CRITICAL: You MUST choose from the exact options provided. Do not paraphrase or use similar words."""

        return system_prompt, user_prompt

    def _format_persona(self, persona: Dict) -> str:
        """
        Format persona dictionary into a readable string for the LLM.

        Args:
            persona: Dictionary containing persona attributes

        Returns:
            Formatted persona string
        """
        lines = ["PERSONA INFORMATION:", "=" * 40]

        for key, value in persona.items():
            if key.lower() in ('id', 'uuid', 'index'):
                continue
            # Convert key to readable format
            readable_key = key.replace('_', ' ').title()
            lines.append(f"{readable_key}: {value}")

        return "\n".join(lines)

    async def _call_llm(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """
        Make async LLM API call with retry logic.

        Args:
            system_prompt: System message
            user_prompt: User message
            max_retries: Number of retry attempts

        Returns:
            LLM response content
        """
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
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
                logger.error(f"Non-retryable API error: {e}")
                raise

    def _parse_responses(
        self,
        llm_output: str,
        expected_questions: Optional[List[int]] = None
    ) -> Dict[str, str]:
        """
        Parse LLM output into question-answer pairs.

        Args:
            llm_output: Raw LLM response
            expected_questions: Expected ordered question numbers for this facet

        Returns:
            Dictionary mapping question numbers to answers
        """
        return parse_facet_responses(llm_output, expected_questions=expected_questions)

    async def augment_persona(
        self,
        persona: Dict,
        facets: Optional[List[str]] = None
    ) -> Dict:
        """
        Augment a single persona with questionnaire responses.

        Args:
            persona: Persona dictionary
            facets: List of facets to generate (None for all)

        Returns:
            Dictionary with original persona and facet responses
        """
        if facets is None:
            facets = list(self.FACET_SECTIONS.keys())

        persona_content = self._format_persona(persona)
        facet_responses = {}

        for facet_key in facets:
            if facet_key not in self.facet_questions:
                logger.warning(f"Unknown facet: {facet_key}")
                continue

            questions = self.facet_questions[facet_key]
            if not questions:
                logger.warning(f"No questions found for facet: {facet_key}")
                continue

            facet_name = self.FACET_SECTIONS[facet_key]
            system_prompt, user_prompt = self._create_prompt(
                persona_content, facet_name, questions
            )

            expected_questions = extract_question_numbers(questions)
            try:
                response = await self._call_llm(system_prompt, user_prompt)
                parsed = self._parse_responses(response, expected_questions=expected_questions)
                facet_responses[facet_key] = parsed
                logger.debug(f"Generated {len(parsed)} responses for {facet_key}")
            except Exception as e:
                logger.error(f"Failed to generate responses for {facet_key}: {e}")
                facet_responses[facet_key] = {"error": str(e)}

        return {
            "id": persona.get("id", persona.get("uuid", "unknown")),
            "original_persona": persona,
            "facet_responses": facet_responses
        }

    async def process_file(
        self,
        input_path: str,
        output_path: str,
        facets: Optional[List[str]] = None,
        limit: Optional[int] = None,
        batch_size: int = 50
    ) -> int:
        """
        Process a JSONL file of personas with batched concurrent requests.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            facets: List of facets to generate
            limit: Maximum number of personas to process
            batch_size: Number of concurrent requests per batch

        Returns:
            Number of personas successfully processed
        """
        # Read input personas
        personas = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    personas.append(json.loads(line))
                    if limit and len(personas) >= limit:
                        break

        logger.info(f"Loaded {len(personas)} personas from {input_path}")

        # Clear output file
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            pass

        processed_count = 0

        # Process in batches
        for batch_start in range(0, len(personas), batch_size):
            batch_end = min(batch_start + batch_size, len(personas))
            batch = personas[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start // batch_size + 1}: "
                       f"personas {batch_start + 1} to {batch_end}")

            # Create tasks for concurrent processing
            tasks = [self.augment_persona(p, facets) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Write results
            async with aiofiles.open(output_path, 'a', encoding='utf-8') as f:
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                        continue
                    await f.write(json.dumps(result) + '\n')
                    processed_count += 1

            logger.info(f"Batch complete: {sum(1 for r in results if not isinstance(r, Exception))} succeeded")

        logger.info(f"Processing complete. {processed_count} personas written to {output_path}")
        return processed_count


def parse_facets(facets_str: str) -> Optional[List[str]]:
    """Parse facets argument from command line."""
    if facets_str.lower() == 'all':
        return None
    return [f.strip().lower() for f in facets_str.split(',')]


async def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="SCOPE: Augment personas with sociopsychological questionnaire responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python augment_persona.py --input personas.jsonl --output augmented.jsonl
  python augment_persona.py --input personas.jsonl --output out.jsonl --facets values,personality
  python augment_persona.py --input personas.jsonl --output out.jsonl --model gpt-4o-mini --limit 100

Available facets:
  demographics, sociodemographic, values, personality, behavioral, identity, professional, creativity

Paper: https://arxiv.org/abs/2601.07110
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSONL file containing personas'
    )
    parser.add_argument(
        '--output', '-o',
        default='augmented_personas.jsonl',
        help='Output JSONL file path (default: augmented_personas.jsonl)'
    )
    parser.add_argument(
        '--model', '-m',
        default='gpt-4o',
        help='LLM model to use (default: gpt-4o)'
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

    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Validate custom questionnaire file early if provided
    if args.questionnaire is not None and (
        not args.questionnaire.strip() or not Path(args.questionnaire).is_file()
    ):
        logger.error(f"Questionnaire file not found: {args.questionnaire}")
        return

    # Parse facets
    facets = parse_facets(args.facets)

    # Initialize augmenter
    try:
        augmenter = PersonaAugmenter(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            questionnaire_path=args.questionnaire
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        return

    # Process file
    logger.info(f"Starting persona augmentation")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Questionnaire: {augmenter.questionnaire_path}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Facets: {facets or 'all'}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Batch size: {args.batch_size}")
    if args.limit:
        logger.info(f"  Limit: {args.limit}")

    count = await augmenter.process_file(
        input_path=args.input,
        output_path=args.output,
        facets=facets,
        limit=args.limit,
        batch_size=args.batch_size
    )

    logger.info(f"Done. Processed {count} personas.")


if __name__ == "__main__":
    asyncio.run(main())
