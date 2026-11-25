#!/usr/bin/env python3
"""
SCOPE: Shared Utilities

Common utilities for persona processing and evaluation.

Author: Pranav Narayanan Venkit, Yu Li, Yada Pruksachatkun, Chien-Sheng Wu
Paper: https://arxiv.org/abs/2601.07110
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

DEFAULT_FACET_SECTIONS: Dict[str, str] = {
    "demographics": "DEMOGRAPHIC INFORMATION",
    "sociodemographic": "SOCIODEMOGRAPHIC BEHAVIOR",
    "values": "PERSONAL VALUES & MOTIVATIONS",
    "personality": "PERSONALITY TRAITS (Big Five)",
    "behavioral": "BEHAVIORAL PATTERNS & PREFERENCES",
    "identity": "PERSONAL IDENTITY & LIFE NARRATIVES",
    "professional": "PROFESSIONAL IDENTITY & CAREER",
    "creativity": "CREATIVITY & INNOVATION",
}


def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load data from a JSON Lines file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """
    Save data to a JSON Lines file.

    Args:
        data: List of dictionaries to save
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


_LEVEL2_HEADING_RE = re.compile(r"^ {0,3}##(?!#)[ \t]+(.+?)[ \t]*$")
_CODE_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _matches_section_heading(heading_text: str, section_name: str) -> bool:
    """Return True if a level-2 Markdown heading matches ``section_name``."""
    # Strip optional trailing ATX closing hashes (e.g., "## TITLE ##") and colons
    cleaned_heading = re.sub(r"[ \t]+#+$", "", heading_text).rstrip(":").strip()
    norm_heading = " ".join(cleaned_heading.split()).lower()
    # Allow optional leading numbering such as "## 1. DEMOGRAPHIC INFORMATION"
    norm_heading_unnumbered = re.sub(r"^\d+[.)]\s*", "", norm_heading)
    norm_target = " ".join(section_name.split()).rstrip(":").lower()
    if not norm_target:
        return False
    return norm_heading == norm_target or norm_heading_unnumbered == norm_target


def parse_questionnaire_section(content: str, section_name: str) -> str:
    """
    Extract a specific level-2 Markdown section (``## ...``) from questionnaire content.

    Uses line-by-line parsing instead of a ``re.DOTALL`` boundary regex so that:
    - ``### Question N:`` subheadings are never mistaken for ``## `` section boundaries.
    - Sections are bounded by the next level-2 heading (regardless of section order) or EOF.
    - Headings inside fenced code blocks and non-heading summary bullets are ignored.

    Args:
        content: Full questionnaire markdown content
        section_name: Name of section to extract

    Returns:
        Section content string (empty string if missing or invalid input)
    """
    if not content or not section_name or not section_name.strip():
        return ""

    collected: List[str] = []
    collecting = False
    fence_char: Optional[str] = None
    fence_len = 0

    # splitlines() normalizes \n, \r\n, and \r, and handles files without a trailing newline
    for line in content.splitlines():
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence_char is None:
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len and line.strip() == marker[0] * len(line.strip()):
                fence_char = None
                fence_len = 0
            if collecting:
                collected.append(line)
            continue

        if fence_char is None:
            heading_match = _LEVEL2_HEADING_RE.match(line)
            if heading_match:
                if collecting:
                    # Stop at the next level-2 section heading
                    break
                if _matches_section_heading(heading_match.group(1), section_name):
                    collecting = True
                continue

        if collecting:
            collected.append(line)

    return "\n".join(collected).strip()


def parse_questionnaire_sections(
    content: str,
    facet_sections: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Parse all configured facet sections from questionnaire markdown content.

    Args:
        content: Full questionnaire markdown content
        facet_sections: Mapping of facet key to level-2 section heading name.
            Defaults to ``DEFAULT_FACET_SECTIONS``.

    Returns:
        Dictionary mapping each facet key to its extracted section text
    """
    sections = facet_sections if facet_sections is not None else DEFAULT_FACET_SECTIONS
    parsed: Dict[str, str] = {}

    for facet_key, section_name in sections.items():
        section_text = parse_questionnaire_section(content, section_name)
        if not section_text:
            logger.warning(f"Could not find or empty section: {section_name}")
        elif not re.search(r"^###\s+Question\s+\d+:", section_text, re.MULTILINE | re.IGNORECASE):
            logger.warning(f"Section '{section_name}' contains no '### Question N:' items")
        else:
            logger.debug(f"Parsed facet: {facet_key}")
        parsed[facet_key] = section_text

    return parsed


_QUESTION_NUM_RE = re.compile(r"^ {0,3}###\s+Question\s+(\d+)\s*:", re.MULTILINE | re.IGNORECASE)
_QA_LINE_PREFIX = r"^(?:[-*•]\s*|#{1,6}\s*)?(?:\*\*)?(?:Q|Question\s+)"
_QA_LINE_RE = re.compile(
    _QA_LINE_PREFIX + r"(\d+)(?:\*\*)?\s*:\s*(?:\*\*)?(.+?)(?=" + _QA_LINE_PREFIX + r"\d+(?:\*\*)?\s*:|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)
_QA_INLINE_RE = re.compile(r"\bQ(\d+):\s*(.+?)(?=\bQ\d+:|\Z)", re.DOTALL | re.IGNORECASE)
_NUMBERED_LINE_RE = re.compile(r"^ {0,3}(\d+)\.\s*(.+?)(?=^ {0,3}\d+\.\s|\Z)", re.DOTALL | re.MULTILINE)


def extract_question_numbers(facet_questions: str) -> List[int]:
    """Extract ordered question numbers from a parsed facet section."""
    if not facet_questions:
        return []
    return [int(m) for m in _QUESTION_NUM_RE.findall(facet_questions)]


def build_answer_format_prompt(expected_questions: List[int]) -> str:
    """
    Build answer format instructions showing the section's actual question numbers.

    Avoids hardcoding ``Q1:/Q2:/Q3:`` for later facets (e.g. Q116-Q125) so the LLM
    does not restart numbering at Q1 or drift mid-section.
    """
    if not expected_questions:
        return (
            "Please provide your answers in the following format:\n"
            "Q1: [Your answer]\n"
            "Q2: [Your answer]\n"
            "Q3: [Your answer]\n"
            "...and so on."
        )
    if len(expected_questions) == 1:
        q = expected_questions[0]
        return (
            f"Please provide your answer in the following format (use the exact question number Q{q}):\n"
            f"Q{q}: [Your answer]"
        )
    if len(expected_questions) == 2:
        q1, q2 = expected_questions
        return (
            f"Please provide your answers in the following format (use the exact question numbers Q{q1} and Q{q2}):\n"
            f"Q{q1}: [Your answer]\n"
            f"Q{q2}: [Your answer]"
        )
    q_first, q_second, q_third = expected_questions[:3]
    q_last = expected_questions[-1]
    return (
        f"Please provide your answers in the following format "
        f"(use the exact question numbers Q{q_first} through Q{q_last}):\n"
        f"Q{q_first}: [Your answer]\n"
        f"Q{q_second}: [Your answer]\n"
        f"Q{q_third}: [Your answer]\n"
        f"...and so on through Q{q_last}."
    )


def _extract_qa_items(response: str) -> List[Tuple[int, str]]:
    """Extract ordered ``(question_num, answer)`` pairs from raw LLM text."""
    if not response:
        return []
    matches = _QA_LINE_RE.findall(response)
    if (not matches or (len(matches) == 1 and "\n" not in response.strip())) and re.search(
        r"\bQ\d+:", response, re.IGNORECASE
    ):
        inline_matches = _QA_INLINE_RE.findall(response)
        if len(inline_matches) > len(matches):
            matches = inline_matches
    return [(int(q_num), answer.strip()) for q_num, answer in matches]


def parse_facet_responses(
    llm_output: str,
    expected_questions: Optional[List[int]] = None,
) -> Dict[str, str]:
    """
    Parse LLM facet output into ``Q<n>: answer`` pairs and align against expected IDs.

    When ``expected_questions`` is given:
    - Exact ID matches are preserved in expected question order.
    - If the response contains the exact expected number of answers in order but with
      misnumbered labels (e.g. ``Q1, Q12..Q20`` instead of ``Q116..Q125``), answers are
      positionally remapped to ``expected_questions`` with a warning.
    - If counts differ, unambiguous 1-based local numbering is remapped when disjoint
      from ``expected_questions``; otherwise parsed IDs are kept as-is with a warning.
    """
    if not llm_output:
        return {}

    items = _extract_qa_items(llm_output)
    if not items and re.search(r"^ {0,3}\d+\.\s", llm_output, re.MULTILINE):
        items = [(int(q_num), ans.strip()) for q_num, ans in _NUMBERED_LINE_RE.findall(llm_output)]

    if not items:
        return {}

    if not expected_questions:
        return {f"Q{q}": ans for q, ans in items}

    expected_set = set(expected_questions)
    parsed_nums = [q for q, _ in items]

    if len(items) == len(expected_questions) and set(parsed_nums) == expected_set:
        by_num = dict(items)
        return {f"Q{q}": by_num[q] for q in expected_questions}

    if len(items) == len(expected_questions):
        logger.warning(
            "Remapping misnumbered responses %s to expected range Q%d-Q%d",
            [f"Q{q}" for q in parsed_nums],
            expected_questions[0],
            expected_questions[-1],
        )
        return {f"Q{exp_q}": ans for exp_q, (_, ans) in zip(expected_questions, items)}

    logger.warning(
        "Expected %d responses (Q%d-Q%d), parsed %d (%s)",
        len(expected_questions),
        expected_questions[0],
        expected_questions[-1],
        len(items),
        [f"Q{q}" for q in parsed_nums],
    )

    local_range = set(range(1, len(expected_questions) + 1))
    if (
        len(set(parsed_nums)) == len(parsed_nums)
        and all(1 <= q <= len(expected_questions) for q in parsed_nums)
        and local_range.isdisjoint(expected_set)
    ):
        return {f"Q{expected_questions[q - 1]}": ans for q, ans in items}

    return {f"Q{q}": ans for q, ans in items}


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    if not answer:
        return ""

    # Convert to lowercase and strip whitespace
    normalized = answer.lower().strip()

    # Remove common prefixes
    prefixes = ['option', 'choice', 'answer:']
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()

    # Remove quotes
    normalized = normalized.strip('"\'')

    return normalized


def extract_numeric_scale(answer: str) -> Optional[int]:
    """
    Extract numeric value from scale answer.

    Args:
        answer: Answer string

    Returns:
        Integer value (1-5) or None
    """
    if not answer:
        return None

    # Direct numeric
    try:
        val = int(answer.strip())
        if 1 <= val <= 5:
            return val
    except ValueError:
        pass

    # Text mapping
    text_mapping = {
        'very inaccurate': 1, 'strongly disagree': 1, 'not at all': 1,
        'inaccurate': 2, 'disagree': 2, 'slightly': 2,
        'neutral': 3, 'neither': 3, 'moderately': 3,
        'accurate': 4, 'agree': 4, 'very': 4,
        'very accurate': 5, 'strongly agree': 5, 'extremely': 5
    }

    normalized = answer.lower().strip()
    for text, value in text_mapping.items():
        if text in normalized:
            return value

    return None


def calculate_similarity(response1: str, response2: str) -> float:
    """
    Calculate simple string similarity between two responses.

    Args:
        response1: First response
        response2: Second response

    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not response1 or not response2:
        return 0.0

    # Normalize
    norm1 = normalize_answer(response1)
    norm2 = normalize_answer(response2)

    if norm1 == norm2:
        return 1.0

    # Simple word overlap similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def format_persona_summary(persona: Dict, max_length: int = 500) -> str:
    """
    Create a brief summary of a persona.

    Args:
        persona: Persona dictionary
        max_length: Maximum character length

    Returns:
        Summary string
    """
    parts = []

    # Key demographic fields
    demo_fields = ['name', 'age', 'gender', 'occupation', 'location', 'education']
    for field in demo_fields:
        if field in persona and persona[field]:
            parts.append(f"{field.title()}: {persona[field]}")

    # Background/description
    for field in ['background', 'description', 'bio', 'persona_main']:
        if field in persona and persona[field]:
            desc = str(persona[field])[:200]
            parts.append(f"Background: {desc}...")
            break

    summary = " | ".join(parts)
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."

    return summary


def get_facet_question_range(facet: str) -> tuple:
    """
    Get the question number range for a facet.

    Args:
        facet: Facet key name

    Returns:
        Tuple of (start_q, end_q) question numbers
    """
    ranges = {
        'demographics': (1, 13),
        'sociodemographic': (14, 50),
        'values': (51, 72),
        'personality': (73, 102),
        'behavioral': (103, 115),
        'identity': (116, 125),
        'professional': (126, 135),
        'creativity': (136, 141)
    }
    return ranges.get(facet.lower(), (0, 0))


def validate_persona_input(persona: Dict) -> List[str]:
    """
    Validate persona input and return any warnings.

    Args:
        persona: Persona dictionary

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    if not persona:
        warnings.append("Persona is empty")
        return warnings

    if not any(v for v in persona.values() if v):
        warnings.append("All persona fields are empty")

    # Check for minimal content
    total_content = sum(len(str(v)) for v in persona.values() if v)
    if total_content < 50:
        warnings.append("Persona has very little content (< 50 characters)")

    return warnings


class ResponseParser:
    """Parser for LLM responses to questionnaire format."""

    @staticmethod
    def parse_qa_format(response: str) -> Dict[str, str]:
        """
        Parse Q1: answer format responses.

        Args:
            response: LLM response text

        Returns:
            Dictionary mapping question numbers to answers
        """
        return {f"Q{q_num}": answer for q_num, answer in _extract_qa_items(response)}

    @staticmethod
    def parse_numbered_format(response: str) -> Dict[str, str]:
        """
        Parse numbered list format (1. answer, 2. answer).

        Args:
            response: LLM response text

        Returns:
            Dictionary mapping question numbers to answers
        """
        if not response:
            return {}
        return {
            f"Q{int(q_num)}": answer.strip()
            for q_num, answer in _NUMBERED_LINE_RE.findall(response)
        }

    @staticmethod
    def auto_parse(response: str) -> Dict[str, str]:
        """
        Automatically detect and parse response format.

        Args:
            response: LLM response text

        Returns:
            Dictionary mapping question numbers to answers
        """
        if not response:
            return {}

        # Try Q format first
        if re.search(r'(?:^|\b)(?:\*\*)?(?:Q|Question\s+)\d+(?:\*\*)?\s*:', response, re.MULTILINE | re.IGNORECASE):
            return ResponseParser.parse_qa_format(response)

        # Try numbered format
        if re.search(r'^ {0,3}\d+\.\s', response, re.MULTILINE):
            return ResponseParser.parse_numbered_format(response)

        # Return empty if no pattern matched
        return {}
