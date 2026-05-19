import re
from typing import Optional


def all_tags_properly_closed(text: str) -> int:
    """Check if all tags are properly closed"""
    tag_stack = []
    tag_pattern = re.finditer(r"</?(think|answer)>", text, re.IGNORECASE)
    
    for tag in tag_pattern:
        tag_text = tag.group()
        tag_type = re.match(r"</?(think|answer)>", tag_text, re.IGNORECASE).group(1)
        
        if tag_text.startswith("</"):
            if not tag_stack or tag_stack[-1] != tag_type:
                return 0
            tag_stack.pop()
        else:
            tag_stack.append(tag_type)
    
    return 1 if not tag_stack else 0

def ends_with_answer(text: str) -> int:
    """Check if text ends with answer tag"""
    return 1 if text.strip().endswith("</answer>") else 0

def starts_with_think(text: str) -> int:
    """Check if text starts with think tag"""
    return 1 if text.strip().startswith("<think>") else 0

def extract_binary_answer(completion: str) -> Optional[bool]:
    """Extract binary yes/no answer from completion text"""
    # Look for answer tags first
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip().lower()
        if "yes" in answer_text:
            return True
        elif "no" in answer_text:
            return False
    
    # Fallback: look for yes/no in the entire text
    completion_lower = completion.lower()
    if "yes" in completion_lower and "no" not in completion_lower:
        return True
    elif "no" in completion_lower and "yes" not in completion_lower:
        return False
    
    return None

def extract_dual_answer(text: str, gene_m: str) -> float:
    """Tiered dual reward mirroring the primal extract_binary_answer fallback logic."""
    gene_m = gene_m.upper()
    # Tier 1: gene inside proper <answer>...</answer> tag
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match and gene_m in answer_match.group(1).strip().upper():
        return 1.0
    # Tier 2: gene appears after last </think> (correct position, tag missing or unclosed)
    last_think = text.upper().rfind('</THINK>')
    if last_think != -1 and gene_m in text[last_think:].upper():
        return 0.5
    # Tier 3: gene mentioned anywhere in reasoning
    if gene_m in text.upper():
        return 0.2
    return 0.0

def extract_think(completion: str) -> str:
    """Extract content from think tags"""
    think_matches = re.findall(r"<think>(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
    return " ".join(think_matches)

def has_at_least_one_think(text: str) -> int:
    """Check if text has at least one think tag"""
    return 1 if re.search(r"<think>", text, re.IGNORECASE) else 0

def has_any_tag(text: str) -> int:
    """Check if text has any tags"""
    return 1 if re.search(r"<(think|answer)>", text, re.IGNORECASE) else 0


def low_untagged_ratio(text: str) -> float:
    """Calculate ratio of tagged vs untagged words"""
    text_no_tags = re.sub(r"</?(think|answer)>", "", text)
    total_words = len(re.findall(r"\b\w+\b", text_no_tags))
    
    tagged_words = 0
    for tag in re.findall(r"<(think|answer)>(.*?)</\1>", text, re.DOTALL):
        tagged_words += len(re.findall(r"\b\w+\b", tag[1]))
    
    ratio = tagged_words / total_words if total_words else 0
    return ratio

def is_not_too_long(text: str, max_words: int = 200) -> int:
    """Check if the answer portion (outside <think> blocks) is not too long."""
    answer_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    word_count = len(re.findall(r"\b\w+\b", answer_text))
    return 1 if word_count <= max_words else 0


def has_one_answer(text: str) -> int:
    """Check if text has exactly one answer tag"""
    answer_count = len(re.findall(r"<answer>", text, re.IGNORECASE))
    return 1 if answer_count == 1 else 0

def answer_after_thinks(text: str) -> int:
    """Check if answer comes after think tags"""
    think_tags = list(re.finditer(r"</think>", text, re.IGNORECASE))
    answer_match = re.search(r"<answer>", text, re.IGNORECASE)
    
    if not answer_match or not think_tags:
        return 0
    
    last_think_end = think_tags[-1].end()
    return 1 if answer_match.start() > last_think_end else 0

def thinks_have_text(text: str) -> int:
    """Check if think tags contain meaningful text"""
    think_contents = extract_think(text)
    return 1 if len(think_contents.strip()) > 10 else 0

def no_nested_tags(text: str) -> int:
    """Check for no nested tags"""
    blocks = re.finditer(r"<(think|answer)>(.*?)</\1>", text, re.DOTALL | re.IGNORECASE)
    
    for block in blocks:
        inner_text = block.group(2)
        if re.search(r"</?(think|answer)>", inner_text, re.DOTALL | re.IGNORECASE):
            return 0
    return 1

def has_limited_thinks(text: str, max_thinks: int = 3) -> int:
    """Check if text has limited number of think tags"""
    think_count = len(re.findall(r"<think>", text, re.IGNORECASE))
    return 1 if think_count <= max_thinks else 0



