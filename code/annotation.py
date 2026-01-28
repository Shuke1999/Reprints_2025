"""Annotation logic for reuse propagation analysis.

This module provides functions to classify reuse patterns:
- Deletion: text is shortened in reuse
- Concatenation: multiple source blocks are combined
- Paraphrase: text is rephrased (placeholder for future implementation)
"""

from typing import Literal

AnnotationType = Literal["deletion", "Concatenation", "paraphrase", "other"]


def classify_deletion(
    src_piece_length: int | None,
    dst_piece_length: int | None,
    threshold: float = 0.85,
) -> bool:
    """Check if reuse is a deletion (text shortened).
    
    Args:
        src_piece_length: Length of source text piece
        dst_piece_length: Length of destination text piece
        threshold: Ratio threshold (default 0.85, meaning dst < 85% of src)
    
    Returns:
        True if deletion, False otherwise
    """
    if src_piece_length is None or dst_piece_length is None:
        return False
    if src_piece_length == 0:
        return False
    return dst_piece_length < src_piece_length * threshold


def classify_concatenation(
    fragment_count: int | None,
    min_fragments: int = 2,
) -> bool:
    """Check if reuse is a concatenation (multiple fragments combined).
    
    Args:
        fragment_count: Number of fragments in the destination block
        min_fragments: Minimum fragments to consider as concatenation (default 2)
    
    Returns:
        True if concatenation, False otherwise
    """
    if fragment_count is None:
        return False
    return fragment_count >= min_fragments


def classify_paraphrase(
    intersection_len: int | None,
    min_block_length: int | None,
    similarity_threshold: float = 0.7,
) -> bool:
    """Check if reuse is a paraphrase (text rephrased).
    
    Note: This is a placeholder implementation. A more sophisticated approach
    would use text similarity metrics.
    
    Args:
        intersection_len: Length of intersection between source and destination
        min_block_length: Minimum block length
        similarity_threshold: Similarity threshold (default 0.7)
    
    Returns:
        True if likely paraphrase, False otherwise
    """
    if intersection_len is None or min_block_length is None:
        return False
    if min_block_length == 0:
        return False
    # Simple heuristic: low intersection ratio might indicate paraphrasing
    # This is a placeholder - should be replaced with proper text similarity
    similarity = intersection_len / min_block_length
    return similarity < similarity_threshold


def annotate_reuse(
    src_piece_length: int | None = None,
    dst_piece_length: int | None = None,
    fragment_count: int | None = None,
    intersection_len: int | None = None,
    min_block_length: int | None = None,
    deletion_threshold: float = 0.85,
    paraphrase_threshold: float = 0.7,
) -> AnnotationType:
    """Annotate a reuse block with its pattern type.
    
    Priority order:
    1. Concatenation (if fragment_count >= 2)
    2. Deletion (if dst < src * threshold)
    3. Paraphrase (if similarity is low)
    4. Other (default)
    
    Args:
        src_piece_length: Length of source text piece
        dst_piece_length: Length of destination text piece
        fragment_count: Number of fragments in destination
        intersection_len: Length of intersection
        min_block_length: Minimum block length
        deletion_threshold: Threshold for deletion detection
        paraphrase_threshold: Threshold for paraphrase detection
    
    Returns:
        Annotation type string
    """
    # Priority 1: Concatenation
    if classify_concatenation(fragment_count):
        return "Concatenation"
    
    # Priority 2: Deletion
    if classify_deletion(src_piece_length, dst_piece_length, deletion_threshold):
        return "deletion"
    
    # Priority 3: Paraphrase (placeholder)
    if classify_paraphrase(intersection_len, min_block_length, paraphrase_threshold):
        return "paraphrase"
    
    # Default: Other
    return "other"
