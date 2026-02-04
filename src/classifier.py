"""
Three-Way Co-occurrence Classifier for Financial Stress
=======================================================
Implements Baker et al. (2019) methodology for classifying
social media posts as indicating financial stress.

A post is classified as "stress" if it contains at least one term
from EACH of the three categories:
1. Financial terms
2. Stress terms
3. Negative terms
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from unidecode import unidecode

from .dictionaries import DICTIONARIES
from .preprocessing import preprocess_text, extract_tokens_for_matching


@dataclass
class ClassificationResult:
    """Result of classifying a single text."""
    is_financial: bool
    is_stress_post: bool  # Three-way co-occurrence
    financial_matches: List[str]
    stress_matches: List[str]
    negative_matches: List[str]

    @property
    def match_counts(self) -> Dict[str, int]:
        return {
            'financial': len(self.financial_matches),
            'stress': len(self.stress_matches),
            'negative': len(self.negative_matches),
        }


class StressClassifier:
    """
    Classifier for financial stress based on dictionary matching.

    Uses three-way co-occurrence: a post must contain at least one term
    from each category (financial, stress, negative) to be classified
    as a stress post.
    """

    def __init__(
        self,
        financial_dict: Optional[Set[str]] = None,
        stress_dict: Optional[Set[str]] = None,
        negative_dict: Optional[Set[str]] = None,
        use_accent_normalization: bool = True,
    ):
        """
        Initialize classifier with dictionaries.

        Args:
            financial_dict: Set of financial terms (uses default if None)
            stress_dict: Set of stress terms (uses default if None)
            negative_dict: Set of negative terms (uses default if None)
            use_accent_normalization: Also match accent-normalized versions
        """
        self.financial_dict = financial_dict or DICTIONARIES['financial']
        self.stress_dict = stress_dict or DICTIONARIES['stress']
        self.negative_dict = negative_dict or DICTIONARIES['negative']
        self.use_accent_normalization = use_accent_normalization

        # Create normalized versions of dictionaries for robust matching
        if use_accent_normalization:
            self.financial_dict_normalized = {
                unidecode(t) for t in self.financial_dict
            }
            self.stress_dict_normalized = {
                unidecode(t) for t in self.stress_dict
            }
            self.negative_dict_normalized = {
                unidecode(t) for t in self.negative_dict
            }

    def _find_matches(
        self,
        tokens: Set[str],
        dictionary: Set[str],
        normalized_dict: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Find matching terms between tokens and dictionary.

        Uses both exact matching and partial matching for multi-word terms.

        Args:
            tokens: Set of tokens from text
            dictionary: Set of dictionary terms
            normalized_dict: Accent-normalized dictionary for fallback matching

        Returns:
            List of matched terms
        """
        matches = []
        text_str = ' '.join(tokens)  # Reconstruct for multi-word matching

        for term in dictionary:
            # Check for term in tokens (single word) or text (multi-word)
            if ' ' in term:
                # Multi-word term: check if present in text
                if term in text_str:
                    matches.append(term)
            else:
                # Single word term: check if in tokens
                if term in tokens:
                    matches.append(term)

        # Also check normalized versions if enabled
        if self.use_accent_normalization and normalized_dict:
            text_normalized = unidecode(text_str)
            tokens_normalized = {unidecode(t) for t in tokens}

            for term in normalized_dict:
                if term not in [unidecode(m) for m in matches]:  # Avoid duplicates
                    if ' ' in term:
                        if term in text_normalized:
                            matches.append(term)
                    else:
                        if term in tokens_normalized:
                            matches.append(term)

        return list(set(matches))  # Remove duplicates

    def classify(self, text: str, preprocess: bool = True) -> ClassificationResult:
        """
        Classify a single text for financial stress.

        Args:
            text: Text to classify
            preprocess: Whether to preprocess text first

        Returns:
            ClassificationResult with detailed matching information
        """
        # Preprocess if needed
        if preprocess:
            processed = preprocess_text(text)
        else:
            processed = text.lower()

        # Extract tokens
        tokens = extract_tokens_for_matching(processed)

        # Find matches in each category
        financial_matches = self._find_matches(
            tokens,
            self.financial_dict,
            self.financial_dict_normalized if self.use_accent_normalization else None
        )

        stress_matches = self._find_matches(
            tokens,
            self.stress_dict,
            self.stress_dict_normalized if self.use_accent_normalization else None
        )

        negative_matches = self._find_matches(
            tokens,
            self.negative_dict,
            self.negative_dict_normalized if self.use_accent_normalization else None
        )

        # Determine classification
        is_financial = len(financial_matches) > 0
        is_stress_post = (
            len(financial_matches) > 0 and
            len(stress_matches) > 0 and
            len(negative_matches) > 0
        )

        return ClassificationResult(
            is_financial=is_financial,
            is_stress_post=is_stress_post,
            financial_matches=financial_matches,
            stress_matches=stress_matches,
            negative_matches=negative_matches,
        )

    def classify_batch(
        self,
        texts: List[str],
        preprocess: bool = True
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify
            preprocess: Whether to preprocess texts

        Returns:
            List of ClassificationResults
        """
        return [self.classify(text, preprocess) for text in texts]

    def get_classification_stats(
        self,
        results: List[ClassificationResult]
    ) -> Dict:
        """
        Calculate statistics from classification results.

        Args:
            results: List of ClassificationResults

        Returns:
            Dictionary with classification statistics
        """
        total = len(results)
        if total == 0:
            return {'total': 0, 'financial': 0, 'stress': 0}

        financial_count = sum(1 for r in results if r.is_financial)
        stress_count = sum(1 for r in results if r.is_stress_post)

        return {
            'total': total,
            'financial': financial_count,
            'financial_pct': financial_count / total * 100,
            'stress': stress_count,
            'stress_pct': stress_count / total * 100,
            'stress_given_financial': (
                stress_count / financial_count * 100
                if financial_count > 0 else 0
            ),
        }


def classify_stress(
    text: str,
    financial_dict: Optional[Set[str]] = None,
    stress_dict: Optional[Set[str]] = None,
    negative_dict: Optional[Set[str]] = None,
) -> bool:
    """
    Convenience function to check if text indicates financial stress.

    Args:
        text: Text to classify
        financial_dict: Optional custom financial dictionary
        stress_dict: Optional custom stress dictionary
        negative_dict: Optional custom negative dictionary

    Returns:
        True if text matches all three categories (financial stress)
    """
    classifier = StressClassifier(
        financial_dict=financial_dict,
        stress_dict=stress_dict,
        negative_dict=negative_dict,
    )
    result = classifier.classify(text)
    return result.is_stress_post


if __name__ == '__main__':
    # Test the classifier
    classifier = StressClassifier()

    test_cases = [
        # Should be classified as stress (all three categories)
        "A crise no mercado financeiro está gerando pânico entre investidores após queda brutal",
        "Ibovespa despenca com temor de recessão e prejuízos bilionários",
        "Risco de colapso bancário assusta investidores com perdas recordes",

        # Should NOT be stress (missing categories)
        "Mercado de ações fecha em alta com otimismo dos investidores",  # Missing stress/negative
        "A crise política continua gerando instabilidade",  # Missing financial
        "Bolsa de valores tem dia de pregão normal",  # Missing stress/negative
        "Preocupação com queda das temperaturas no sul",  # Missing financial
    ]

    print("Classification Examples:")
    print("=" * 70)

    for text in test_cases:
        result = classifier.classify(text)
        status = "✓ STRESS" if result.is_stress_post else "✗ NOT STRESS"
        print(f"\n{status}")
        print(f"Text: {text[:60]}...")
        print(f"  Financial ({len(result.financial_matches)}): {result.financial_matches[:3]}")
        print(f"  Stress ({len(result.stress_matches)}): {result.stress_matches[:3]}")
        print(f"  Negative ({len(result.negative_matches)}): {result.negative_matches[:3]}")
