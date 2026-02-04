"""
Text Preprocessing for Portuguese Financial Texts
=================================================
Handles Twitter and Reddit posts in Brazilian Portuguese.
"""

import re
import string
from typing import Optional

from unidecode import unidecode


# Portuguese stopwords (common words to filter out)
# Using a manual list to avoid NLTK download dependency at import time
PORTUGUESE_STOPWORDS = {
    'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo',
    'as', 'até', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele',
    'deles', 'depois', 'do', 'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em',
    'entre', 'era', 'eram', 'essa', 'essas', 'esse', 'esses', 'esta', 'estas',
    'este', 'estes', 'eu', 'foi', 'fomos', 'for', 'fora', 'foram', 'forem',
    'formos', 'fosse', 'fossem', 'fui', 'há', 'isso', 'isto', 'já', 'lhe',
    'lhes', 'lo', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha',
    'minhas', 'muito', 'na', 'nas', 'nem', 'no', 'nos', 'nossa', 'nossas',
    'nosso', 'nossos', 'num', 'numa', 'não', 'nós', 'o', 'os', 'ou', 'para',
    'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem',
    'se', 'seja', 'sejam', 'sejamos', 'sem', 'ser', 'será', 'serão', 'seria',
    'seriam', 'seu', 'seus', 'sido', 'só', 'somos', 'sou', 'sua', 'suas',
    'também', 'te', 'tem', 'tendo', 'tenha', 'tenham', 'tenhamos', 'tenho',
    'ter', 'teu', 'teus', 'teve', 'tinha', 'tinham', 'tive', 'tivemos',
    'tiver', 'tivera', 'tiveram', 'tiverem', 'tivermos', 'tivesse', 'tivessem',
    'tu', 'tua', 'tuas', 'um', 'uma', 'umas', 'uns', 'você', 'vocês', 'vos',
    # Additional common words
    'aqui', 'ali', 'assim', 'ainda', 'agora', 'então', 'pois', 'porque',
    'porém', 'portanto', 'contudo', 'todavia', 'entretanto', 'logo', 'onde',
    'aonde', 'lá', 'cá', 'ora', 'pra', 'pro', 'pros', 'pras', 'tá', 'tô',
    'né', 'vc', 'vcs', 'tbm', 'tb', 'pq', 'q', 'rt', 'via'
}


def clean_text(text: str) -> str:
    """
    Basic text cleaning for social media posts.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove user mentions (@user)
    4. Convert hashtags (#crise -> crise)
    5. Remove excessive whitespace

    Args:
        text: Raw text from social media post

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ''

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove user mentions
    text = re.sub(r'@\w+', '', text)

    # Convert hashtags to words (#crise -> crise)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove RT prefix (retweet indicator)
    text = re.sub(r'^rt\s+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_accents(text: str, remove_accents: bool = False) -> str:
    """
    Handle Portuguese accents.

    Args:
        text: Input text
        remove_accents: If True, remove all accents. If False, just normalize.

    Returns:
        Normalized text
    """
    if remove_accents:
        return unidecode(text)
    return text


def remove_punctuation(text: str, keep_chars: str = '') -> str:
    """
    Remove punctuation from text.

    Args:
        text: Input text
        keep_chars: Characters to keep (e.g., '%' for percentages)

    Returns:
        Text with punctuation removed
    """
    # Create translation table
    chars_to_remove = string.punctuation
    for char in keep_chars:
        chars_to_remove = chars_to_remove.replace(char, '')

    translator = str.maketrans('', '', chars_to_remove)
    return text.translate(translator)


def remove_stopwords(text: str, additional_stopwords: Optional[set] = None) -> str:
    """
    Remove Portuguese stopwords from text.

    Args:
        text: Input text
        additional_stopwords: Extra stopwords to remove

    Returns:
        Text with stopwords removed
    """
    stopwords = PORTUGUESE_STOPWORDS.copy()
    if additional_stopwords:
        stopwords.update(additional_stopwords)

    words = text.split()
    filtered_words = [w for w in words if w not in stopwords]

    return ' '.join(filtered_words)


def tokenize(text: str) -> list:
    """
    Simple whitespace tokenization.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    return text.split()


def preprocess_text(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    convert_hashtags: bool = True,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_stops: bool = False,  # Default False - we check dictionary against original words
    remove_accents_flag: bool = False,  # Default False - preserve Portuguese accents
) -> str:
    """
    Full preprocessing pipeline for social media text.

    Args:
        text: Raw text from social media
        remove_urls: Remove URLs
        remove_mentions: Remove @mentions
        convert_hashtags: Convert #hashtag to hashtag
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation
        remove_stops: Remove stopwords
        remove_accents_flag: Remove accents (á -> a)

    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str) or not text.strip():
        return ''

    processed = text

    # Lowercase first (important for matching)
    if lowercase:
        processed = processed.lower()

    # Remove URLs
    if remove_urls:
        processed = re.sub(r'http[s]?://\S+', '', processed)
        processed = re.sub(r'www\.\S+', '', processed)

    # Remove mentions
    if remove_mentions:
        processed = re.sub(r'@\w+', '', processed)

    # Convert hashtags
    if convert_hashtags:
        processed = re.sub(r'#(\w+)', r'\1', processed)

    # Remove RT prefix
    processed = re.sub(r'^rt\s+', '', processed)

    # Handle accents
    if remove_accents_flag:
        processed = unidecode(processed)

    # Remove punctuation
    if remove_punct:
        processed = remove_punctuation(processed)

    # Normalize whitespace
    processed = re.sub(r'\s+', ' ', processed).strip()

    # Remove stopwords (optional, usually not needed for dictionary matching)
    if remove_stops:
        processed = remove_stopwords(processed)

    return processed


def extract_tokens_for_matching(text: str) -> set:
    """
    Extract tokens from text for dictionary matching.
    Creates both original and accent-removed versions for robust matching.

    Args:
        text: Preprocessed text

    Returns:
        Set of tokens (original + normalized)
    """
    if not text:
        return set()

    words = text.split()
    tokens = set(words)

    # Also add accent-removed versions for matching
    tokens.update(unidecode(w) for w in words)

    return tokens


# Convenience function for batch processing
def preprocess_dataframe(df, text_column: str = 'text') -> 'pd.Series':
    """
    Preprocess all texts in a DataFrame column.

    Args:
        df: DataFrame with text data
        text_column: Name of the column containing text

    Returns:
        Series of preprocessed texts
    """
    return df[text_column].apply(preprocess_text)


if __name__ == '__main__':
    # Test preprocessing
    test_texts = [
        "RT @usuario: A #crise no mercado financeiro está gerando muito PÂNICO! https://t.co/abc123",
        "Ibovespa despenca 5% com temor de recessão no Brasil #bolsa #investimentos",
        "URGENTE: Banco Central eleva Selic para conter inflação @BCB_oficial",
        "Ações da Petrobras em queda livre após prejuízo bilionário 📉💸",
    ]

    print("Text Preprocessing Examples:")
    print("=" * 60)

    for text in test_texts:
        processed = preprocess_text(text)
        tokens = extract_tokens_for_matching(processed)
        print(f"\nOriginal: {text}")
        print(f"Processed: {processed}")
        print(f"Tokens ({len(tokens)}): {sorted(tokens)[:10]}...")
