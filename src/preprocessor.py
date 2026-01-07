# [google에서 id만든거+레이트청킹과 비교군인 일반적인 청킹 실험] ID 부여 및 텍스트 청킹(Span 인덱스 계산) 로직

# src/preprocessor.py

"""
텍스트 청킹(Span 인덱스 계산) 로직. 현재 나의 임베딩 단위는 column이라 레이트 청킹을 안함. 
하지만, 만약에 리뷰가 문단 단위로 길이가 길어지면 사용할 것.

- 레이트 청킹 vs 그냥 텍스트 청킹 실험을 위해,
  나중에 이 파일에 sentence-level / token-level chunking 함수들을 분리해 둘 예정.

예시 아이디어:
- chunk_by_sentences(text, n_sentences, tokenizer) -> List[(start_tok, end_tok)]
- chunk_by_fixed_tokens(text, chunk_size, tokenizer) -> List[(start_tok, end_tok)]
"""

from typing import List, Tuple
from transformers import AutoTokenizer


def identity_chunks(text: str) -> List[str]:
    """
    현재는 청킹 안 하고 전체 텍스트를 그대로 하나의 청크로 쓰는 placeholder.
    나중에 sentence/semantic chunking 실험할 때 교체 예정.
    """
    return [text]


