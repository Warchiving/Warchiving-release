# 업체별 점수 합산 및 리스크 패널티
# src/aggregator.py

"""
업체별 점수 합산 로직 (reranker 이후 단계)


- passage-level 점수들을 venue_id 기준으로 모아서
  provider-level score를 계산
  
[원칙]
- provider ranking은 rerank_score 기준
- rrf_score는 explanation(왜 이 문장이 후보였는지)용  

"""
from typing import List, Dict, Any
from collections import defaultdict


def aggregate_to_providers(
    passage_results: List[Dict[str, Any]],
    top_n_passages: int = 5,
):
    """
    passage_results: hybrid_search 결과 리스트
      - 각 원소: {
          "venue_id", "hall_name", "aspect",
          "text_chunk", "rrf_score", ...
        }

    * 각 원소는 반드시 rerank_score를 포함해야 함.
    
    1) venue_id 기준으로 passage들을 묶고
    2) 각 업체 안에서 상위 N개 passage만 사용해 평균 score 계산
    3) 업체 리스트를 최종 score로 정렬해 반환
    
    """
    provider_passages = defaultdict(list)

    for p in passage_results:
        provider_passages[p["venue_id"]].append(p)

    aggregated = []
    for venue_id, passages in provider_passages.items():
        # RRF score 기준 정렬
        passages_sorted = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)
        top_passages = passages_sorted[:top_n_passages]

        avg_score = sum(p["rerank_score"] for p in top_passages) / len(top_passages)

        aggregated.append(
            {
                "venue_id": venue_id,
                "hall_name": top_passages[0]["hall_name"],
                "score": avg_score,
                "evidences": top_passages,
            }
        )

    aggregated_sorted = sorted(aggregated, key=lambda x: x["score"], reverse=True)
    return aggregated_sorted
