# 전체 파이프라인 제어 (Run)

# src/main.py

from .embedder import BGEEmbedder
from .config import RAW_CSV_PATH, PROCESSED_PARQUET_PATH
from .vector_db import load_passage_df, DenseSparseIndex
from .aggregator import aggregate_to_providers
from .reranker import CrossEncoderReranker


def run_build_embeddings():
    embedder = BGEEmbedder()
    embedder.build_vector_parquet(
        input_csv_path=RAW_CSV_PATH,
        output_parquet_path=PROCESSED_PARQUET_PATH,
    )

def run_hybrid_search_example():
    """
    파이프라인:
    1) Dense + Sparse retrieval (recall)
    2) Cross-Encoder reranking (precision)
    3) Provider aggregation (decision)
    """

    # 1) passage-level parquet 로드
    df = load_passage_df(PROCESSED_PARQUET_PATH)

    # 2) 인덱스 초기화
    index = DenseSparseIndex(df)

    # 3) Dense 인덱스 구축
    index.build_dense_index()

    # 4) 쿼리
    query = "강남 근처고 주차할 곳이 많은 곳"

    # 5) Hybrid search (recall 단계)
    passage_results = index.hybrid_search(
        query_text=query,
        top_k_dense=50,
        top_k_sparse=50,
        k_final=50,
    )


    # 6) 업체 단위로 aggregate 
    # [2단계 Retrieval 유닛테스트하려면 주석을 푸세요. 그리고 아래 6과 7번을 주석하세요.]
    # provider_results = aggregate_to_providers(passage_results, top_n_passages=3)

    # 6) Reranker (precision 단계)
    reranker = CrossEncoderReranker()
    reranked_passages = reranker.rerank(
        query=query,
        passages=passage_results,
        top_k=30,
    )

    # 7) Provider aggregation (decision 단계)
    provider_results = aggregate_to_providers(
        reranked_passages,
        top_n_passages=3,
    )

    # 8) 출력
    print("\n==============================")
    print(f"🔍 Query: {query}")
    print("==============================")

    print("\n🏆 Provider ranking:")
    for i, prov in enumerate(provider_results, start=1):
        print(f"\n[{i}위] {prov['hall_name']} (venue_id={prov['venue_id']})")
        print(f"  - score: {prov['score']:.4f}")
        print("  - evidences:")
        for ev in prov["evidences"]:
            snippet = str(ev["text_chunk"]).replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            print(
                f"    * [{ev['aspect']}] "
                f"(rerank={ev['rerank_score']:.4f}, "
                f"RRF={ev['rrf_score']:.4f}) "
                f"{snippet}"
            )

if __name__ == "__main__":
    
    # run_build_embeddings() # 초기 떄 한번만
    run_hybrid_search_example()
    
    # 1) raw데이터 임베딩
    # 2) ChromaDB 적재
    # 3) Hybrid Search
    # 4) Reranker
    # 5) Aggregator
    
    
