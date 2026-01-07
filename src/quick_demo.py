from aggregator import aggregate_to_providers

def quick_test():
    # hybrid_search 결과라고 가정한 가짜 passage_results
    fake_passage_results = [
        {
            "venue_id": "H01",
            "hall_name": "메리엘홀",
            "aspect": "food",
            "text_chunk": "음식이 정말 맛있고 양이 많았어요.",
            "dense_score": 0.9,
            "sparse_score": 1.5,
            "rrf_score": 1.3,
        },
        {
            "venue_id": "H01",
            "hall_name": "메리엘홀",
            "aspect": "mood",
            "text_chunk": "조명이 어둡고 분위기가 고급스러웠어요.",
            "dense_score": 0.85,
            "sparse_score": 1.2,
            "rrf_score": 1.1,
        },
        {
            "venue_id": "H02",
            "hall_name": "라마다홀",
            "aspect": "access",
            "text_chunk": "역에서 거리가 조금 멀어요.",
            "dense_score": 0.8,
            "sparse_score": 1.8,
            "rrf_score": 1.4,
        },
    ]

    provider_results = aggregate_to_providers(fake_passage_results, top_n_passages=2)

    print("\n🏆 Provider ranking (FAKE DATA):")
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
                f"(RRF={ev['rrf_score']:.4f}, dense={ev['dense_score']:.4f}, sparse={ev['sparse_score']:.4f}) "
                f"{snippet}"
            )

if __name__ == "__main__":
    quick_test()
