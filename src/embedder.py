# [핵심] BGE-M3 기반 Late Chunking 구현# src/embedder.py

import os
from typing import List

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .config import (
    DENSE_MODEL_NAME,
    ASPECT_COLUMNS,
    VENUE_ID_COL,
    HALL_NAME_COL,
    RAW_CSV_PATH,
    PROCESSED_PARQUET_PATH,
    MAX_LENGTH,
    BATCH_SIZE,
    USE_FP16,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Mac M1/M2
    else:
        return torch.device("cpu")


DEVICE = get_device()


class BGEEmbedder:
    """
    BGE-M3 기반 임베더.
    - Late chunking 관점에서: 토큰 임베딩을 mean pooling 해서 컬럼 단위 벡터로 만듦
    """

    def __init__(self):
        print(f"🧠 Loading dense model: {DENSE_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME)
        self.model = AutoModel.from_pretrained(DENSE_MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()

        self.normalize = True

    def _mean_pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        토큰 임베딩을 attention mask 기준으로 평균내서 문장/청크 벡터로 만드는 함수.
        last_hidden_state: (batch, seq_len, hidden)
        attention_mask: (batch, seq_len)
        return: (batch, hidden)
        """
        # mask 확장 (batch, seq_len, 1) → (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # 마스크된 hidden의 합
        masked_hidden = last_hidden_state * mask
        sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden)
        # 실제 토큰 길이 (패딩 제외)
        lengths = mask.sum(dim=1)  # (batch, hidden) - hidden dimension이지만 같은 값 반복됨
        lengths = torch.clamp(lengths, min=1e-9)
        pooled = sum_hidden / lengths
        return pooled

    def embed_texts(self, texts: List[str]) -> List[list]:
        """
        여러 개의 텍스트를 받아서,
        각 텍스트에 대한 벡터(list[float])를 반환.
        """
        if len(texts) == 0:
            return []

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            if USE_FP16 and DEVICE.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(**enc)
            else:
                outputs = self.model(**enc)

            token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
            sentence_embeddings = self._mean_pooling(token_embeddings, enc["attention_mask"])

            if self.normalize:
                sentence_embeddings = torch.nn.functional.normalize(
                    sentence_embeddings, p=2, dim=1
                )

        return sentence_embeddings.cpu().tolist()

    def build_vector_parquet(
        self,
        input_csv_path: str = RAW_CSV_PATH,
        output_parquet_path: str = PROCESSED_PARQUET_PATH,
    ):
        """
        1) venues.csv 읽기
        2) row × aspect 컬럼마다 텍스트 뽑기
        3) BGE-M3로 임베딩
        4) data/processed/processed.parquet 에 저장
        """

        print(f"📂 Loading CSV from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        records = [] # 나중에 parquet로 저장될 row들. 임베딩 끝난 뒤 vecotr까지 채워서 이후에 벡터 db로 저장
        texts_for_embedding = [] # 임베딩용 순수 텍스트 ex. 강남역에서 도보 3분

        print("🔄 Building (venue_id, aspect) records...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            venue_id = row[VENUE_ID_COL]
            hall_name = row[HALL_NAME_COL]

            for aspect in ASPECT_COLUMNS:
                raw_text = row.get(aspect, "") # 딕셔너리에서 쓰는 함수로, 주어진 key에 대한 value를 반환 row.get(ket, "")

                if pd.isna(raw_text):
                    raw_text = ""

                text = str(raw_text).strip()

                # vector는 나중에 넣을 거라 일단 None
                record = {
                    "venue_id": venue_id,
                    "hall_name": hall_name,
                    "aspect": aspect,
                    "text_chunk": text,
                    "vector": None,
                }

                records.append(record)
                texts_for_embedding.append(text)

        print(f"✅ Total records: {len(records)}")

        # 2) 텍스트들 임베딩
        print("🧠 Embedding text chunks...")
        all_vectors: List[list] = []
        for i in tqdm(range(0, len(texts_for_embedding), BATCH_SIZE)):
            batch_texts = texts_for_embedding[i : i + BATCH_SIZE]
            batch_vecs = self.embed_texts(batch_texts)
            all_vectors.extend(batch_vecs)

        assert len(all_vectors) == len(records), "텍스트 개수와 벡터 개수 불일치!"

        # 3) 벡터를 records에 채워
        print("🧩 Attaching vectors to records...")
        for rec, vec in zip(records, all_vectors):
            rec["vector"] = vec

        processed_df = pd.DataFrame(records)

        # 4) Parquet 저장
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        print(f"💾 Saving Parquet to: {output_parquet_path}")
        processed_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)

        print("🎉 Done: vector parquet ready.")
