import numpy as np
from Services.BERTServices.BertEmbedding import BERTRepresentation
from Services.BM25.BM import BM25ModelData

class HybridBERTBM25:
    """دمج موازي بين تمثيل BERT ودرجات BM25 بمبدأ Parallel Fusion.

    - يعتمد على تمثيلات BERT الجاهزة (self.bert_model).
    - يعتمد على درجات BM25 الجاهزة (self.bm25_model).
    - يطبّق تطبيعًا Min‑Max ثم يدمج بالوزن المحدّد.
    """

    def __init__(self, dataset_name, weights=None):
        self.dataset_name = dataset_name
        self.bert_model  = BERTRepresentation(dataset_name)
        self.bm25_model  = BM25ModelData(dataset_name)

        # أوزان الدمج الافتراضية
        self.weights = weights or {"bert": 0.7, "bm25": 0.3}

        # عدد الوثائق الكلي (الأصغر بين النموذجين احتياطًا)
        self.total_docs = min(len(self.bert_model.documents_vectors),
                              len(self.bm25_model.tokenized_corpus))

    # ----------------- utils -----------------
    @staticmethod
    def _normalize(arr: np.ndarray):
        mx, mn = arr.max(), arr.min()
        return arr if mx == mn else (arr - mn) / (mx - mn)

    # ----------------- main ------------------
    def get_top_k_documents(self, query, top_k=20):
    # --- 1) BERT ---
     q_vec = self.bert_model.convertQueryToVector(query)
     bert_ids, bert_scores = self.bert_model.getSimilarityScores(q_vec, top_k=None)
     bert_dict = {doc_id: score for doc_id, score in zip(bert_ids, bert_scores)}
     bert_norm = self._normalize(np.array(list(bert_dict.values())))
     for (k, _), n in zip(bert_dict.items(), bert_norm):
        bert_dict[k] = n

    # --- 2) BM25 ---
     bm25_pairs = self.bm25_model.search(query, top_n=None)  # [(doc_id, score), ...]
     bm25_dict = {doc: score for doc, score in bm25_pairs}
     bm25_norm = self._normalize(np.array(list(bm25_dict.values())))
     for (k, _), n in zip(bm25_dict.items(), bm25_norm):
        bm25_dict[k] = n

    # --- 3) دمج ---
     fused_dict = {}
     for doc, score in bert_dict.items():
        fused_dict[doc] = self.weights["bert"] * score
     for doc, score in bm25_dict.items():
        fused_dict[doc] = fused_dict.get(doc, 0) + self.weights["bm25"] * score

    # --- 4) ترتيب ---
     ranked = sorted(fused_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
     top_indices, fused_scores = zip(*ranked)

     return list(top_indices), np.array(fused_scores)
