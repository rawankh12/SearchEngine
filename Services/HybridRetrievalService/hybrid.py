import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Services.BM25.BM import BM25ModelData
from Services.DataRepresentationService.VectorSpaceModelData import VectorSpaceModelData
from Services.TextProcessingService.TextProcessingServices import cleanTextAlgorithm

class HybridBM25TFIDF:
    def __init__(self, dataset_name, weights=None):
        self.dataset_name = dataset_name
        self.bm25_model = BM25ModelData(dataset_name)
        self.tfidf_model = VectorSpaceModelData(dataset_name)

        # الأوزان الافتراضية (BM25 و TF-IDF فقط)
        self.weights = weights or {"bm25": 0.6, "tfidf": 0.4}

    @staticmethod
    def _normalize(arr: np.ndarray):
        mx, mn = arr.max(), arr.min()
        return arr if mx == mn else (arr - mn) / (mx - mn)

    def get_top_k_documents(self, query, top_k=20):
        # ----------- 1) BM25 -----------
        bm25_pairs = self.bm25_model.search(query, top_n=None)
        bm25_dict = {doc_id: score for doc_id, score in bm25_pairs}
        bm25_norm = self._normalize(np.array(list(bm25_dict.values())))
        for (doc_id, _), norm_score in zip(bm25_dict.items(), bm25_norm):
            bm25_dict[doc_id] = norm_score

        # ----------- 2) TF-IDF -----------
        cleaned_query = cleanTextAlgorithm(query)
        query_vector = self.tfidf_model.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, self.tfidf_model.vector_space_model).flatten()

        doc_ids_map = self.tfidf_model.getDocumentMap()
        tfidf_scores_dict = {doc_ids_map[i]: score for i, score in enumerate(similarities)}
        tfidf_norm = self._normalize(np.array(list(tfidf_scores_dict.values())))
        for (doc_id, _), norm_score in zip(tfidf_scores_dict.items(), tfidf_norm):
            tfidf_scores_dict[doc_id] = norm_score

        # ----------- 3) Fusion -----------
        fused_scores = {}
        all_doc_ids = set(bm25_dict.keys()) | set(tfidf_scores_dict.keys())
        for doc_id in all_doc_ids:
            fused_scores[doc_id] = (
                self.weights["bm25"]  * bm25_dict.get(doc_id, 0.0) +
                self.weights["tfidf"] * tfidf_scores_dict.get(doc_id, 0.0)
            )

        # ----------- 4) Ranking -----------
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_doc_ids, scores = zip(*ranked) if ranked else ([], [])
        return list(top_doc_ids), np.array(scores)