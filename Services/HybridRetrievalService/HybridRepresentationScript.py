import numpy as np
from Services.BERTServices.BertEmbedding import BERTRepresentation
from Services.BM25.BM import BM25ModelData
from Services.DataRepresentationService.VectorSpaceModelData import VectorSpaceModelData
from Services.TextProcessingService.TextProcessingServices import cleanTextAlgorithm

class HybridBERTBM25TFIDF:
    def __init__(self, dataset_name, weights=None):
        self.dataset_name = dataset_name
        self.bert_model = BERTRepresentation(dataset_name)
        self.bm25_model = BM25ModelData(dataset_name)
        self.tfidf_model = VectorSpaceModelData(dataset_name)

        # الأوزان الافتراضية (يمكن تعديلها حسب الحاجة)
        self.weights = weights or {"bert": 0.6, "bm25": 0.2, "tfidf": 0.2}

    @staticmethod
    def _normalize(arr: np.ndarray):
        mx, mn = arr.max(), arr.min()
        return arr if mx == mn else (arr - mn) / (mx - mn)

    def get_top_k_documents(self, query, top_k=20):
        # ----------- 1) BERT -----------
        q_vec = self.bert_model.convertQueryToVector(query)
        bert_ids, bert_scores = self.bert_model.getSimilarityScores(q_vec, top_k=None)
        bert_dict = {doc_id: score for doc_id, score in zip(bert_ids, bert_scores)}
        bert_norm = self._normalize(np.array(list(bert_dict.values())))
        for (doc_id, _), norm_score in zip(bert_dict.items(), bert_norm):
            bert_dict[doc_id] = norm_score

        # ----------- 2) BM25 -----------
        bm25_pairs = self.bm25_model.search(query, top_n=None)
        bm25_dict = {doc_id: score for doc_id, score in bm25_pairs}
        bm25_norm = self._normalize(np.array(list(bm25_dict.values())))
        for (doc_id, _), norm_score in zip(bm25_dict.items(), bm25_norm):
            bm25_dict[doc_id] = norm_score

        # ----------- 3) TF-IDF -----------
        cleaned_query = cleanTextAlgorithm(query)
        query_vector = self.tfidf_model.vectorizer.transform([cleaned_query])

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vector, self.tfidf_model.vector_space_model).flatten()

        tfidf_scores_dict = {i: score for i, score in enumerate(similarities)}
        tfidf_norm = self._normalize(np.array(list(tfidf_scores_dict.values())))
        for (doc_id, _), norm_score in zip(tfidf_scores_dict.items(), tfidf_norm):
          tfidf_scores_dict[doc_id] = norm_score

        # ----------- 4) Fusion -----------
        fused_scores = {}
        for doc_id in set(bert_dict.keys()) | set(bm25_dict.keys()) | set(tfidf_scores_dict.keys()):
            fused_scores[doc_id] = (
                self.weights["bert"]  * bert_dict.get(doc_id, 0.0) +
                self.weights["bm25"]  * bm25_dict.get(doc_id, 0.0) +
                self.weights["tfidf"] * tfidf_scores_dict.get(doc_id, 0.0)
            )

        # ----------- 5) Ranking -----------
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_doc_ids, scores = zip(*ranked)
        return list(top_doc_ids), np.array(scores)
