 # -*- coding: utf-8 -*-
 
from Services.HybridRetrievalService.HybridRepresentationScript import HybridRepresentationParallel

hybrid = HybridRepresentationParallel(
    dataset_name="beir_quora",
    weights={
        "tfidf": 0.4,
        "word2vec": 0.3,
        "bm25": 0.3 
    }
)

top_indices, scores = hybrid.get_top_k_documents("your query here", top_k=10)

# أسماء الوثائق بالاعتماد على خريطة TF‑IDF
doc_names = [hybrid.tfidf_model.get_document_name(i) for i in top_indices]
for rank, (idx, name) in enumerate(zip(top_indices, doc_names), 1):
    print(f"{rank:2d}. {name}  —  score: {scores[idx]:.4f}")
