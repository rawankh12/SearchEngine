from Services.WordEmbeddingServices.WordEmbeddingModelData import WordEmbeddingModelData
from Services.FilesManagmentService.FilesServices import get_document_from_db
from Services.TextProcessingService.TextProcessingServices import cleanTokenizeText

class SearchEngineWord2VecOnly:

    def __init__(self, dataset_name):
        self.results = []
        self.dataset_name = dataset_name
        self.corrected_query = None

        # تحميل نموذج Word2Vec فقط
        self.word2vec_model = WordEmbeddingModelData(dataset_name)
        
      # --------- أداة مساعدة لتوسيع الاستعلام بالكلمات المشابهة ---------
    def expand_query_with_similars(self, query, topn=3):
        tokens = cleanTokenizeText(query)               # tokenize
        expanded_tokens = list(tokens)                  # ابدأ بالكلمات الأصلية

        for tok in tokens:
            similars = self.word2vec_model.search_similar_words(tok, topn=topn)
            # كل عنصر في similars هو (word, score)
            expanded_tokens.extend([w for w, _ in similars])

        return " ".join(expanded_tokens)
    
    def search(self, query, top_k=20, expand=True, topn_similar=3):
        """
        expand=True  : يفعّل توسيع الاستعلام
        topn_similar : عدد الكلمات المشابهة لكل كلمة في الاستعلام
        """
        try:
            # 1) وسِّع الاستعلام إذا طُلب ذلك
            expanded_query = self.expand_query_with_similars(query, topn=topn_similar) if expand else query
            print(f"📝 الاستعلام الموسّع: [{expanded_query}]")

            # 2) حوِّل الاستعلام إلى متّجه
            query_vec = self.word2vec_model.convertQueryToVector(expanded_query)

            # 3) احسب تشابه الاستعلام مع كل وثيقة
            sims = self.word2vec_model.getSimilarityScores(query_vec)
            top_indices = sims.argsort()[::-1][:top_k]

            # 4) جهّز النتائج للعرض
            results = []
            for idx in top_indices:
                doc_name = self.word2vec_model.get_document_name(idx)   # يحتاج دالة كما أوضحنا
                doc     = get_document_from_db(self.dataset_name, doc_name)

                if doc:        # قد يكون None لو الوثيقة غير مخزَّنة
                    results.append({
                        "doc_id":   idx,
                        "doc_name": doc_name,
                        "score":    float(sims[idx]),
                        "content":  doc.get("text", "")
                    })

            return results

        except Exception as e:
            print(f"❌ خطأ في البحث: {e}")
            return []