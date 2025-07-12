from Services.DataRepresentationService.VectorSpaceModelData import VectorSpaceModelData
from Services.FilesManagmentService.FilesServices import get_document_from_db

class SearchEngineTFIDFOnly:

    def __init__(self, dataset_name):
        self.results = []
        self.dataset_name = dataset_name
        self.corrected_query = None

        # تحميل نموذج TF-IDF فقط
        self.tfidf_model = VectorSpaceModelData(dataset_name)

    def search(self, query, top_k=20):
        try:
            print("🔍 بدء البحث باستخدام TF-IDF فقط...")
            query_vector = self.tfidf_model.convertQueryToVector(query)
            similarity_scores = self.tfidf_model.getSimilarityScores(query_vector)

            # ترتيب النتائج حسب التشابه
            sorted_indices = similarity_scores.argsort()[::-1][:top_k]

            self.results = []
            for index in sorted_indices:
                try:
                    doc_name = self.tfidf_model.get_document_name(index)
                    print(f"📄 تحميل الوثيقة: {doc_name}")
                    doc = get_document_from_db(self.dataset_name, doc_name)

                    if doc:
                        print("✅ تم تحميل الوثيقة من قاعدة البيانات")
                        self.results.append({
                            'doc_id': index,
                            'doc_name': doc_name,
                            'score': float(similarity_scores[index]),
                            'content': doc.get('text', '')
                        })
                    else:
                        print("⚠️ الوثيقة غير موجودة في قاعدة البيانات")
                except Exception as inner_e:
                    print(f"⚠️ خطأ أثناء جلب الوثيقة ذات الفهرس {index}: {inner_e}")

            return self.results

        except Exception as e:
            print(f"❌ خطأ أثناء البحث: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
