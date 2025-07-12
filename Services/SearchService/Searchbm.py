from Services.BM25.BM import BM25ModelData
from Services.FilesManagmentService.FilesServices import get_document_from_db

class SearchEngineBM25Only:

    def __init__(self, dataset_name):
        self.results = []
        self.dataset_name = dataset_name
        self.corrected_query = None

        # تحميل نموذج BM25 فقط
        self.bm25_model = BM25ModelData(dataset_name)

    def search(self, query, top_k=20):
        try:
            print("🔍 بدء البحث باستخدام BM25 فقط...")
            bm25_scores = self.bm25_model.search(query, top_n=top_k)

            self.results = []
            for rank, (index, score) in enumerate(bm25_scores):
                try:
                    doc_name = self.bm25_model.get_document_name(index)
                    print(f"📄 تحميل الوثيقة: {doc_name}")
                    doc = get_document_from_db(self.dataset_name, doc_name)

                    if doc:
                        print("✅ تم تحميل الوثيقة من قاعدة البيانات")
                        self.results.append({
                            'doc_id': index,
                            'doc_name': doc_name,
                            'score': float(score),
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
