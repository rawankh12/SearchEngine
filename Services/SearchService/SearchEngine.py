from Services.DatabaseService.DatabaseServices import get_document_from_db
from Services.HybridRetrievalService.hybrid import HybridBM25TFIDF  # تأكد من المسار الصحيح

class SearchEngineHybrid:
    def __init__(self, dataset_name, weights=None):
        self.dataset_name = dataset_name
        self.corrected_query = None
        self.results = []

        # تحميل النموذج الهجين (BM25 + TF-IDF)
        print("⚙️ تحميل نموذج الهجين (BM25 + TF-IDF)...")
        self.hybrid_model = HybridBM25TFIDF(dataset_name, weights=weights)
        print("✅ تم تحميل النموذج بنجاح")

    def search(self, query, top_k=20):
        try:
            print("🔍 بدء البحث باستخدام النموذج الهجين...")

            # 1) احصل على أفضل الوثائق + الدرجات
            top_doc_ids, fused_scores = self.hybrid_model.get_top_k_documents(query, top_k=top_k)
            print(f"✅ تم استخراج {len(top_doc_ids)} وثيقة بأعلى درجات التشابه.")

            self.results = []
            for rank, (doc_id, score) in enumerate(zip(top_doc_ids, fused_scores), start=1):
                try:
                    doc_name = str(doc_id)
                    print(f"\n📄 [المرتبة {rank}] محاولة تحميل الوثيقة: {doc_name}")

                    doc = get_document_from_db(self.dataset_name, doc_name)

                    if doc:
                        print("✅ تم تحميل الوثيقة من قاعدة البيانات")
                        self.results.append({
                            'rank': rank,
                            'doc_id': doc_id,
                            'doc_name': doc_name,
                            'score': float(score),
                            'content': doc.get('text', '')[:500]  # عرض أول 500 حرف
                        })
                    else:
                        print("⚠️ الوثيقة غير موجودة في قاعدة البيانات")

                except Exception as inner_e:
                    print(f"⚠️ خطأ أثناء تحميل الوثيقة {doc_id}: {inner_e}")

            return self.results

        except Exception as e:
            print(f"❌ خطأ أثناء تنفيذ البحث: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
