
from Services.DatabaseService.DatabaseServices import get_document_from_db
from Services.FilesManagmentService.FilesServices import getObjectFrom_Joblib_File , get_document_Joblib_File , get_all_documents_from_joblib_folder
from Services.HybridRetrievalService.Hybridbmbert import HybridBERTBM25

class SearchEngineBERTBM25Hybrid:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.hybrid_model = HybridBERTBM25(dataset_name)
        self.results = []
        self.corrected_query = None  # للاستخدام لاحقًا إن أضفت تصحيح الكويري

    def search(self, query, top_k=20):
        print("🔍 بدء البحث باستخدام النموذج الهجين BERT + BM25 ...")

        top_indices, scores = self.hybrid_model.get_top_k_documents(query, top_k=top_k)

        self.results = []
        for rank, doc_id in enumerate(top_indices):
            try:
                doc_name = self.hybrid_model.bert_model.document_names[doc_id]
                doc = get_document_Joblib_File(self.dataset_name, doc_name)

                if doc:
                    self.results.append({
                        'rank': rank + 1,
                        'doc_id': doc_id,
                        'doc_name': doc_name,
                        'score': float(scores[rank]),
                        'content': getattr(doc, "text", "")
                    })
                else:
                    print(f"⚠️ الوثيقة {doc_name} غير موجودة في قاعدة البيانات")
            except Exception as e:
                print(f"❌ خطأ في تحميل الوثيقة رقم {doc_id}: {str(e)}")

        return self.results
    
    
