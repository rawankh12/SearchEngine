from Services.BERTServices.BertEmbedding import BERTRepresentation
from Services.DatabaseService.DatabaseServices import get_document_from_db

class SearchEngineBERTOnly:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.bert_model = BERTRepresentation(dataset_name)
        self.results = []
        self.corrected_query = None

    def search(self, query, top_k=20):
        print("🔍 بدء البحث باستخدام BERT فقط...")
        query_vector = self.bert_model.convertQueryToVector(query)
        top_indices, scores = self.bert_model.getSimilarityScores(query_vector, top_k=top_k)

        self.results = []
        for rank, index in enumerate(top_indices):
            try:
                doc_name = f"{index}"  # أو حسب الطريقة المستخدمة لديك في تسمية الوثائق
                doc = get_document_from_db(self.dataset_name, doc_name)

                if doc:
                    print(f"✅ تم تحميل الوثيقة {doc_name}")
                    self.results.append({
                        'doc_id': index,
                        'doc_name': doc_name,
                        'score': float(scores[index]),
                        'content': doc.get("text", "")
                    })
                else:
                    print(f"⚠️ الوثيقة {doc_name} غير موجودة في قاعدة البيانات")
            except Exception as e:
                print(f"❌ خطأ في تحميل الوثيقة رقم {index}: {str(e)}")

        return self.results
