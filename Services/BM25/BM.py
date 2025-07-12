import os
import joblib
from rank_bm25 import BM25Okapi
from Services.TextProcessingService.TextProcessingServices import cleanTokenizeText
from Services.FilesManagmentService.FilesServices import *
from Services.DatabaseService.DatabaseServices import get_all_document_ids, get_all_documents_contents

class BM25ModelData:
    def __init__(self, dataset_name, firstTime=False):
        self.dataset_name = dataset_name
        self.bm25 = None
        self.tokenized_corpus = None
        self.document_map = None
        # self.doc_ids = get_documents_names_list(dataset_name)  
        
        
        if not firstTime:
            try:
                self.bm25 = getObjectFrom_Joblib_File(dataset_name, "BM25Model")
                self.tokenized_corpus = getObjectFrom_Joblib_File(dataset_name, "BM25Corpus")
                self.document_map = getObjectFrom_Joblib_File(dataset_name, 'docs_map')
                print("✅ تم تحميل نموذج BM25 بنجاح")
            except Exception as e:
                print(f"⚠️ فشل تحميل نموذج BM25: {str(e)} – سيتم تدريبه الآن")
                self.build_model()

    def build_model(self):
        print("🔄 جاري تحميل ومعالجة المستندات لنموذج BM25...")
        raw_documents = get_all_documents_contents(self.dataset_name)
        self.document_map = get_all_document_ids(self.dataset_name)  # 👈 تأكيد الترتيب
        tokenized_corpus = []

        for content in raw_documents:
            try:
                tokens = cleanTokenizeText(content)
                tokenized_corpus.append(tokens)
            except Exception as e:
                print(f"⚠️ خطأ في مستند: {str(e)}")

        if not tokenized_corpus:
            raise ValueError("❌ لا توجد مستندات صالحة لتدريب BM25")

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.tokenized_corpus = tokenized_corpus

        saveObjectTo_Joblib_File(self.bm25, self.dataset_name, "BM25Model.joblib")
        saveObjectTo_Joblib_File(self.tokenized_corpus, self.dataset_name, "BM25Corpus.joblib")

        print("✅ تم تدريب نموذج BM25 وحفظه بنجاح")

    def search(self, query, top_n=5):
        if self.bm25 is None or self.tokenized_corpus is None:
            raise ValueError("❌ النموذج غير محمّل أو غير مُهيأ")

        query_tokens = cleanTokenizeText(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        results = [(i, scores[i]) for i in ranked_indices]
        return results

    def get_document_name(self, index):
        try:
            return self.document_map[index]
        except IndexError:
            return f"doc_{index}"
