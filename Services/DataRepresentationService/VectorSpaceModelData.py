import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Services.FilesManagmentService.FilesServices import saveObjectTo_Joblib_File, getObjectFrom_Joblib_File
from Services.TextProcessingService.TextProcessingServices import cleanTextAlgorithm
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from Services.DatabaseService.DatabaseServices import get_all_documents_contents  # دالة جلب النصوص من DB
from Services.DatabaseService.DatabaseServices import get_all_document_ids
from Services.FilesManagmentService.FilesServices import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



class VectorSpaceModelData:

    def __init__(self, dataset_name, firstTime=False):
        self.dataset_name = dataset_name
        self.vector_space_model = None
        self.document_map = None
        self.vectorizer = None
        

        if not firstTime:
            # تحميل النماذج المخزنة مسبقًا من ملفات joblib
            self.vector_space_model = getObjectFrom_Joblib_File(dataset_name, 'doc_vectorsvsm')
            self.document_map = getObjectFrom_Joblib_File(dataset_name, 'docs_map')
            self.vectorizer = getObjectFrom_Joblib_File(dataset_name, 'Model')

    def buildeVectorSpaceModel(self):
        # جلب محتوى المستندات من قاعدة البيانات مباشرة
        print('جاري تحميل المس تندات من قاعدة البيانات...')
        documents_texts = get_all_documents_contents(self.dataset_name)

        print("\nأول 5 مستندات (نصوص):")
        for i in range(min(5, len(documents_texts))):
            print(f"\n--- مستند {i+1} ---\n", documents_texts[i])

        vectorizer = TfidfVectorizer(
            tokenizer=word_tokenize,
            max_df=0.75,
            preprocessor=cleanTextAlgorithm
        )

        print("\nProcessing, please wait...")

        vectorizer.fit(documents_texts)
        self.vector_space_model = vectorizer.transform(documents_texts)
        self.vectorizer = vectorizer

        # بناء خريطة المستندات من الداتا بيز أو من مصدر خارجي حسب ما هو متوفر
        # هنا سأفترض أن get_all_document_ids ترجع قائمة الـ doc_ids بنفس الترتيب الذي تم جلب النصوص به

        self.document_map = get_all_document_ids(self.dataset_name)

        # حفظ النماذج والخرائط
        saveObjectTo_Joblib_File(self.vector_space_model, self.dataset_name, "doc_vectorsvsm.joblib")
        saveObjectTo_Joblib_File(self.vectorizer, self.dataset_name, "Model.joblib")
        saveObjectTo_Joblib_File(self.document_map, self.dataset_name, "docs_map.joblib")

    def buildDocumentsMap(self):
        # إذا أردت فقط تحديث خريطة الوثائق
        self.document_map = get_all_document_ids(self.dataset_name)
        saveObjectTo_Joblib_File(self.document_map, self.dataset_name, "docs_map.joblib")

    def printVectorSpaceModel(self):
        df = pd.DataFrame(self.vector_space_model.toarray(), columns=self.vectorizer.get_feature_names_out())
        print(df)

    def getWords(self):
        return self.vectorizer.get_feature_names_out()

    def convertQueryToVector(self, query):
        return self.vectorizer.transform([query])

    def getSimilarityDocuments(self, queryVector, top_k):
        similarity = cosine_similarity(queryVector, self.vector_space_model)
        similarity = similarity.flatten()
        sorted_indices = similarity.argsort()[::-1]
        top_documents = sorted_indices[:top_k]
        return [index for index in top_documents if similarity[index] > 0.00009]

    def getDocumentVectors(self, doc_id=None):
        if doc_id is not None:
            # لا يمكن استخدام doc_id كفهرس مباشر، لأنه قد لا يطابق ترتيب المصفوفة
            # لذلك يجب البحث عن موضع doc_id في self.document_map
            try:
                index = self.document_map.index(doc_id)
                return self.vector_space_model[index]
            except ValueError:
                return None
        return self.vector_space_model

    def get_document_name(self, index):
        return self.document_map[index]

    def setVectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def setDocumentMap(self, document_map):
        self.document_map = document_map  

    def setDocumentVectors(self, doc_vectors):
        self.vector_space_model = doc_vectors

    def getDocumentMap(self):
        return self.document_map 

    def get_vectorizer(self):
        return self.vectorizer
    
    def getSimilarityScores(self, query_vector):
     """
      إرجاع مصفوفة التشابه بين متجه الاستعلام و جميع الوثائق باستخدام cosine similarity.
     """
     if self.vector_space_model is None:
           raise ValueError("vector_space_model غير مُحمّل")

     similarity = cosine_similarity(query_vector, self.vector_space_model)
     return similarity.flatten()
 
    def get_top_k_documents(self, query: str, top_k: int = 1000):
        """
        ترجع أعلى top_k وثائق تشابهًا مع الاستعلام بناءً على TF-IDF.
        """
        query_vector = self.convertQueryToVector(query)
        similarity_scores = self.getSimilarityScores(query_vector)

        top_indices = similarity_scores.argsort()[::-1][:top_k]
        top_doc_ids = [self.document_map[i] for i in top_indices]
        top_scores = [similarity_scores[i] for i in top_indices]

        return top_doc_ids, top_scores
