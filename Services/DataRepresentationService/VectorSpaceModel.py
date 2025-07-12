import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Services.FilesManagmentService.FilesServices import *
from Services.TextProcessingService.TextProcessingServices import * 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import numpy as np
from Services.DatabaseService.DatabaseServices import checkFileExist, get_all_document_ids, get_all_documents_contents 


class VectorSpaceModel:
    def __init__(self , datasetName , firstTime = False ):
        self.dataset_name = datasetName
        if not firstTime:
            self.vector_space_model = getObjectFrom_Joblib_File(datasetName, 'doc_vectors')
            self.document_map = getObjectFrom_Joblib_File(datasetName, 'docs_map')
            self.vectorizer = getObjectFrom_Joblib_File(datasetName, 'Model')
            # self.inverted_index = getObjectFrom_Joblib_File(datasetName, 'inverted_index')  

   
    def buildeVectorSpaceModel(self):
    # تحقق من وجود الملفات مسبقًا
     if all([
        checkFileExist(self.dataset_name, "doc_vectors.joblib"),
        checkFileExist(self.dataset_name, "Model.joblib"),
        checkFileExist(self.dataset_name, "docs_map.joblib")
     ]):
        print("✅ التمثيلات موجودة مسبقًا، تم تجاوز عملية التدريب.")
        return

     print("\n📡 جاري جلب المستندات من قاعدة البيانات...")

     raw_documents = get_all_documents_contents(self.dataset_name)
     document_ids = get_all_document_ids(self.dataset_name)

     documents_texts = []
     valid_doc_ids = []

     for i, content in enumerate(raw_documents):
        cleaned = cleanTextAlgorithm(content)
        if cleaned and cleaned.strip():
            documents_texts.append(cleaned)
            valid_doc_ids.append(document_ids[i])
        else:
            print(f"⚠️ المستند رقم {i} تم تجاهله (النص فارغ بعد المعالجة)")

     if not documents_texts:
        print("❌ لا يوجد مستندات صالحة بعد المعالجة. إلغاء العملية.")
        return

     print(f"\n✅ عدد المستندات المقبولة: {len(documents_texts)}")
     print("\n🔍 عرض أول 5 مستندات:")
     for i in range(min(5, len(documents_texts))):
        print(f"\n--- مستند {i+1} ---\n{documents_texts[i]}")

     # بناء النموذج
     vectorizer = TfidfVectorizer(
        tokenizer=word_tokenize,
        max_df=0.75,
        preprocessor=cleanTextAlgorithm  
     )

     print("\n⚙️ جاري بناء نموذج TF-IDF...")
     vectorizer.fit(documents_texts)
     self.vector_space_model = vectorizer.transform(documents_texts)
     self.vectorizer = vectorizer
     self.document_map = valid_doc_ids

     # حفظ النتائج
     saveObjectTo_Joblib_File(self.vector_space_model, self.dataset_name, "doc_vectorsvsm.joblib")
     saveObjectTo_Joblib_File(self.vectorizer, self.dataset_name, "Model.joblib")
     saveObjectTo_Joblib_File(self.document_map, self.dataset_name, "docs_map.joblib")

     print("\n✅ تم حفظ النموذج، التمثيلات، وخريطة الوثائق بنجاح.")
    
    
        # ✳️ بناء الفهرس المعكوس
        # self.inverted_index = self.build_inverted_index(documents_texts)
        # saveObjectTo_Joblib_File(self.inverted_index, self.dataset_name, "inverted_index")

        # حفظ الفهرس المعكوس بصيغة JSON لسهولة التحليل
        # json_path = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{self.dataset_name}/inverted_index.json"
        # with open(json_path, 'w', encoding='utf-8') as f:
        #   json.dump(self.inverted_index, f, ensure_ascii=False, indent=4)

  
    # def build_inverted_index(self, documents):
    #     inverted_index = {}
    #     for idx, doc in enumerate(documents):
    #         tokens = word_tokenize(cleanTextAlgorithm(doc))
    #         for token in set(tokens):  # استخدام set لتجنب التكرار
    #             if token not in inverted_index:
    #                 inverted_index[token] = []
    #             inverted_index[token].append(idx)
    #     return inverted_index

    # def get_inverted_index(self):
    #     return self.inverted_index
    
    def buildDocumentsMap(self):
        print("please wait...")
        docs_path = getprojectPath() + "Services/FilesManagmentService/storge/Datasets/" + self.dataset_name + "/docs"
        document_map = [get_document_Id_Joblib_File(self.dataset_name, filename) for filename in os.listdir(docs_path)]
        saveObjectTo_Joblib_File(document_map, self.dataset_name, "docs_map")


    def printVectorSpaceModel(self):
        df = pd.DataFrame(self.vector_space_model.toarray(), columns=self.vectorizer.get_feature_names_out())
        print(df)

    def getWords(self):
        return self.vectorizer.get_feature_names_out()

    def convertQueryToVector(self , query):
        return self.vectorizer.transform([query])

    def getSimilarityDocuments(self , queryVector , top_k):
        similarity = cosine_similarity(queryVector , self.vector_space_model)
        similarity = similarity.flatten()
        sorted_indices = similarity.argsort()[::-1]
        top_documents = sorted_indices[:top_k]
        return [index for index in top_documents if similarity[index] > 0.00009]

    def getDocumentVectors(self, doc_id=None):
        if doc_id is not None:
            if doc_id in self.vector_space_model:
                return self.vector_space_model[doc_id]
            else:
                return None
        return self.vector_space_model

    def get_document_name(self, index):
        return self.document_map[index]

    def setVectorizer(self , vectorizer):
        self.vectorizer = vectorizer

    def setDocumentMap(self , document_map):
        self.document_map = document_map  

    def setDocumentVectors(self , doc_vectors):
        self.vector_space_model = doc_vectors

    def getDocumentMap(self):
        return self.document_map 

    def get_vectorizer(self):
        return self.vectorizer
    