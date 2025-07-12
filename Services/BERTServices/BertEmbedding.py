import os
import joblib
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from Services.DatabaseService.DatabaseServices import get_all_documents_contents, get_all_document_ids
from Services.TextProcessingService.TextProcessingServices import cleanTextAlgorithm
from Services.FilesManagmentService.FilesServices import get_all_documents_from_joblib_folder, get_document_Joblib_File, getprojectPath


class BERTRepresentation:
    def __init__(self, dataset_name, firstTime=False):
        self.dataset_name = dataset_name
        self.tokenizer = None
        self.model = None
        self.documents_vectors = None
        self.document_names = None

        if not firstTime:
            self.load_model()
        else:
            self.train_model()

    def load_model(self):
        try:
            model_path = os.path.join(getprojectPath(), f"Services/FilesManagmentService/storge/Datasets/{self.dataset_name}/objects/BERTModel.joblib")
            vectors_path = os.path.join(getprojectPath(), f"Services/FilesManagmentService/storge/Datasets/{self.dataset_name}/objects/BERTVectors.joblib")
            names_path = os.path.join(getprojectPath(), f"Services/FilesManagmentService/storge/Datasets/{self.dataset_name}/objects/docs_map.joblib")

            if os.path.exists(model_path) and os.path.exists(vectors_path) and os.path.exists(names_path):
                print("✅ تم تحميل نموذج BERT من الملفات...")
                self.tokenizer, self.model = joblib.load(model_path)
                self.documents_vectors = joblib.load(vectors_path)
                self.document_names = joblib.load(names_path)
            else:
                print("⚠️ لم يتم العثور على ملفات BERT - سيتم تدريب النموذج.")
                self.train_model()
        except Exception as e:
            print(f"❌ خطأ في تحميل نموذج BERT: {str(e)}")

    def train_model(self):
        print("🧠 تدريب نموذج BERT على المستندات...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

        raw_documents = get_all_documents_from_joblib_folder(self.dataset_name)
        self.document_names = get_all_document_ids(self.dataset_name)

        self.documents_vectors = []

        for i, doc in enumerate(raw_documents):
            try:
                cleaned = cleanTextAlgorithm(doc)
                vec = self.text_to_vector(cleaned)
                self.documents_vectors.append(vec)
                print(f"✅ تمثيل المستند {i+1} تم بنجاح")
            except Exception as e:
                print(f"⚠️ خطأ في مستند {i+1}: {e}")
                self.documents_vectors.append(np.zeros(768))

        self.documents_vectors = np.array(self.documents_vectors)

        # حفظ النموذج والتمثيلات والأسماء
        model_path = os.path.join(getprojectPath(), f"Services/FilesManagmentService/storge/Datasets/{self.dataset_name}/objects/BERTModel.joblib")
        vectors_path = os.path.join(getprojectPath(), f"Services/FilesManagmentService/storge/Datasets/{self.dataset_name}/objects/BERTVectors.joblib")
        names_path = os.path.join(getprojectPath(), f"Services/FilesManagmentService/storge/Datasets/{self.dataset_name}/objects/BERTDocNames.joblib")
        joblib.dump((self.tokenizer, self.model), model_path)
        joblib.dump(self.documents_vectors, vectors_path)
        joblib.dump(self.document_names, names_path)
        print("✅ تم حفظ نموذج BERT والتمثيلات وأسماء الوثائق.")

    def text_to_vector(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def convertQueryToVector(self, query):
        return self.text_to_vector(query)

    def getSimilarityScores(self, query_vector, top_k=10):
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_vector], self.documents_vectors)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        return top_indices, similarities
      