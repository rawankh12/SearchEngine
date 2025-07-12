import os
import joblib
import numpy as np
from gensim.models import Word2Vec
from Services.TextProcessingService.TextProcessingServices import cleanTokenizeText
from Services.FilesManagmentService.FilesServices import *
from Services.DatabaseService.DatabaseServices import get_all_documents_contents, get_all_document_ids
from sklearn.metrics.pairwise import cosine_similarity


class WordEmbeddingModelData:

    def __init__(self, dataset_name, firstTime=False):
        self.dataset_name = dataset_name
        self.model = None
        self.doc_vectors = None

        if not firstTime:
            try:
                self.model = getObjectFrom_Joblib_File(dataset_name, "EmbeddingModel")
                print("✅ تم تحميل نموذج Word2Vec بنجاح")
            except Exception as e:
                print(f"⚠️ لم يتم تحميل النموذج: {str(e)} - سيتم تدريبه الآن")
                self.train_model()

            try:
                self.doc_vectors = getObjectFrom_Joblib_File(dataset_name, "doc_vectors")
                print("✅ تم تحميل تمثيلات الوثائق")
            except:
                print("⚠️ لم يتم تحميل تمثيلات الوثائق")


    def train_model(self):
        print('🔄 جاري تحميل المستندات ومعالجتها...')
        raw_documents = get_all_documents_contents(self.dataset_name)
        documents = []

        for content in raw_documents:
            try:
                tokens = cleanTokenizeText(content)
                documents.append(tokens)
            except Exception as e:
                print(f"⚠️ خطأ في مستند: {str(e)}")

        if not documents:
            raise ValueError("❌ لا توجد مستندات صالحة للتدريب")

        self.model = Word2Vec(
            documents,
            vector_size=100,
            window=5,
            min_count=3,
            workers=4
        )

        saveObjectTo_Joblib_File(self.model, self.dataset_name, 'EmbeddingModel.joblib')
        print("✅ تم تدريب النموذج وحفظه بنجاح")

    def buildDocumentVectors(self):
        print("🔄 جاري بناء تمثيلات الوثائق...")
        raw_documents = get_all_documents_contents(self.dataset_name)
        doc_vectors = []

        for i, content in enumerate(raw_documents):
            try:
                tokens = cleanTokenizeText(content)
                embeddings = [self.model.wv[token] for token in tokens if token in self.model.wv]

                if embeddings:
                    doc_vector = np.mean(embeddings, axis=0)
                else:
                    doc_vector = np.zeros(self.model.vector_size)

                doc_vectors.append(doc_vector)
            except Exception as e:
                print(f"⚠️ خطأ في الوثيقة {i}: {str(e)}")
                doc_vectors.append(np.zeros(self.model.vector_size))

        self.doc_vectors = np.array(doc_vectors)
        saveObjectTo_Joblib_File(self.doc_vectors, self.dataset_name, 'doc_vectors.joblib')
        print("✅ تم حفظ تمثيلات الوثائق")

    def convertQueryToVector(self, query):
        tokens = cleanTokenizeText(query)
        embeddings = [self.model.wv[token] for token in tokens if token in self.model.wv]

        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def getSimilarityScores(self, query_vector):
        if self.doc_vectors is None:
            raise ValueError("❌ تمثيلات الوثائق غير محمّلة")
        return cosine_similarity([query_vector], self.doc_vectors).flatten()

    def search_similar_words(self, word, topn=5):
        if not self.model:
            self.train_model()

        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            print(f"⚠️ الكلمة '{word}' غير موجودة في المفردات")
            return []
        except Exception as e:
            print(f"❌ خطأ في البحث: {str(e)}")
            return []

    def get_document_name(self, index):
        try:
            all_ids = get_all_document_ids(self.dataset_name)
            return all_ids[index]
        except IndexError:
            print(f"⚠️ لا يوجد اسم وثيقة للفهرس: {index}")
            return None
