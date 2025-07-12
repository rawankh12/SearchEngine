import os
import joblib
from gensim.models import Word2Vec
from Services.TextProcessingService.TextProcessingServices import *
from Services.FilesManagmentService.FilesServices import *

def saveObjectTo_Joblib_File(obj, dataset_name, filename):
    path = os.path.join(getprojectPath(), "Services/FilesManagmentService/storge/Datasets/", dataset_name, filename)
    joblib.dump(obj, path)

def getObjectFrom_Joblib_File(dataset_name, filename):
    path = os.path.join(getprojectPath(), "Services/FilesManagmentService/storge/Datasets/", dataset_name, filename)
    return joblib.load(path)

class WordEmbeddingModel:

    def __init__(self, dataset_name, firstTime=False):
        self.dataset_name = dataset_name
        self.model = None

        if not firstTime:
            try:
                self.model = getObjectFrom_Joblib_File(dataset_name, "EmbeddingModel.joblib")
                print("تم تحميل النموذج المدرب مسبقاً بنجاح")
            except Exception as e:
                print(f"تحذير: {str(e)} - سيتم إنشاء نموذج جديد")
                self.train_model()

    def train_model(self):
        try:
            docs_path = os.path.join(getprojectPath(),
                                     "Services/FilesManagmentService/storge/Datasets/",
                                     self.dataset_name,
                                     "docs")

            if not os.path.exists(docs_path):
                raise FileNotFoundError(f"مسار المستندات غير موجود: {docs_path}")

            print('جاري معالجة البيانات وتدريب النموذج، يرجى الانتظار...')

            documents = []
            for filename in os.listdir(docs_path):
                try:
                    content = get_document_content_Joblib_File(self.dataset_name, filename)
                    tokens = cleanTokenizeText(content)
                    documents.append(tokens)
                except Exception as e:
                    print(f"تحذير: خطأ في معالجة الملف {filename}: {str(e)}")
                    continue

            if not documents:
                raise ValueError("لا توجد مستندات صالحة للتدريب")

            self.model = Word2Vec(
                documents,
                vector_size=100,
                window=5,
                min_count=3,
                workers=4
            )

            saveObjectTo_Joblib_File(self.model, self.dataset_name, 'EmbeddingModel.joblib')
            print("تم تدريب النموذج وحفظه بنجاح")

        except Exception as e:
            print(f"خطأ في تدريب النموذج: {str(e)}")
            raise

    def get_word_embedding(self, word):
        if not self.model:
            self.train_model()

        try:
            return self.model.wv[word]
        except KeyError:
            print(f"تحذير: الكلمة '{word}' غير موجودة في المفردات")
            return None

    def search_similar_words(self, word, topn=5):
        if not self.model:
            self.train_model()

        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            print(f"تحذير: الكلمة '{word}' غير موجودة في المفردات")
            return []
        except Exception as e:
            print(f"خطأ في البحث عن كلمات مماثلة: {str(e)}")
            return []