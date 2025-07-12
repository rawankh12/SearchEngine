# -*- coding: utf-8 -*-
from Services.FilesManagmentService.FilesServices import *
import ir_datasets
import pandas as pd
import os
from Services.TextProcessingService.TextProcessingServices import cleanTextAlgorithm, queryCleanTextAlgorithm
from nltk.tokenize import word_tokenize


dataset = ir_datasets.load('beir/quora/dev')

# dataset_name = "test"


class InitializeDatasetes:
    
    def __init__(self, loadDataaset, dataset_folder):
        self.dataset = ir_datasets.load(loadDataaset)
        self.dataset_name = dataset_folder

    def loadDocuments(self):
        i = 0
        for doc in self.dataset.docs_iter():
            i += 1
            print(i)
            save_document_To_Joblib_File(doc, self.dataset_name, str(doc.doc_id) + ".joblib")
        print("--- Done ---")

    def loadQueries(self):
        i = 1
        for query in self.dataset.queries_iter():
            save_query_To_Joblib_File(query, self.dataset_name, "Query_" + query.query_id + ".joblib")
            print(i)
            i += 1
        print("--- Done ---")

    def loadQrels(self):
        i = 0
        for qrel in self.dataset.qrels_iter():
            save_qrel_To_Joblib_File(qrel, self.dataset_name, "qrel_" + qrel.query_id + ".joblib")
            i += 1
            print(i)
        print("--- Done ---")

    def createDocumentsMap(self):
        i = 0
        docs_path = os.path.join(getprojectPath(), "Services/FilesManagmentService/storge/Datasets/",
                                 self.dataset_name, "docs")
        documents_map = []
        for filename in os.listdir(docs_path):
            i += 1
            print(i)
            doc_id = get_document_Id_Joblib_File(self.dataset_name, filename)
            documents_map.append(doc_id)
        saveObjectTo_Joblib_File(documents_map, self.dataset_name, 'docs_map.joblib')
        print("--- Done ---")


#------------------------------------ CODE --------------------------------  

# dataset = InitializeDatasetes('trec-tot/2023/train', "trec-tot-2023-train") 
dataset = InitializeDatasetes('beir/quora/dev', "beir_quora") 

dataset.loadDocuments()       # تحميل الوثائق إلى مجلد docs
dataset.loadQueries()         # تحميل الاستعلامات إلى مجلد queries
dataset.loadQrels()           # تحميل علاقات الصلة (relevance) إلى مجلد qrels
dataset.createDocumentsMap()  # بناء خريطة الوثائق docs_map.joblib
