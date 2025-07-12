import mysql.connector
from mysql.connector import Error
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Services.FilesManagmentService.FilesServices import *
from Services.TextProcessingService.TextProcessingServices import * 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import numpy as np


def get_all_documents_contents(dataset_name):
    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',  # عدل حسب بيانات الاتصال
            password='',  # عدل حسب بيانات الاتصال
            database='search_engine'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            query = """
                SELECT text FROM documents2
                WHERE dataset_name = %s
            """
            cursor.execute(query, (dataset_name,))
            results = cursor.fetchall()
            return [row[0] for row in results]

    except Error as e:
        print(f"❌ خطأ في الاتصال بقاعدة البيانات أو تنفيذ الاستعلام: {e}")
        return []

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

def get_all_document_ids(dataset_name):
    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='search_engine'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            query = """
                SELECT doc_id FROM documents2
                WHERE dataset_name = %s
                ORDER BY doc_id
            """
            cursor.execute(query, (dataset_name,))
            results = cursor.fetchall()
            return [row[0] for row in results]

    except Error as e:
        print(f"❌ خطأ في جلب معرفات المستندات: {e}")
        return []

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

def get_all_queries(dataset_name):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='search_engine'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            query = """
                SELECT query_id, content FROM queries2
                WHERE dataset_name = %s
            """
            cursor.execute(query, (dataset_name,))
            return cursor.fetchall()

    except Error as e:
        print(f"❌ خطأ في جلب الاستعلامات: {e}")
        return []

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_all_qrels(dataset_name):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='search_engine'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            query = """
                SELECT query_id, doc_id, relevance FROM qrels2
                WHERE dataset_name = %s
            """
            cursor.execute(query, (dataset_name,))
            return cursor.fetchall()

    except Error as e:
        print(f"❌ خطأ في جلب علاقات الصلة (qrels): {e}")
        return []

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def checkFileExist(dataset_name, object_name):
     path = os.path.join(
        getprojectPath(),
        f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/objects",
        object_name
    )
     return os.path.isfile(path)