import re
import pandas as pd
import joblib
import csv
from pathlib import Path
import os
import mysql.connector
#---- Files Services -----------


def getprojectPath():
    return "C:/Users/VICTUS/Search_Engine/"



mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "",  
    "database": "search_engine"
}

def get_document_from_db(dataset_name, doc_id):
    try:
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor(dictionary=True)  # لكي يرجع نتائج بشكل dict

        query = "SELECT * FROM documents2 WHERE dataset_name = %s AND doc_id = %s LIMIT 1"
        cursor.execute(query, (dataset_name, doc_id))
        result = cursor.fetchone()

        cursor.close()
        conn.close()
        return result  # dict فيه: id, doc_id, dataset_name, text

    except Exception as e:
        print(f"❌ خطأ في get_document_from_db: {e}")
        return None
def getFileContent(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def fileClear(datasetName, fileName):
    try:
        with open(getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{datasetName}/{fileName}", "w+", encoding="utf-8") as file:
            file.write("")
    except Exception as e:
        print("error:", str(e))

def writeContentToFile(datasetName, fileName, content):
    try:
        with open(getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{datasetName}/{fileName}", "w+", encoding="utf-8") as file:
            file.write(content)
    except Exception as e:
        print("error:", str(e))

def AddContentToFile(datasetName, fileName, content):
    try:
        with open(getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{datasetName}/{fileName}", "a+", encoding="utf-8") as file:
            file.write(content)
    except Exception as e:
        print("error:", str(e))

def getFileContentAsArray(filename):
    with open(getprojectPath() + filename, "r", encoding="utf-8") as file:
        text = file.read()
    return re.findall(r"\S+", text)

def getDatasetFileContentAsArray(datasetName, filename):
    with open(getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{datasetName}/{filename}", "r", encoding="utf-8") as file:
        text = file.read()
    return re.findall(r"\S+", text)

def read_file(directory, filename):
    with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
        return file.read()

def saveObjectTo_CSV_File(myObject, dataset_name, object_name):
    df = pd.DataFrame(myObject.toarray())
    df.to_csv(getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/objects/{object_name}.csv", index=False)

def getObjectFrom_CSV_File(dataset_name, object_name):
    df = pd.read_csv(getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/objects/{object_name}.csv")
    return df.values


# ------------ JOBLIB Versions --------------

def saveObjectTo_Joblib_File(myObject, dataset_name, object_name):
    directory = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/objects"
    Path(directory).mkdir(parents=True, exist_ok=True)
    joblib.dump(myObject, os.path.join(directory, object_name))


def getObjectFrom_Joblib_File(dataset_name, object_name):
    path = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/objects/{object_name}.joblib"
    return joblib.load(path)


def save_document_To_Joblib_File(doc, dataset_name, object_name):
    directory = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/docs"
    Path(directory).mkdir(parents=True, exist_ok=True)
    joblib.dump(doc, os.path.join(directory, object_name))
    

def get_document_Joblib_File(dataset_name, object_name):
    path = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/docs/{object_name}.joblib"
    return joblib.load(path)

def get_all_documents_from_joblib_folder(dataset_name):
    docs_dir = os.path.join(getprojectPath(), f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/docs")
    documents = []

    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".joblib"):
            path = os.path.join(docs_dir, filename)
            try:
                doc_text = joblib.load(path)
                documents.append(doc_text)
            except Exception as e:
                print(f"⚠️ خطأ في تحميل {filename}: {e}")
                documents.append("")  # إذا فشل التحميل، أضف نص فارغ

    print(f"📦 عدد المستندات المحمّلة من الملفات: {len(documents)}")
    return documents

def save_query_To_Joblib_File(query, dataset_name, object_name):
    directory = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/queries"
    Path(directory).mkdir(parents=True, exist_ok=True)
    joblib.dump(query, os.path.join(directory, object_name))


def get_query_Joblib_File(dataset_name, object_name):
    path = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/queries/{object_name}"
    return joblib.load(path)


def save_qrel_To_Joblib_File(qrel, dataset_name, object_name):
    directory = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/qrels"
    Path(directory).mkdir(parents=True, exist_ok=True)
    joblib.dump(qrel, os.path.join(directory, object_name))


def get_qrel_Joblib_File(dataset_name, object_name):
    path = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/qrels/{object_name}"
    return joblib.load(path)


def get_document_Id_Joblib_File(dataset_name, object_name):
    doc = get_document_Joblib_File(dataset_name, object_name)
    return doc.doc_id


def get_document_content_Joblib_File(dataset_name, object_name):
    doc = get_document_Joblib_File(dataset_name, object_name)
    return doc.text


# --------------------------------------------

def build_key_value_transformer(keys, values):
    return dict(zip(keys, values))


# CSV Operations for backup (no change)
def save_To_Dataset(myObject, path):
    df = pd.DataFrame(myObject.toarray())
    df.to_csv(path, index=False)

def get_document_content(datasetName, docNumber):
    path = getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{datasetName}/docs/document_{docNumber}.csv"
    df = pd.read_csv(path)
    return pd.DataFrame(df.values.toarray())

def get_document_id(datasetName, docNumber):
    with open(getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{datasetName}/docs/document_{docNumber}.csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row['doc_id'] for row in reader]

def get_document(datasetName, docNumber):
    with open(getprojectPath() + f"Services/FilesManagmentService/storge/Datasets/{datasetName}/docs/document_{docNumber}.csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]