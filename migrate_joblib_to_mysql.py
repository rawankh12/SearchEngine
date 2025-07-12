# -*- coding: utf-8 -*-
import mysql.connector
import os
import joblib

# إعدادات الاتصال بقاعدة البيانات
mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "",  # ضع كلمة مرورك هنا إن وجدت
    "database": "search_engine"
}

def migrate_documents(dataset_name, conn):
    print("📄 بدء ترحيل الوثائق...")
    docs_path = os.path.join("Services", "FilesManagmentService", "storge", "Datasets", dataset_name, "docs")
    cursor = conn.cursor()
    success, fail = 0, 0
    files = os.listdir(docs_path)

    for i, file in enumerate(files, 1):
        try:
            file_path = os.path.join(docs_path, file)
            doc = joblib.load(file_path)
            cursor.execute(
                "INSERT INTO documents2 (doc_id, dataset_name, text) VALUES (%s, %s, %s)",
                (doc.doc_id, dataset_name, doc.text)
            )
            conn.commit()
            success += 1
            if i % 100 == 0:
                print(f"  ✅ تم {i} / {len(files)}")
        except Exception as e:
            fail += 1
            print(f"  ⚠️ خطأ في {file}: {e}")

    cursor.close()
    print(f"📄 الوثائق: {success} ناجحة، {fail} فاشلة")


def migrate_queries(dataset_name, conn):
    print("❓ بدء ترحيل الاستعلامات...")
    queries_path = os.path.join("Services", "FilesManagmentService", "storge", "Datasets", dataset_name, "queries")
    cursor = conn.cursor()
    success, fail = 0, 0
    files = os.listdir(queries_path)

    for file in files:
        try:
            file_path = os.path.join(queries_path, file)
            query = joblib.load(file_path)
            cursor.execute(
                "INSERT INTO queries2 (query_id, dataset_name, text) VALUES (%s, %s, %s)",
                (query.query_id, dataset_name, query.text)
            )
            conn.commit()
            success += 1
        except Exception as e:
            fail += 1
            print(f"  ⚠️ خطأ في {file}: {e}")

    cursor.close()
    print(f"❓ الاستعلامات: {success} ناجحة، {fail} فاشلة")


def migrate_qrels(dataset_name, conn):
    print("🔗 بدء ترحيل qrels...")
    qrels_path = os.path.join("Services", "FilesManagmentService", "storge", "Datasets", dataset_name, "qrels")
    cursor = conn.cursor()
    success, fail = 0, 0
    files = os.listdir(qrels_path)

    for file in files:
        try:
            file_path = os.path.join(qrels_path, file)
            qrel = joblib.load(file_path)
            cursor.execute(
                "INSERT INTO qrels2 (query_id, doc_id, relevance, dataset_name) VALUES (%s, %s, %s, %s)",
                (qrel.query_id, qrel.doc_id, qrel.relevance, dataset_name)
            )
            conn.commit()
            success += 1
        except Exception as e:
            fail += 1
            print(f"  ⚠️ خطأ في {file}: {e}")

    cursor.close()
    print(f"🔗 qrels: {success} ناجحة، {fail} فاشلة")


def migrate_all(dataset_name):
    print(f"🚀 بدء الترحيل لـ: {dataset_name}")
    mysql_config = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "search_engine"
    }

    conn = mysql.connector.connect(**mysql_config)
    migrate_documents(dataset_name, conn)
    migrate_queries(dataset_name, conn)
    migrate_qrels(dataset_name, conn)
    conn.close()
    print("✅ الترحيل الكامل تم.")
    

# شغّل السكربت هنا
if __name__ == "__main__":
    dataset_name = "beir_quora"  # غيّره حسب اسم مجلد الداتاست
    migrate_all(dataset_name)
