import mysql.connector

create_tables_sql = """
CREATE DATABASE IF NOT EXISTS search_engine;
USE search_engine;

CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doc_id VARCHAR(255),
    dataset_name VARCHAR(255),
    text LONGTEXT
);

CREATE TABLE IF NOT EXISTS queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_id VARCHAR(255),
    dataset_name VARCHAR(255),
    text TEXT
);

CREATE TABLE IF NOT EXISTS qrels (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_id VARCHAR(255),
    doc_id VARCHAR(255),
    relevance INT,
    dataset_name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS documents2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doc_id VARCHAR(255),
    dataset_name VARCHAR(255),
    text LONGTEXT
);

CREATE TABLE IF NOT EXISTS queries2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_id VARCHAR(255),
    dataset_name VARCHAR(255),
    text TEXT
);

CREATE TABLE IF NOT EXISTS qrels2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_id VARCHAR(255),
    doc_id VARCHAR(255),
    relevance INT,
    dataset_name VARCHAR(255)
);
"""

mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
}

conn = mysql.connector.connect(**mysql_config)
cursor = conn.cursor()

# قسم النص إلى أوامر منفصلة حسب الفواصل المنقوطة
commands = create_tables_sql.split(';')
for command in commands:
    cmd = command.strip()
    if cmd:
        cursor.execute(cmd)

conn.commit()
cursor.close()
conn.close()

print("✅ تم إنشاء قاعدة البيانات والجداول بنجاح.")
