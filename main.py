from flask import Flask, render_template, request
from elasticsearch import Elasticsearch
# from Services.HybridRetrievalService.Hybridbmbert import SearchEngineBERT_BM25
# from Services.HybridRetrievalService.HybridRepresentationScript import HybridRepresentationParallel
from Services.SearchService.SearchEngine import SearchEngineHybrid
from Services.SearchService.Searchbert import SearchEngineBERTOnly
from Services.SearchService.Searchbertbm import SearchEngineBERTBM25Hybrid
from Services.SearchService.Searchbm import SearchEngineBM25Only
from flask import Flask, jsonify, request
from Services.FilesManagmentService.FilesServices import *
from Services.SearchService.Searchtfidf import SearchEngineTFIDFOnly
from Services.SearchService.Searchword import SearchEngineWord2VecOnly


app = Flask(__name__)


es = Elasticsearch(['http://127.0.0.1:5000'])

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    dataset_type = request.form.get('dataset_type')

    searchEngine = SearchEngineWord2VecOnly(dataset_type)  # 👈 إنشاء الكائن
    search_results = searchEngine.search(query)        # 👈 استخدام الكائن الصحيح

    corrected_query = searchEngine.corrected_query  # ← نأخذ الكلمة المصححة هنا (لو طبقت تصحيح)

    results = []
    for doc in search_results:
        result = {
            'doc_id': doc['doc_name'],
            'content': doc['content']
        }
        results.append(result)

    return render_template('results.html', results=results, query=query, corrected_query=corrected_query)

def getDatasetFileContentAsArray(dataset_type, file_name):
    path = f"Services/FilesManagmentService/storge/Datasets/{dataset_type}/{file_name}"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    return []

@app.route("/suggestions", methods=["GET"])
def generate_suggestions():
    query = request.args.get("query", "").strip().lower()
    dataset_type = request.args.get("dataset_type", "").strip()

    if not query or not dataset_type:
        return jsonify([])

    suggestions = getDatasetFileContentAsArray(dataset_type, "suggestions.txt")

    # فلترة الجمل التي تحتوي على الاستعلام
    filtered = [s.strip() for s in suggestions if query in s.lower()]
    
    # إزالة التكرار
    filtered = list(set(filtered))

    # تجاهل الجمل القصيرة جدًا (أقل من 3 كلمات)
    filtered = [s for s in filtered if len(s.split()) >= 3]

    return jsonify(filtered)


if __name__ == '__main__':
    app.run(debug=True)
    59 
    
    
    
    
    
    
    
