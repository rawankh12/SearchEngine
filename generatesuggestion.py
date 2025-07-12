from ir_datasets import load
import os

def generate_suggestions_from_queries(dataset_name):
    dataset = load(dataset_name)
    queries = dataset.queries_iter()

    suggestions_path = f"Services/FilesManagmentService/storge/Datasets/{dataset_name}/suggestions.txt"
    os.makedirs(os.path.dirname(suggestions_path), exist_ok=True)

    with open(suggestions_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(q.text.strip() + "\n")

    print(f"✅ تم توليد ملف الاقتراحات بنجاح: {suggestions_path}")

# مثال الاستخدام:
if __name__ == "__main__":
    generate_suggestions_from_queries("antique/train")  # ← عدلي هذا الاسم حسب داتاستك
