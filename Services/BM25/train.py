
# اسم الداتاست لازم يكون نفس الاسم يلي بتستخدمه بقاعدة البيانات أو التخزين
from Services.BM25.BM import BM25ModelData


dataset_name = "antique-train"

# تهيئة النموذج
bm25 = BM25ModelData(dataset_name)

# استعلام تجربة
query = "important"

# الحصول على أفضل 5 نتائج
results = bm25.search(query, top_n=5)

# طباعة النتائج
for index, score in results:
    print(f"📄 مستند رقم {index} – درجة التشابه: {score:.3f}")
