import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# دوال التقييم الأساسية
def precisionAt_k(relevant_docs, retrieved_docs, k):
    retrieved_at_k = retrieved_docs[:k]
    relevant_and_retrieved = set(relevant_docs) & set(retrieved_at_k)
    return len(relevant_and_retrieved) / k

def recall(relevant_docs, retrieved_docs):
    relevant_and_retrieved = set(relevant_docs) & set(retrieved_docs)
    return len(relevant_and_retrieved) / len(relevant_docs) if len(relevant_docs) != 0 else 0

def averagePrecision(relevant_docs, retrieved_docs):
    relevant_docs_set = set(relevant_docs)
    score = 0.0
    num_hits = 0.0
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(relevant_docs) if len(relevant_docs) != 0 else 0

def meanAveragePrecision(all_relevant_docs, all_retrieved_docs):
    avg_precisions = []
    for relevant_docs, retrieved_docs in zip(all_relevant_docs, all_retrieved_docs):
        avg_precisions.append(averagePrecision(relevant_docs, retrieved_docs))
    return sum(avg_precisions) / len(avg_precisions) if len(avg_precisions) != 0 else 0

def meanReciprocalRank(all_relevant_docs, all_retrieved_docs):
    reciprocal_ranks = []
    for relevant_docs, retrieved_docs in zip(all_relevant_docs, all_retrieved_docs):
        for rank, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if len(reciprocal_ranks) != 0 else 0


# ✅ لتخزين نتائج النماذج
evaluation_results = {}

# ✅ تسجيل نتائج كل نموذج
def log_model_results(model_name, all_relevant_docs, all_retrieved_docs):
    p_at_20 = sum(precisionAt_k(rel, ret, 20) for rel, ret in zip(all_relevant_docs, all_retrieved_docs)) / len(all_relevant_docs)
    rec = sum(recall(rel, ret) for rel, ret in zip(all_relevant_docs, all_retrieved_docs)) / len(all_relevant_docs)
    map_score = meanAveragePrecision(all_relevant_docs, all_retrieved_docs)
    mrr_score = meanReciprocalRank(all_relevant_docs, all_retrieved_docs)

    evaluation_results[model_name] = {
        'P@20': round(p_at_20, 4),
        'MAP': round(map_score, 4),
        'R@100': round(rec, 4),
        'MRR': round(mrr_score, 4)
    }


# ✅ رسم جدول النتائج النهائية
def generate_results_table():
    if not evaluation_results:
        print("⚠️ لا توجد نتائج لتوليد الجدول.")
        return

    df = pd.DataFrame(evaluation_results).T
    df = df[['P@20', 'MAP', 'R@100', 'MRR']]

    # رسم Heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="cubehelix", linewidths=1, linecolor='black')
    plt.title("Evaluation Metrics by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig("evaluation_table.png", dpi=300)
    plt.show()
    print("✅ تم حفظ الجدول في: evaluation_table.png")


# 📝 مثال استخدام:

all_qrels = [[
    "1000063_4", "1000063_8", "1000063_9", "1000063_0", "1000063_1", 
    "1000063_2", "1000063_3", "1000063_5", "1000063_6", "1000063_7", 
    "1000063_12", "1000063_13", "1000063_10", "1000063_11", 
    "1000063_16", "1000063_14", "1000063_15"
]]

all_results_hybrid = [[
    "1036253_4", "1000063_1", "1001403_10", "1360943_23", "4409820_9",
    "2561962_5", "1767104_45", "3304296_3", "3298155_11", "1855876_7"
]]

all_results_bm25 = [[
     "1036253_4",
    "1000063_1",
    "1001403_10",
    "1360943_23",
    "1767104_45",
    "1855876_7",
    "2561962_5",
    "3298155_11",
    "3304296_3",
    "3807324_1",
    "4409820_9",
    "4981_13",
    "706526_1",
    "2098550_13",
    "1000063_10",
    "1646376_3",
    "2228578_4",
    "3763316_1",
    "3244724_1",
    "1360943_26"
]]

all_results_tfidf = [[
    "4409820_9",
    "3304296_3",
    "706526_1",
    "3298155_11",
    "3807324_1",
    "1855876_7",
    "2561962_5",
    "4981_13",
    "1767104_45",
    "1036253_4",
    "1646376_3",
    "2098550_13",
    "1001403_10",
    "1360943_7",
    "435035_7",
    "3763316_1",
    "1000063_10",
    "1036253_8",
    "1208376_20",
    "3244724_14"
    ]]

log_model_results("BM25", all_qrels, all_results_bm25)
log_model_results("TF-IDF", all_qrels, all_results_tfidf)
# log_model_results("BERT", all_qrels, all_results_bert)
log_model_results("Hybrid", all_qrels, all_results_hybrid)

generate_results_table()
