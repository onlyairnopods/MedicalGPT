# 别人的

import os
import json
import numpy as np
from tqdm import tqdm

# 目标数据集路径（例如训练数据向量化后的数据集）
dataset_vector = "/root/autodl-tmp/processdata/medical/finetune/train_zh_0_vectorized.jsonl"

# 四个查询数据集路径
clinical_medicinequery_dataset_vector = "/root/autodl-tmp/processdata/ceval_vectorized_dataset/clinical_medicine_vectorized.jsonl"
basic_medicinequery_dataset_vector = "/root/autodl-tmp/processdata/ceval_vectorized_dataset/basic_medicine_vectorized.jsonl"
physicianquery_dataset_vector = "/root/autodl-tmp/processdata/ceval_vectorized_dataset/physician_vectorized.jsonl"
veterinary_medicinequery_dataset_vector = "/root/autodl-tmp/processdata/ceval_vectorized_dataset/veterinary_medicine_vectorized.jsonl"

# 将查询数据集名称与对应的路径整理成字典，方便遍历处理
query_dataset_vectors = {
    "clinical_medicine": clinical_medicinequery_dataset_vector,
    "basic_medicine": basic_medicinequery_dataset_vector,
    "physician": physicianquery_dataset_vector,
    "veterinary_medicine": veterinary_medicinequery_dataset_vector
}

def load_full_records(filename):
    """加载完整数据记录，保留所有原始字段
    如果存在 'vector' 字段，则将其转换为 numpy 数组（float32）"""
    records = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"加载 {os.path.basename(filename)}"):
            item = json.loads(line)
            if 'vector' in item:
                item['vector'] = np.array(item['vector'], dtype=np.float32)
            records.append(item)
    return records

def process_query_dataset(query_dataset_vector, output_file):
    """
    加载查询数据集并逐块处理：根据全局变量 target_normalized（目标数据归一化向量）计算相似度，
    为每条查询记录找到 TopK 匹配的目标记录（保留目标记录除 'vector' 外的其他字段并添加 match_score），
    然后写入 output_file 文件中。
    """
    print("\n加载查询数据集...")
    query_data = load_full_records(query_dataset_vector)

    # 分块处理以降低内存占用
    BATCH_SIZE = 5000
    total_batches = (len(query_data) + BATCH_SIZE - 1) // BATCH_SIZE

    with open(output_file, 'w', encoding='utf-8') as fout:
        for batch_idx in tqdm(range(total_batches), desc="处理进度"):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, len(query_data))
            batch = query_data[start:end]

            # 构建查询向量矩阵，并进行归一化
            query_vectors = np.array([item['vector'] for item in batch])
            query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            query_norms = np.where(query_norms == 0, 1e-10, query_norms)
            query_normalized = query_vectors / query_norms

            # 利用归一化向量进行内积计算相似度
            similarity_matrix = np.dot(query_normalized, target_normalized.T)

            # 选择 Top k 个匹配记录（此处 k=50，可根据需要调整）
            k = 50
            top_indices = np.argpartition(-similarity_matrix, k, axis=1)[:, :k]
            sorted_indices = np.argsort(-np.take_along_axis(similarity_matrix, top_indices, axis=1), axis=1)
            final_indices = np.take_along_axis(top_indices, sorted_indices, axis=1)

            # 遍历当前批次的每个查询记录，构造输出结果
            for i in range(len(batch)):
                # 复制原始查询记录，保留原始字段
                original_record = dict(batch[i])
                original_record['matches'] = []
                # 为每个匹配目标记录添加匹配得分
                for idx in final_indices[i]:
                    target_record = dict(target_data[idx])
                    if 'vector' in target_record:
                        del target_record['vector']  # 不保存向量，减小输出体积
                    target_record['match_score'] = float(similarity_matrix[i, idx])
                    original_record['matches'].append(target_record)
                if 'vector' in original_record:
                    del original_record['vector']
                fout.write(json.dumps(original_record, ensure_ascii=False) + '\n')


def main():
    global target_data, target_normalized

    # 1. 加载目标数据集
    print("加载目标数据集...")
    target_data = load_full_records(dataset_vector)
    # 为目标数据添加自动生成的 id（按照顺序）
    for idx, item in enumerate(target_data):
        item['id'] = idx

    # 预处理目标数据：构建向量矩阵并归一化
    target_vectors = np.array([item['vector'] for item in target_data])
    target_norms = np.linalg.norm(target_vectors, axis=1, keepdims=True)
    target_norms = np.where(target_norms == 0, 1e-10, target_norms)
    target_normalized = target_vectors / target_norms

    # 2. 遍历每个查询数据集，逐个处理并保存结果
    for key, query_file in query_dataset_vectors.items():
        print(f"\n处理查询数据集：{key}")
        # 根据查询数据集名称确定输出文件名
        output_filename = f"full_output_with_all_fields_{key}.jsonl"
        print("保存输出到:", output_filename)
        process_query_dataset(query_file, output_filename)


if __name__ == "__main__":
    main()