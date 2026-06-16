import os
from collections import defaultdict
#本程序用于统计清洗后的各种类的样本数量
# 标签文件目录
label_dir = r'.\datasets\labels'

# 类别名称（按照 data.yaml 中 names 顺序）
class_names = {
    0: 'person',
    1: 'umbrella',
    2: 'knife',
    3: 'gun'
}

# 统计结果
class_count = defaultdict(int)

# 遍历所有 txt 标签文件
for file in os.listdir(label_dir):

    if not file.endswith('.txt'):
        continue

    txt_path = os.path.join(label_dir, file)

    with open(txt_path, 'r', encoding='utf-8') as f:

        lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            # YOLO标签至少5列
            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            class_count[class_id] += 1

# 输出统计结果
print("=" * 50)
print("数据集类别统计")
print("=" * 50)

total = 0

for class_id in sorted(class_names.keys()):

    count = class_count[class_id]

    print(f"{class_names[class_id]:<10} : {count}")

    total += count

print("=" * 50)
print(f"总目标数: {total}")
print("=" * 50)

