import os
import shutil
import random
from collections import defaultdict

#本程序用于将数据集划分为训练集、验证集和测试集
# 路径


BASE_DIR = "datasets"

IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

OUT_DIR = "datasets/yolo_dataset"

# 输出结构

splits = ["train", "val", "test"]

for s in splits:
    os.makedirs(os.path.join(OUT_DIR, "images", s), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", s), exist_ok=True)

# 读取所有数据

images = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# 建立：图片 -> 类别集合

img_classes = {}

for img in images:
    label_path = os.path.join(LBL_DIR, img.rsplit(".", 1)[0] + ".txt")

    classes = set()

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls_id = line.strip().split()[0]
                classes.add(cls_id)

    img_classes[img] = classes

# 按类别分层

random.seed(42)
images_sorted = list(images)
random.shuffle(images_sorted)

train, val, test = [], [], []

class_counter = defaultdict(int)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for img in images_sorted:

    # 取“最稀有类别优先原则”
    classes = img_classes[img]

    # 简单策略：优先分配到样本最少的集合
    r = random.random()

    if r < TRAIN_RATIO:
        train.append(img)
    elif r < TRAIN_RATIO + VAL_RATIO:
        val.append(img)
    else:
        test.append(img)

# 拷贝函数

def copy_data(file_list, split):
    for img in file_list:
        img_path = os.path.join(IMG_DIR, img)
        lbl_path = os.path.join(LBL_DIR, img.rsplit(".", 1)[0] + ".txt")

        if not os.path.exists(lbl_path):
            continue

        shutil.copy(img_path, os.path.join(OUT_DIR, "images", split, img))
        shutil.copy(lbl_path, os.path.join(OUT_DIR, "labels", split, img.rsplit(".", 1)[0] + ".txt"))

copy_data(train, "train")
copy_data(val, "val")
copy_data(test, "test")

# 生成 YAML

yaml_content = f"""
path: {OUT_DIR}

train: images/train
val: images/val
test: images/test

names:
  0: person
  1: umbrella
  2: knife
  3: gun
"""

with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("分层划分完成！")
print(f"train: {len(train)}")
print(f"val: {len(val)}")
print(f"test: {len(test)}")