from PIL import Image
import imagehash

import os
import shutil

#本程序用于去掉重复的图片
# 输入输出目录


INPUT_DIR = (
    "datasets/"
    "standardized_images"
)

OUTPUT_DIR = (
    "datasets/"
    "deduplicated_images"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 哈希字典

hash_dict = {}


# 相似度阈值
# 数值越小越严格



HASH_THRESHOLD = 6


# 统计


total_images = 0
duplicate_images = 0
saved_images = 0


# 开始遍历


for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith(".jpg"):
        continue

    total_images += 1

    file_path = os.path.join(
        INPUT_DIR,
        filename
    )

    try:

        # 打开图片
        img = Image.open(file_path)

        # 计算感知哈希
        img_hash = imagehash.phash(img)

        is_duplicate = False

        # 与已有图片比较
        for existing_hash in hash_dict:

            distance = (
                img_hash - existing_hash
            )

            # 判断相似
            if distance <= HASH_THRESHOLD:

                print(
                    f"重复图片: {filename}"
                )

                duplicate_images += 1

                is_duplicate = True

                break

        # 保存非重复图片
        if not is_duplicate:

            output_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            shutil.copy(
                file_path,
                output_path
            )

            hash_dict[img_hash] = filename

            saved_images += 1

            print(
                f"保留: {filename}"
            )

    except Exception as e:

        print(
            f"处理失败: {filename}"
        )

        print(e)

# 输出统计


print("\n================================")
print("去重完成！")
print("================================")

print(f"总图片数量: {total_images}")
print(f"重复图片数量: {duplicate_images}")
print(f"保留图片数量: {saved_images}")

print("================================")
