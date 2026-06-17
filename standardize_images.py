from PIL import Image
import os
#本程序用于把获得的图片统一格式
# 输入输出目录

INPUT_DIR = (
    "datasets/"
    "deduplicated_images"
)

OUTPUT_DIR = (
    "datasets/"
    "deduplicated_sorted_images"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 支持的图片格式

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# 开始处理

image_index = 1

for filename in os.listdir(INPUT_DIR):

    file_lower = filename.lower()

    # 跳过非图片文件
    if not file_lower.endswith(VALID_EXTENSIONS):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    try:

        # 打开图片
        img = Image.open(input_path)

        # 转 RGB
        img = img.convert("RGB")

        # 新文件名
        new_filename = (
            f"{image_index:06d}.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            new_filename
        )

        # 保存 JPG
        img.save(
            output_path,
            "JPEG",
            quality=95
        )

        print(
            f"[{image_index}] "
            f"{filename}"
            f" -> "
            f"{new_filename}"
        )

        image_index += 1

    except Exception as e:

        print(f"处理失败: {filename}")

        print(e)

print("\n================================")
print("图片标准化完成！")
print(f"总图片数量: {image_index - 1}")
print("================================")
