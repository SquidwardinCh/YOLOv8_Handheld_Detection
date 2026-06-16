from icrawler.builtin import BingImageCrawler
from icrawler.downloader import ImageDownloader
from PIL import Image

import os
import glob

#本程序用于爬取网络图片

# 定义要爬取的图片种类

scenes = {

    # "surveillance0":
    #     "urban CCTV pedestrian scene",
    # "surveillance1":
    #     "shopping mall CCTV pedestrians",
    # "surveillance2":
    #     "campus surveillance pedestrians", 
    # "surveillance3":
    #     "security camera pedestrian scene",
    # "surveillance4":
    #     "top view CCTV pedestrians",

        

}


# 每类下载数量

max_num = {

    # "surveillance0": 200,
    # "surveillance1": 200,
    # "surveillance2": 200,
    # "surveillance3": 200,
    # "surveillance4": 200,

}


# 清理垃圾图片

MIN_SIZE = 15 * 1024

def clean_images(directory):

    for filepath in glob.glob(os.path.join(directory, '*.*')):

        try:

            # 跳过日志文件
            if filepath.endswith(".txt"):
                continue

            # 删除小文件
            if os.path.getsize(filepath) < MIN_SIZE:

                print(f"删除小文件: {filepath}")

                os.remove(filepath)

                continue

            img = Image.open(filepath)

            w, h = img.size

            # 删除过低分辨率
            if w < 200 or h < 200:

                print(f"删除低分辨率图片: {filepath}")

                os.remove(filepath)

                continue

            # 删除极端比例
            ratio = max(w / h, h / w)

            if ratio > 4:

                print(f"删除异常比例图片: {filepath}")

                os.remove(filepath)

                continue

        except Exception:

            try:
                os.remove(filepath)
            except:
                pass

# 屏蔽一些很多水印图片的网站

class UrlLoggingDownloader(ImageDownloader):

    def download(self,
                 task,
                 default_ext,
                 timeout=5,
                 max_retry=3,
                 overwrite=False,
                 **kwargs):

        # 获取图片URL
        file_url = task.get('file_url')

        # 某些版本URL保存在meta中
        if not file_url:
            file_url = task.get('meta', {}).get('img_url')

       
        # 屏蔽指定网站

        blocked_domains = [

            "dreamstime.com",
            "alamy.com",
            "gettyimages.com"

        ]

        if file_url:

            lower_url = file_url.lower()

            # 如果URL属于屏蔽网站
            for domain in blocked_domains:

                if domain in lower_url:

                    print(f"跳过屏蔽网站图片: {file_url}")

                    return None

            # 写入日志以找出水印多的网站

            save_dir = self.storage.root_dir

            log_path = os.path.join(save_dir, "url_log.txt")

            try:

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(file_url + "\n")

            except Exception as e:

                print(f"写入日志失败: {e}")

  
        # 下载

        return super().download(
            task,
            default_ext,
            timeout,
            max_retry,
            overwrite,
            **kwargs
        )

# 开始下载

for scene, keyword in scenes.items():

    save_dir = (
        f"datasets/"
        f"raw_images/{scene}"
    )

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*50)
    print(f"开始下载场景: {scene}")
    print(f"关键词: {keyword}")
    print("="*50)

    crawler = BingImageCrawler(

        downloader_threads=4,

        downloader_cls=UrlLoggingDownloader,

        storage={
            'root_dir': save_dir
        }
    )

    crawler.crawl(

        keyword=keyword,

        max_num=max_num[scene],

        filters={
            'type': 'photo',
            'size': 'large'
        }
    )

    print("\n开始清理图片...")

    clean_images(save_dir)

print("全部场景下载完成！")