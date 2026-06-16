from ultralytics import YOLO
import os
#本程序用于训练模型
# 模型与数据集路径

model_path = 'yolov8s.pt'

data_path = r'.\datasets\yolo_dataset\data.yaml'



# 主函数


if __name__ == '__main__':

    # 加载预训练模型

    model = YOLO(model_path)

    # 开始训练

    train_results = model.train(

        data=data_path,

        epochs=50,

        batch=16,

        device='0',

        workers=4,

        mixup=0.05,#exp-1 __刀枪容易融合背景。
        #mosaic=0.5,
        #close_mosaic=10,#训练最后10个epoch关闭 Mosaic

        # scale=0.25,#随机缩放目标
        # degrees=3,#随机旋转角度
        # translate=0.1,#平移增强
        # fliplr=0.5,#左右翻转

        project='',

        name='exp',

        save=True,

        plots=True,

        val=True,


    )

    # 获取 best.pt 路径

    best_model_path = os.path.join(
        'runs',
        'detect',
        'exp',
        'weights',
        'best.pt'
    )

    print("\n" + "=" * 60)
    print("训练完成")
    print(f"最优模型路径: {best_model_path}")
    print("=" * 60)

    # 重新加载训练后的 best.pt


    best_model = YOLO(best_model_path)

    # 模型验证

    metrics = best_model.val(

        data=data_path,


        batch=8,

 

        plots=True,

        save_json=True
    )

    # 输出评估指标

    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)

    # Precision
    print(f"Precision（精确率）: {metrics.box.mp:.4f}")

    # Recall
    print(f"Recall（召回率）: {metrics.box.mr:.4f}")

    # mAP50
    print(f"mAP@0.5: {metrics.box.map50:.4f}")

    # mAP50-95
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

    # 输出每个类别 AP

    print("\n各类别 AP50:")

    names = best_model.names

    ap50_list = metrics.box.ap50

    for i in range(len(ap50_list)):

        print(f"{names[i]} : {ap50_list[i]:.4f}")

    print("=" * 60)

    # 输出生成文件说明


    save_dir = metrics.save_dir

    print("\n所有结果保存路径:")
    print(save_dir)

    print("\n自动生成文件包括：")

    print("1. confusion_matrix.png")
    print("   混淆矩阵")

    print("2. PR_curve.png")
    print("   PR曲线")

    print("3. F1_curve.png")
    print("   F1曲线")

    print("4. P_curve.png")
    print("   Precision曲线")

    print("5. R_curve.png")
    print("   Recall曲线")

    print("6. results.png")
    print("   训练损失变化曲线")

    print("7. val_batch0_pred.jpg")
    print("   验证集预测结果")

    print("8. weights/best.pt")
    print("   最优训练模型")

    print("\n训练与验证全部完成！")