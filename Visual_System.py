# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QTableWidgetItem,
                             QStyledItemDelegate, QHeaderView)
import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime
import sys

#本程序为检测系统

class CenteredDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 900)
        MainWindow.setWindowTitle("YOLOv8 手持物品检测系统")   # 修改标题

        # 设置窗口图标
        if hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, 'icon.ico')
        else:
            icon_path = 'icon.ico'
        if os.path.exists(icon_path):
            MainWindow.setWindowIcon(QIcon(icon_path))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 主布局（保持不变）
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(15)

        # 左侧布局 (图像显示)
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.setSpacing(15)

        # 原始图像组
        self.original_group = QtWidgets.QGroupBox("原始图像")
        self.original_group.setMinimumHeight(400)
        self.original_img_label = QtWidgets.QLabel()
        self.original_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_img_label.setText("等待加载图像...")
        self.original_img_label.setStyleSheet("background-color: #F0F0F0; border: 1px solid #CCCCCC;")

        original_layout = QtWidgets.QVBoxLayout()
        original_layout.addWidget(self.original_img_label)
        self.original_group.setLayout(original_layout)
        self.left_layout.addWidget(self.original_group)

        # 检测结果图像组
        self.result_group = QtWidgets.QGroupBox("检测结果")
        self.result_group.setMinimumHeight(400)
        self.result_img_label = QtWidgets.QLabel()
        self.result_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_img_label.setText("检测结果将显示在这里")
        self.result_img_label.setStyleSheet("background-color: #F0F0F0; border: 1px solid #CCCCCC;")

        result_layout = QtWidgets.QVBoxLayout()
        result_layout.addWidget(self.result_img_label)
        self.result_group.setLayout(result_layout)
        self.left_layout.addWidget(self.result_group)

        self.main_layout.addLayout(self.left_layout, stretch=3)

        # 右侧布局（控制面板）
        self.right_layout = QtWidgets.QVBoxLayout()
        self.right_layout.setSpacing(15)

        # 模型选择组
        self.model_group = QtWidgets.QGroupBox("模型设置")
        self.model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.model_layout = QtWidgets.QVBoxLayout()

        # 模型选择（注意路径中的反斜杠需要转义，这里保持原样）
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems([".\\runs\\detect\\exp\\weights\\best.pt"])   # 建议改为绝对路径或正斜杠
        self.model_combo.setCurrentIndex(0)

        # 加载模型按钮
        self.load_model_btn = QtWidgets.QPushButton(" 加载模型")
        self.load_model_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_model_btn.setStyleSheet(
            "QPushButton { padding: 8px; background-color: #4CAF50; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )

        self.model_layout.addWidget(self.model_combo)
        self.model_layout.addWidget(self.load_model_btn)
        self.model_group.setLayout(self.model_layout)
        self.right_layout.addWidget(self.model_group)

        # 参数设置组
        self.param_group = QtWidgets.QGroupBox("检测参数")
        self.param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.param_layout = QtWidgets.QFormLayout()
        self.param_layout.setLabelAlignment(Qt.AlignLeft)
        self.param_layout.setFormAlignment(Qt.AlignLeft)
        self.param_layout.setVerticalSpacing(15)

        # 置信度滑块
        self.conf_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_value = QtWidgets.QLabel("0.25")
        self.conf_value.setAlignment(Qt.AlignCenter)
        self.conf_value.setStyleSheet("font-weight: bold; color: #2196F3;")

        # IoU滑块
        self.iou_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)
        self.iou_value = QtWidgets.QLabel("0.45")
        self.iou_value.setAlignment(Qt.AlignCenter)
        self.iou_value.setStyleSheet("font-weight: bold; color: #2196F3;")

        self.param_layout.addRow("置信度阈值:", self.conf_slider)
        self.param_layout.addRow("当前值:", self.conf_value)
        self.param_layout.addRow(QtWidgets.QLabel(""))  # 空行
        self.param_layout.addRow("IoU阈值:", self.iou_slider)
        self.param_layout.addRow("当前值:", self.iou_value)

        self.param_group.setLayout(self.param_layout)
        self.right_layout.addWidget(self.param_group)

        # 功能按钮组
        self.func_group = QtWidgets.QGroupBox("检测功能")
        self.func_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.func_layout = QtWidgets.QVBoxLayout()
        self.func_layout.setSpacing(10)

        # 图片检测按钮
        self.image_btn = QtWidgets.QPushButton(" 图片检测")
        self.image_btn.setIcon(QIcon.fromTheme("image-x-generic"))

        # 视频检测按钮
        self.video_btn = QtWidgets.QPushButton(" 视频检测")
        self.video_btn.setIcon(QIcon.fromTheme("video-x-generic"))

        # 摄像头检测按钮
        self.camera_btn = QtWidgets.QPushButton(" 摄像头检测")
        self.camera_btn.setIcon(QIcon.fromTheme("camera-web"))

        # 停止检测按钮
        self.stop_btn = QtWidgets.QPushButton(" 停止检测")
        self.stop_btn.setIcon(QIcon.fromTheme("process-stop"))
        self.stop_btn.setEnabled(False)

        # 保存结果按钮
        self.save_btn = QtWidgets.QPushButton(" 保存结果")
        self.save_btn.setIcon(QIcon.fromTheme("document-save"))
        self.save_btn.setEnabled(False)

        # 设置按钮样式
        button_style = """
        QPushButton {
            padding: 10px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            text-align: left;
        }
        QPushButton:hover {
            background-color: #0b7dda;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
        """

        for btn in [self.image_btn, self.video_btn, self.camera_btn,
                    self.stop_btn, self.save_btn]:
            btn.setStyleSheet(button_style)
            self.func_layout.addWidget(btn)

        self.func_group.setLayout(self.func_layout)
        self.right_layout.addWidget(self.func_group)

        # 检测结果表格组
        self.table_group = QtWidgets.QGroupBox("检测结果详情")
        self.table_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.table_layout = QtWidgets.QVBoxLayout()

        self.result_table = QtWidgets.QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["类别", "置信度", "左上坐标", "右下坐标"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.result_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # 设置表格样式
        self.result_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                alternate-background-color: #f5f5f5;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 5px;
                border: none;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)

        # 设置居中代理
        delegate = CenteredDelegate(self.result_table)
        self.result_table.setItemDelegate(delegate)

        self.table_layout.addWidget(self.result_table)
        self.table_group.setLayout(self.table_layout)
        self.right_layout.addWidget(self.table_group, stretch=1)

        self.main_layout.addLayout(self.right_layout, stretch=1)

        MainWindow.setCentralWidget(self.centralwidget)

        # 状态栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("QStatusBar { border-top: 1px solid #c0c0c0; }")
        MainWindow.setStatusBar(self.statusbar)

        # 初始化变量
        self.model = None
        self.names = None
        self.cap = None
        self.timer = QTimer()
        self.is_camera_running = False
        self.current_image = None
        self.current_result = None
        self.video_writer = None
        self.output_path = "output"
        self.video_fps = 30

        # 创建输出目录
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # 连接信号槽
        self.load_model_btn.clicked.connect(self.load_model)
        self.image_btn.clicked.connect(self.detect_image)
        self.video_btn.clicked.connect(self.detect_video)
        self.camera_btn.clicked.connect(self.detect_camera)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.save_btn.clicked.connect(self.save_result)
        self.conf_slider.valueChanged.connect(self.update_conf_value)
        self.iou_slider.valueChanged.connect(self.update_iou_value)
        self.timer.timeout.connect(self.update_camera_frame)

        # 设置全局样式
        self.set_style()

    def set_style(self):
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QLabel {
            color: #333333;
        }
        QComboBox {
            padding: 5px;
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 16px;
            height: 16px;
            margin: -5px 0;
            background: #2196F3;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal {
            background: #2196F3;
            border-radius: 3px;
        }
        """
        self.centralwidget.setStyleSheet(style)

   
    def count_objects(self, results):
        """
        统计检测结果中各类目标的数量（人、雨伞、刀、枪）
        返回四个整数：persons, umbrellas, knives, guns
        """
        persons = 0
        umbrellas = 0
        knives = 0
        guns = 0
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = self.names[cls_id]
                if class_name == "person":
                    persons += 1
                elif class_name == "umbrella":          # 雨伞
                    umbrellas += 1
                elif class_name == "knife":
                    knives += 1
                elif class_name == "gun":
                    guns += 1
        return persons, umbrellas, knives, guns

    def display_image(self, img, label):
        """将 OpenCV (RGB) 图像显示在 QLabel 上"""
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        # 保持宽高比缩放
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))

    def load_model(self):
        model_name = self.model_combo.currentText().split(" ")[0]
        try:
            self.model = YOLO(model_name)
            self.names = self.model.names   # 获取类别映射
            self.statusbar.showMessage(f"模型 {model_name} 加载成功", 3000)
            self.image_btn.setEnabled(True)
            self.video_btn.setEnabled(True)
            self.camera_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(None, "错误", f"模型加载失败: {str(e)}")

    def update_conf_value(self):
        conf = self.conf_slider.value() / 100
        self.conf_value.setText(f"{conf:.2f}")

    def update_iou_value(self):
        iou = self.iou_slider.value() / 100
        self.iou_value.setText(f"{iou:.2f}")

    def detect_image(self):
        if self.model is None:
            QMessageBox.warning(None, "警告", "请先加载模型")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            None, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if file_path:
            try:
                # 读取图片
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 显示原始图片
                self.display_image(img, self.original_img_label)
                self.current_image = img.copy()

                # 检测参数
                conf = self.conf_slider.value() / 100
                iou = self.iou_slider.value() / 100

                self.statusbar.showMessage("正在检测图片...")
                QtWidgets.QApplication.processEvents()

                results = self.model.predict(img, conf=conf, iou=iou)
                result_img = results[0].plot()   # 基础检测图

                # ========== 修改点：统计各类数量，绘制文字 ==========
                persons, umbrellas, knives, guns = self.count_objects(results[0])

                # 在结果图上绘制统计信息
                cv2.putText(result_img,
                            f"Person:{persons}  Umbrella:{umbrellas}  Knife:{knives}  Gun:{guns}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0), 2)
  

                # 显示检测结果
                self.display_image(result_img, self.result_img_label)
                self.current_result = result_img.copy()

                # 更新结果表格
                self.update_result_table(results[0])

                self.save_btn.setEnabled(True)
                self.statusbar.showMessage(f"图片检测完成: {os.path.basename(file_path)}", 3000)

            except Exception as e:
                QMessageBox.critical(None, "错误", f"图片检测失败: {str(e)}")
                self.statusbar.showMessage("图片检测失败", 3000)

    def detect_video(self):
        if self.model is None:
            QMessageBox.warning(None, "警告", "请先加载模型")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            None, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if file_path:
            try:
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    raise Exception("无法打开视频文件")

                # 获取视频信息
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # 创建视频写入器
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.output_path, f"output_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(output_file, fourcc, self.video_fps, (width, height))

                # 启用停止按钮，禁用其他按钮
                self.stop_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                self.image_btn.setEnabled(False)
                self.video_btn.setEnabled(False)
                self.camera_btn.setEnabled(False)

                # 开始定时处理帧
                self.timer.start(int(1000 / self.video_fps))   # 按视频帧率间隔
                self.statusbar.showMessage(f"正在处理视频: {os.path.basename(file_path)}...")

            except Exception as e:
                QMessageBox.critical(None, "错误", f"视频检测失败: {str(e)}")
                self.statusbar.showMessage("视频检测失败", 3000)

    def detect_camera(self):
        """打开摄像头检测"""
        if self.model is None:
            QMessageBox.warning(None, "警告", "请先加载模型")
            return
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
            self.is_camera_running = True
            self.video_fps = 30   # 默认30fps

            # 禁用按钮
            self.stop_btn.setEnabled(True)
            self.image_btn.setEnabled(False)
            self.video_btn.setEnabled(False)
            self.camera_btn.setEnabled(False)
            self.save_btn.setEnabled(False)

            self.timer.start(30)
            self.statusbar.showMessage("摄像头已开启，正在进行实时检测...")
        except Exception as e:
            QMessageBox.critical(None, "错误", f"摄像头启动失败: {str(e)}")

    def update_camera_frame(self):
        """定时器触发的帧处理，用于视频/摄像头"""
        if self.cap is None or not self.cap.isOpened():
            self.stop_detection()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        # 转换为 RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 显示原始帧
        self.display_image(img_rgb, self.original_img_label)
        self.current_image = img_rgb.copy()

        # 检测
        conf = self.conf_slider.value() / 100
        iou = self.iou_slider.value() / 100
        results = self.model.predict(img_rgb, conf=conf, iou=iou)
        result_img = results[0].plot()

        
        persons, umbrellas, knives, guns = self.count_objects(results[0])

        cv2.putText(result_img,
                    f"Person:{persons}  Umbrella:{umbrellas}  Knife:{knives}  Gun:{guns}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0), 2)

        # 显示结果
        self.display_image(result_img, self.result_img_label)
        self.current_result = result_img.copy()

        # 更新表格
        self.update_result_table(results[0])

        # 写入视频文件（如果开启）
        if self.video_writer is not None and not self.is_camera_running:
            # 写入 BGR 帧
            self.video_writer.write(cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    def stop_detection(self):
        """停止检测"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.is_camera_running = False

        # 恢复按钮状态
        self.stop_btn.setEnabled(False)
        self.image_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.camera_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.statusbar.showMessage("检测已停止", 3000)

    def save_result(self):
        """保存当前检测结果图"""
        if self.current_result is None:
            QMessageBox.warning(None, "警告", "没有可保存的结果")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(
            None, "保存结果", f"result_{timestamp}.jpg",
            "JPEG (*.jpg);;PNG (*.png);;所有文件 (*)"
        )
        if file_path:
            # 保存 OpenCV 格式（BGR）
            cv2.imwrite(file_path, cv2.cvtColor(self.current_result, cv2.COLOR_RGB2BGR))
            self.statusbar.showMessage(f"结果已保存至: {file_path}", 3000)

    def update_result_table(self, result):
        """更新右侧检测结果表格"""
        self.result_table.setRowCount(0)
        if result.boxes is not None:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                class_name = self.names[cls_id]   # 使用类别映射
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                row_pos = self.result_table.rowCount()
                self.result_table.insertRow(row_pos)

                # 类别
                item = QTableWidgetItem(class_name)
                self.result_table.setItem(row_pos, 0, item)
                # 置信度
                item = QTableWidgetItem(f"{conf:.3f}")
                self.result_table.setItem(row_pos, 1, item)
                # 左上
                item = QTableWidgetItem(f"({int(xyxy[0])}, {int(xyxy[1])})")
                self.result_table.setItem(row_pos, 2, item)
                # 右下
                item = QTableWidgetItem(f"({int(xyxy[2])}, {int(xyxy[3])})")
                self.result_table.setItem(row_pos, 3, item)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())