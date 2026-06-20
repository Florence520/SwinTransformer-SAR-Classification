# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox,
    QFileDialog, QProgressBar, QFrame, QToolButton, QMessageBox, QStackedWidget,
    QSizePolicy
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QProcess, QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于Swin Transformer的时序SAR地物分类研究与实现")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("QMainWindow { background-color: #f5f5f5; }")
        
        self.train_process = None
        self.infer_process = None
        
        # 主布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 侧边栏
        self.create_sidebar()
        
        # 主内容区
        self.create_content()
        
    def create_sidebar(self):
        """创建侧边栏"""
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(50)
        self.sidebar.setStyleSheet("background-color: #2c3e50;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # 展开按钮
        self.expand_btn = QToolButton()
        self.expand_btn.setText("☰")
        self.expand_btn.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 15px;
                color: white;
                font-size: 18px;
            }
            QToolButton:hover { background-color: #34495e; }
        """)
        self.expand_btn.clicked.connect(self.toggle_sidebar)
        sidebar_layout.addWidget(self.expand_btn)
        
        sidebar_layout.addSpacing(30)
        
        # 初始化导航按钮列表
        self.nav_btns = []
        
        # 主页按钮
        home_btn = self.create_nav_btn("⌂", "主页面")
        home_btn.clicked.connect(self.show_home)
        self.nav_btns.append(home_btn)
        sidebar_layout.addWidget(home_btn)
        
        train_btn = self.create_nav_btn("▷", "训练")
        train_btn.clicked.connect(self.show_train)
        self.nav_btns.append(train_btn)
        sidebar_layout.addWidget(train_btn)
        
        single_infer_btn = self.create_nav_btn("○", "单图推理")
        single_infer_btn.clicked.connect(self.show_single_infer)
        self.nav_btns.append(single_infer_btn)
        sidebar_layout.addWidget(single_infer_btn)
        
        infer_btn = self.create_nav_btn("◉", "批量推理")
        infer_btn.clicked.connect(self.show_infer)
        self.nav_btns.append(infer_btn)
        sidebar_layout.addWidget(infer_btn)
        
        sidebar_layout.addStretch()
        
        self.main_layout.addWidget(self.sidebar)
        
    def create_nav_btn(self, icon, tooltip):
        """创建导航按钮"""
        btn = QToolButton()
        btn.setText(icon)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 15px;
                color: white;
                font-size: 22px;
            }
            QToolButton:hover { background-color: #34495e; }
            QToolButton:pressed { background-color: #1abc9c; }
        """)
        return btn
        
    def toggle_sidebar(self):
        """切换侧边栏展开/收起"""
        if self.sidebar.width() == 50:
            self.sidebar.setFixedWidth(180)
            self.expand_btn.setText("◀")
            # 显示按钮文字
            for btn, text in zip(self.nav_btns, ["主页面", "训练", "单图推理", "批量推理"]):
                btn.setText(f"{btn.text()}  {text}")
        else:
            self.sidebar.setFixedWidth(50)
            self.expand_btn.setText("☰")
            # 只显示图标
            for btn in self.nav_btns:
                btn.setText(btn.text()[0])
        
    def create_content(self):
        """创建主内容区"""
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        
        # 页面栈
        self.page_stack = QStackedWidget()
        
        # 创建页面
        self.create_home_page()
        self.create_train_page()
        self.create_single_infer_page()
        self.create_infer_page()
        
        self.content_layout.addWidget(self.page_stack)
        self.main_layout.addWidget(self.content_area)
        
    def create_home_page(self):
        """创建主页面"""
        self.home_page = QWidget()
        layout = QVBoxLayout(self.home_page)
        layout.setContentsMargins(60, 60, 60, 60)
        
        # 标题
        title_label = QLabel("基于Swin Transformer的时序SAR地物分类研究与实现")
        title_label.setFont(QFont("微软雅黑", 22, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; padding: 30px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 信息卡片
        card = QFrame()
        card.setStyleSheet("background-color: white; border-radius: 15px; padding: 40px;")
        card_layout = QVBoxLayout(card)
        
        # 校徽
        logo_label = QLabel()
        logo_label.setStyleSheet("border-radius: 10px; background-color: #ffffff;")
        logo_label.setAlignment(Qt.AlignCenter)
        
        # 加载校徽图片
        logo_path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                max_logo_size = 260
                scaled_pixmap = pixmap.scaled(
                    max_logo_size, max_logo_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                logo_label.setScaledContents(True)
                logo_label.setFixedSize(scaled_pixmap.width() + 20, scaled_pixmap.height() + 20)
                logo_label.setStyleSheet("border-radius: 10px; background-color: #ffffff; padding: 10px;")
                logo_label.setPixmap(scaled_pixmap)
            else:
                logo_label.setFixedSize(150, 150)
                logo_label.setStyleSheet("background-color: #ecf0f1; border-radius: 10px;")
                logo_label.setText("<font size='24'>🏛️</font>")
        else:
            logo_label.setFixedSize(150, 150)
            logo_label.setStyleSheet("background-color: #ecf0f1; border-radius: 10px;")
            logo_label.setText("<font size='24'>🏛️</font>")
            
        card_layout.addWidget(logo_label, alignment=Qt.AlignCenter)
        
        card_layout.addSpacing(30)
        
        # 个人信息（竖列显示）
        info_items = [
            ("学号", "B22042203"),
            ("姓名", "王嘉丽"),
            ("指导老师", "倪康")
        ]
        
        for label, value in info_items:
            info_label = QLabel(f"{label}：{value}")
            info_label.setFont(QFont("微软雅黑", 14))
            info_label.setStyleSheet("color: #34495e; padding: 10px;")
            info_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(info_label)
        
        layout.addWidget(card)
        layout.addStretch()
        
        self.page_stack.addWidget(self.home_page)
        
    def create_train_page(self):
        """创建训练页面"""
        self.train_page = QWidget()
        layout = QVBoxLayout(self.train_page)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel("模型训练")
        title_label.setFont(QFont("微软雅黑", 18, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; padding-bottom: 20px;")
        layout.addWidget(title_label)
        
        # 配置区域
        config_frame = QFrame()
        config_frame.setStyleSheet("background-color: white; border-radius: 10px; padding: 20px;")
        config_layout = QVBoxLayout(config_frame)
        
        # 数据集选择
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("数据集路径："))
        self.train_data_edit = QLineEdit()
        self.train_data_edit.setPlaceholderText("请选择数据集目录")
        self.train_data_edit.setFixedWidth(400)
        data_layout.addWidget(self.train_data_edit)
        data_btn = QPushButton("浏览")
        data_btn.clicked.connect(self.select_train_data)
        data_layout.addWidget(data_btn)
        config_layout.addLayout(data_layout)
        
        # 配置文件选择
        cfg_layout = QHBoxLayout()
        cfg_layout.addWidget(QLabel("配置文件："))
        self.train_cfg_edit = QLineEdit()
        self.train_cfg_edit.setPlaceholderText("请选择配置文件")
        self.train_cfg_edit.setFixedWidth(400)
        cfg_layout.addWidget(self.train_cfg_edit)
        cfg_btn = QPushButton("浏览")
        cfg_btn.clicked.connect(self.select_train_cfg)
        cfg_layout.addWidget(cfg_btn)
        config_layout.addLayout(cfg_layout)
        
        # 训练结果保存路径
        result_layout = QHBoxLayout()
        result_layout.addWidget(QLabel("结果保存路径："))
        self.train_result_edit = QLineEdit()
        self.train_result_edit.setPlaceholderText("请选择训练结果保存位置")
        self.train_result_edit.setFixedWidth(400)
        result_layout.addWidget(self.train_result_edit)
        result_btn = QPushButton("浏览")
        result_btn.clicked.connect(self.select_train_result)
        result_layout.addWidget(result_btn)
        config_layout.addLayout(result_layout)
        
        layout.addWidget(config_frame)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.train_start_btn = QPushButton("开始训练")
        self.train_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 40px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        self.train_start_btn.clicked.connect(self.run_training)
        btn_layout.addWidget(self.train_start_btn)
        
        self.train_stop_btn = QPushButton("停止训练")
        self.train_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 12px 40px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.train_stop_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.train_stop_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 进度条
        self.train_progress = QProgressBar()
        self.train_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk { background-color: #27ae60; }
        """)
        layout.addWidget(self.train_progress)
        
        # 训练日志
        log_frame = QFrame()
        log_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        log_layout = QVBoxLayout(log_frame)
        
        log_label = QLabel("训练日志")
        log_label.setStyleSheet("padding: 10px; font-weight: bold; color: #2c3e50;")
        log_layout.addWidget(log_label)
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("background-color: #f8f9fa;")
        log_layout.addWidget(self.train_log)
        
        layout.addWidget(log_frame)
        
        self.page_stack.addWidget(self.train_page)
        
    def create_single_infer_page(self):
        """创建单图推理页面"""
        self.single_infer_page = QWidget()
        layout = QHBoxLayout(self.single_infer_page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 左侧：图片显示区域（正方形）
        left_frame = QFrame()
        left_frame.setStyleSheet("background-color: #d3d3d3; border-radius: 10px;")
        left_frame.setFixedSize(900, 900)  # 设置固定的正方形尺寸
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(0)
        
        # 结果图片显示区域（正方形）
        self.single_result_label = QLabel()
        self.single_result_label.setAlignment(Qt.AlignCenter)
        self.single_result_label.setStyleSheet("background-color: #f8f9fa; border-radius: 10px;")
        self.single_result_label.setText("推理结果将显示在这里")
        self.single_result_label.setScaledContents(True)  # 让图片自动缩放填充
        left_layout.addWidget(self.single_result_label, stretch=1)
        
        layout.addWidget(left_frame)
        
        # 右侧：控制面板（自动扩展占满剩余空间）
        right_frame = QFrame()
        right_frame.setStyleSheet("background-color: white; border-radius: 10px; padding: 20px;")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)
        
        # 页面标题
        title_label = QLabel("单项推理")
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        right_layout.addWidget(title_label)
        
        # 模型选择
        model_label = QLabel("选择模型：")
        model_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        right_layout.addWidget(model_label)
        
        self.single_model_combo = QComboBox()
        self.single_model_combo.setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 8px;")
        self.scan_single_models()
        right_layout.addWidget(self.single_model_combo)
        
        # 图片路径（放在模型选择下方）
        path_label = QLabel("文件路径：")
        path_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        right_layout.addWidget(path_label)
        
        self.single_mat_edit = QLineEdit()
        self.single_mat_edit.setPlaceholderText("请选择.mat数据文件")
        self.single_mat_edit.setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 8px;")
        right_layout.addWidget(self.single_mat_edit)
        
        # 选择图片路径按钮
        select_btn = QPushButton("选择文件")
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #bdc3c7;
                color: #2c3e50;
                border: none;
                padding: 8px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #95a5a6; }
        """)
        select_btn.clicked.connect(self.select_single_mat)
        right_layout.addWidget(select_btn)
        
        # 开始识别按钮
        self.single_infer_btn = QPushButton("开始识别")
        self.single_infer_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 40px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        self.single_infer_btn.clicked.connect(self.run_single_inference)
        right_layout.addWidget(self.single_infer_btn)
        
        right_layout.addSpacing(20)
        
        # 识别结果显示区域
        result_text_frame = QFrame()
        result_text_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 10px;")
        result_text_layout = QVBoxLayout(result_text_frame)
        
        self.single_result_text = QTextEdit()
        self.single_result_text.setReadOnly(True)
        self.single_result_text.setStyleSheet("background-color: transparent; border: none; font-size: 12px;")
        self.single_result_text.setFixedHeight(150)
        result_text_layout.addWidget(self.single_result_text)
        right_layout.addWidget(result_text_frame)
        
        right_layout.addStretch()
        
        layout.addWidget(right_frame)
        
        self.page_stack.addWidget(self.single_infer_page)
        
    def create_infer_page(self):
        """创建推理页面"""
        self.infer_page = QWidget()
        layout = QVBoxLayout(self.infer_page)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel("批量推理")
        title_label.setFont(QFont("微软雅黑", 18, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; padding-bottom: 20px;")
        layout.addWidget(title_label)
        
        # 配置区域
        config_frame = QFrame()
        config_frame.setStyleSheet("background-color: white; border-radius: 10px; padding: 20px;")
        config_layout = QVBoxLayout(config_frame)
        
        # 测试数据选择
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("推理数据集："))
        self.infer_data_edit = QLineEdit()
        self.infer_data_edit.setPlaceholderText("请选择测试数据目录")
        self.infer_data_edit.setFixedWidth(400)
        data_layout.addWidget(self.infer_data_edit)
        data_btn = QPushButton("浏览")
        data_btn.clicked.connect(self.select_infer_data)
        data_layout.addWidget(data_btn)
        config_layout.addLayout(data_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型："))
        self.infer_model_combo = QComboBox()
        self.infer_model_combo.setFixedWidth(400)
        self.infer_model_combo.addItem("请选择模型")
        # 扫描 work_dirs 中的 .pth 文件
        self.scan_models()
        model_layout.addWidget(self.infer_model_combo)
        model_btn = QPushButton("刷新")
        model_btn.clicked.connect(self.scan_models)
        model_layout.addWidget(model_btn)
        config_layout.addLayout(model_layout)
        
        # 推理结果保存路径
        result_layout = QHBoxLayout()
        result_layout.addWidget(QLabel("结果保存路径："))
        self.infer_result_edit = QLineEdit()
        self.infer_result_edit.setPlaceholderText("请选择推理结果保存位置")
        self.infer_result_edit.setFixedWidth(400)
        result_layout.addWidget(self.infer_result_edit)
        result_btn = QPushButton("浏览")
        result_btn.clicked.connect(self.select_infer_result)
        result_layout.addWidget(result_btn)
        config_layout.addLayout(result_layout)
        
        layout.addWidget(config_frame)
        
        # 开始推理按钮
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        self.infer_start_btn = QPushButton("开始推理")
        self.infer_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 40px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.infer_start_btn.clicked.connect(self.run_inference)
        btn_layout.addWidget(self.infer_start_btn)
        btn_layout.addStretch()
        layout.addWidget(btn_widget)
        
        # 进度条
        self.infer_progress = QProgressBar()
        self.infer_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk { background-color: #3498db; }
        """)
        layout.addWidget(self.infer_progress)
        
        # 推理结果
        result_frame = QFrame()
        result_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        result_layout = QVBoxLayout(result_frame)
        
        result_label = QLabel("推理结果")
        result_label.setStyleSheet("padding: 10px; font-weight: bold; color: #2c3e50;")
        result_layout.addWidget(result_label)
        
        self.infer_result = QTextEdit()
        self.infer_result.setReadOnly(True)
        self.infer_result.setStyleSheet("background-color: #f8f9fa;")
        result_layout.addWidget(self.infer_result)
        
        layout.addWidget(result_frame)
        
        self.page_stack.addWidget(self.infer_page)
        
    # 页面切换
    def show_home(self):
        self.page_stack.setCurrentWidget(self.home_page)
        
    def show_train(self):
        self.page_stack.setCurrentWidget(self.train_page)
        
    def show_single_infer(self):
        self.page_stack.setCurrentWidget(self.single_infer_page)
        
    def show_infer(self):
        self.page_stack.setCurrentWidget(self.infer_page)
        
    # 文件选择
    def select_train_data(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.train_data_edit.setText(path)
            
    def select_train_cfg(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "配置文件 (*.py)")
        if path:
            self.train_cfg_edit.setText(path)
            
    def select_infer_data(self):
        path = QFileDialog.getExistingDirectory(self, "选择测试数据目录")
        if path:
            self.infer_data_edit.setText(path)
            
    def select_train_result(self):
        path = QFileDialog.getExistingDirectory(self, "选择训练结果保存目录")
        if path:
            self.train_result_edit.setText(path)
            
    def select_infer_result(self):
        path = QFileDialog.getExistingDirectory(self, "选择推理结果保存目录")
        if path:
            self.infer_result_edit.setText(path)
            
    def select_single_mat(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择.mat文件", "", "MAT文件 (*.mat)")
        if path:
            self.single_mat_edit.setText(path)
            
    def scan_single_models(self):
        """扫描work_dirs中的模型文件"""
        work_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work_dirs")
        self.single_model_combo.clear()
        self.single_model_combo.addItem("请选择模型")
        
        if os.path.exists(work_dir):
            for root, dirs, files in os.walk(work_dir):
                for f in files:
                    if f.endswith(".pth"):
                        full_path = os.path.join(root, f)
                        # 使用文件名作为显示文本，完整路径作为用户数据
                        self.single_model_combo.addItem(f, full_path)
            
    def scan_models(self):
        """扫描work_dirs中的模型文件"""
        work_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work_dirs")
        self.infer_model_combo.clear()
        self.infer_model_combo.addItem("请选择模型")
        
        if os.path.exists(work_dir):
            for root, dirs, files in os.walk(work_dir):
                for f in files:
                    if f.endswith(".pth"):
                        full_path = os.path.join(root, f)
                        self.infer_model_combo.addItem(full_path)
                        
    # 训练和推理
    def run_training(self):
        if not self.train_data_edit.text() or not self.train_cfg_edit.text():
            QMessageBox.warning(self, "警告", "请选择数据集路径和配置文件")
            return

        if self.train_process is not None and self.train_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "警告", "训练已在进行中")
            return

        cfg_path = self.train_cfg_edit.text()
        dataset_root = self.train_data_edit.text()

        train_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools", "train.py"
        )
        if not os.path.exists(train_script):
            QMessageBox.critical(self, "错误", f"找不到训练脚本：{train_script}")
            return

        if self.train_result_edit.text():
            work_dir = self.train_result_edit.text()
        else:
            work_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "work_dirs",
                os.path.splitext(os.path.basename(cfg_path))[0]
            )
        os.makedirs(work_dir, exist_ok=True)

        self.train_log.clear()
        self.train_log.append("开始训练...")
        self.train_log.append(f"数据集: {dataset_root}")
        self.train_log.append(f"配置文件: {cfg_path}")

        arguments = [
            train_script,
            "--config", cfg_path,
            "--work-dir", work_dir,
            "--cfg-options", f"data_root={dataset_root}"
        ]

        self.train_process = QProcess(self)
        self.train_process.setWorkingDirectory(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.train_process.readyReadStandardOutput.connect(self.handle_train_output)
        self.train_process.readyReadStandardError.connect(self.handle_train_error)
        self.train_process.finished.connect(self.on_train_finished)

        self.train_start_btn.setEnabled(False)
        self.train_stop_btn.setEnabled(True)
        self.train_progress.setValue(0)

        self.train_process.start(sys.executable, arguments)
        if not self.train_process.waitForStarted(3000):
            QMessageBox.critical(self, "错误", "训练进程启动失败")
            self.train_start_btn.setEnabled(True)
            self.train_stop_btn.setEnabled(False)
            self.train_process = None

    def stop_training(self):
        if self.train_process is not None and self.train_process.state() != QProcess.NotRunning:
            self.train_process.terminate()
            if not self.train_process.waitForFinished(3000):
                self.train_process.kill()
            self.train_log.append("训练已停止")
        self.train_start_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)

    def handle_train_output(self):
        if not self.train_process:
            return
        text = self.train_process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
        if text:
            self.train_log.append(text)
            self.update_train_progress(text)

    def handle_train_error(self):
        if not self.train_process:
            return
        text = self.train_process.readAllStandardError().data().decode('utf-8', errors='ignore')
        if text:
            self.train_log.append(text)

    def update_train_progress(self, text):
        import re
        match = re.search(r'Iter\\(train\\) \[\s*([0-9]+)\/([0-9]+)\]', text)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                value = int(current / total * 100)
                self.train_progress.setValue(min(max(value, 0), 100))

    def on_train_finished(self):
        exit_code = self.train_process.exitCode() if self.train_process else -1
        self.train_log.append("训练完成！" if exit_code == 0 else f"训练进程结束，退出码：{exit_code}")
        self.train_progress.setValue(100)
        self.train_start_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.train_process = None

    def run_inference(self):
        if not self.infer_data_edit.text() or self.infer_model_combo.currentIndex() == 0:
            QMessageBox.warning(self, "警告", "请选择推理数据集和模型文件")
            return
            
        cfg_path = self.train_cfg_edit.text() if self.train_cfg_edit.text() else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "swinUnetr", "SwinUNETR_224x224_40k_slovenia_s1.py"
        )
        checkpoint_path = self.infer_model_combo.currentText()
        dataset_root = self.infer_data_edit.text()

        test_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools", "test.py"
        )
        if not os.path.exists(test_script):
            QMessageBox.critical(self, "错误", f"找不到推理脚本：{test_script}")
            return

        if self.infer_result_edit.text():
            out_dir = self.infer_result_edit.text()
        else:
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "work_dirs", "infer_results"
            )
        os.makedirs(out_dir, exist_ok=True)

        self.infer_result.clear()
        self.infer_result.append("开始推理...")
        self.infer_result.append(f"数据集: {dataset_root}")
        self.infer_result.append(f"模型: {checkpoint_path}")

        arguments = [
            test_script,
            "--config", cfg_path,
            "--checkpoint", checkpoint_path,
            "--out", out_dir,
            "--cfg-options", f"data_root={dataset_root}"
        ]

        self.infer_process = QProcess(self)
        self.infer_process.setWorkingDirectory(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.infer_process.readyReadStandardOutput.connect(self.handle_infer_output)
        self.infer_process.readyReadStandardError.connect(self.handle_infer_error)
        self.infer_process.finished.connect(self.on_infer_finished)

        self.infer_start_btn.setEnabled(False)
        self.infer_progress.setValue(0)
        self.infer_process.start(sys.executable, arguments)
        if not self.infer_process.waitForStarted(3000):
            QMessageBox.critical(self, "错误", "推理进程启动失败")
            self.infer_start_btn.setEnabled(True)
            self.infer_process = None

    def handle_infer_output(self):
        if not self.infer_process:
            return
        text = self.infer_process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
        if text:
            self.infer_result.append(text)
            self.update_infer_progress(text)

    def handle_infer_error(self):
        if not self.infer_process:
            return
        text = self.infer_process.readAllStandardError().data().decode('utf-8', errors='ignore')
        if text:
            self.infer_result.append(text)

    def update_infer_progress(self, text):
        import re
        match = re.search(r'Iter\\(test\\) \[\s*([0-9]+)\/([0-9]+)\]', text)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                value = int(current / total * 100)
                self.infer_progress.setValue(min(max(value, 0), 100))

    def on_infer_finished(self):
        exit_code = self.infer_process.exitCode() if self.infer_process else -1
        self.infer_result.append("推理完成！" if exit_code == 0 else f"推理进程结束，退出码：{exit_code}")
        self.infer_progress.setValue(100)
        self.infer_start_btn.setEnabled(True)
        self.infer_process = None

    def run_single_inference(self):
        """执行单图推理"""
        mat_path = self.single_mat_edit.text()
        if not mat_path:
            QMessageBox.warning(self, "警告", "请选择.mat数据文件")
            return
            
        if self.single_model_combo.currentIndex() == 0:
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return
            
        checkpoint_path = self.single_model_combo.currentData()
        # 如果没有数据（比如选择的是"请选择模型"），使用当前文本作为备选
        if checkpoint_path is None:
            checkpoint_path = self.single_model_combo.currentText()
        
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "swinUnetr", "SwinUNETR_224x224_40k_slovenia_s1.py"
        )
        
        self.single_infer_btn.setEnabled(False)
        self.single_result_label.setText("正在推理，请稍候...")
        
        # 固定输出目录
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "testval", "testimages"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据输入文件名生成输出文件名
        input_filename = os.path.basename(mat_path)
        result_img_path = os.path.join(output_dir, os.path.splitext(input_filename)[0] + "_pred.png")
        
        infer_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools", "infer_single.py"
        )
        
        arguments = [
            infer_script,
            "--mat", mat_path,
            "--config", cfg_path,
            "--checkpoint", checkpoint_path,
            "--output", result_img_path
        ]
        
        self.single_infer_process = QProcess(self)
        self.single_infer_process.setWorkingDirectory(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.single_infer_process.readyReadStandardOutput.connect(self.handle_single_infer_output)
        self.single_infer_process.readyReadStandardError.connect(self.handle_single_infer_error)
        self.single_infer_process.finished.connect(lambda exit_code, status: self.on_single_infer_finished(exit_code, result_img_path))
        
        self.single_infer_process.start(sys.executable, arguments)
        if not self.single_infer_process.waitForStarted(3000):
            QMessageBox.critical(self, "错误", "单图推理进程启动失败")
            self.single_infer_btn.setEnabled(True)
            self.single_infer_process = None
            
    def handle_single_infer_output(self):
        if not self.single_infer_process:
            return
        text = self.single_infer_process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
        if text:
            print("单图推理输出:", text)
            
    def handle_single_infer_error(self):
        if not self.single_infer_process:
            return
        text = self.single_infer_process.readAllStandardError().data().decode('utf-8', errors='ignore')
        if text:
            print("单图推理错误:", text)
            
    def on_single_infer_finished(self, exit_code, result_img_path):
        try:
            if exit_code == 0 and os.path.exists(result_img_path):
                pixmap = QPixmap(result_img_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        500, 400,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.single_result_label.setPixmap(scaled_pixmap)
                    
                    # 更新结果文本区域
                    result_text = f"""识别结果:
识别完成！
结果已保存到:
{result_img_path}"""
                    self.single_result_text.setText(result_text)
                else:
                    self.single_result_label.setText("推理结果图片加载失败")
                    self.single_result_text.setText("推理结果图片加载失败")
            else:
                self.single_result_label.setText("推理失败，请检查输入和模型")
                self.single_result_text.setText("识别失败，请检查输入和模型")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理失败：{str(e)}")
            self.single_result_label.setText("推理失败，请检查输入和模型")
            self.single_result_text.setText(f"识别失败：{str(e)}")
        finally:
            self.single_infer_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())