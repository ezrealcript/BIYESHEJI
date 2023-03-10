import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QLabel, QMessageBox
from PyQt5 import QtGui

BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))

# 第三版：创建线程进行目标检测处理，结果处理后返回主窗口
# 处理时间信息在表单中显示
# 处理后的图片单独显示

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 控件
        self.txt_asin = None  # 待检测图像位置输入行
        self.txt_asin1 = None  # checkpoint位置输入行
        self.pic_show_label = None  # 图片1显示位置
        self.pic_show_label2 = None
        self.pic_show_label3 = None
        self.table_widget = None  # 检测结果显示表单
        self.table_mid_widget = None  # 维数检测表单
        self.label_status = None  # 状态标签

        self.isRun = 1

        # 窗体大小
        self.setWindowTitle("高光谱目标检测")
        self.resize(1200, 800)

        # 窗体位置
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

        # 布局
        # 创建垂直方向的布局
        layout = QVBoxLayout()

        # 1.
        layout.addLayout(self.init_header())

        # 2.
        layout.addLayout(self.init_form())

        layout.addLayout(self.init_form_mid())

        # 3.
        layout.addLayout(self.init_table())

        layout.addLayout(self.init_table_mid())

        layout.addStretch()
        # 4.
        layout.addLayout(self.init_footer())

        # 加入弹簧
        # layout.addStretch()
        # 窗体加入布局
        self.setLayout(layout)

    # 头部布局
    # 按钮：开始检测，停止，重新初始化，当前状态
    def init_header(self):
        # 1.
        header_layout = QHBoxLayout()
        # 1.1 创建按钮
        btn_start = QPushButton("开始检测")
        header_layout.addWidget(btn_start)
        btn_start.clicked.connect(self.event_start_click)

        btn_stop = QPushButton("停止")
        header_layout.addWidget(btn_stop)
        btn_stop.clicked.connect(self.event_stop_click)

        btn_tip = QPushButton("说明")
        header_layout.addWidget(btn_tip)
        btn_tip.clicked.connect(self.event_tip_click)

        btn_reset = QPushButton("重新初始化")
        btn_reset.clicked.connect(self.event_reset_click)
        header_layout.addWidget(btn_reset)

        header_layout.addStretch()

        label_status = QLabel("未检测", self)
        header_layout.addWidget(label_status)
        self.label_status = label_status

        return header_layout

    def init_form(self):
        form_layout = QHBoxLayout()
        # 2.1 输入框
        txt_asin = QLineEdit()
        form_layout.addWidget(txt_asin)
        txt_asin.setPlaceholderText("请输入待检测的图像")
        # txt_asin.isClearButtonEnabled()
        self.txt_asin = txt_asin
        # 2.2 按钮
        btn_add = QPushButton("自动填入")
        btn_add.clicked.connect(self.event_add_click)
        form_layout.addWidget(btn_add)
        return form_layout

    def init_form_mid(self):
        form_mid_layout = QHBoxLayout()
        # 2.1 输入框
        txt_asin1 = QLineEdit()
        form_mid_layout.addWidget(txt_asin1)
        txt_asin1.setPlaceholderText("请输入checkpoint")
        self.txt_asin1 = txt_asin1
        # 2.2 按钮
        btn_add = QPushButton("自动填入")
        btn_add.clicked.connect(self.event_add1_click)
        form_mid_layout.addWidget(btn_add)
        return form_mid_layout

    def init_table(self):
        table_layout = QHBoxLayout()
        # 3.1 创建表格
        # 检测阶段，时间，维数，正确率，trainsetELBO1，trainsetELBO2
        table_widget = QTableWidget(0, 3)
        item = QTableWidgetItem()
        item.setText("检测阶段")
        table_widget.setHorizontalHeaderItem(0, item)

        item = QTableWidgetItem()
        item.setText("时间")
        table_widget.setHorizontalHeaderItem(1, item)

        # item = QTableWidgetItem()
        # item.setText("维数")
        # table_widget.setHorizontalHeaderItem(2, item)

        item = QTableWidgetItem()
        item.setText("正确率")
        table_widget.setHorizontalHeaderItem(2, item)

        # item = QTableWidgetItem()
        # item.setText("trainsetELBO1")
        # table_widget.setHorizontalHeaderItem(4, item)

        # item = QTableWidgetItem()
        # item.setText("trainsetELBO2")
        # table_widget.setHorizontalHeaderItem(5, item)
        # table_widget.setColumnWidth(0, 100)

        self.table_widget = table_widget

        # 3.2 初始化表格
        # 读取数据文件
        # import json
        # file_path = os.path.join(BASE_DIR, "db", "db.json")
        # with open(file_path, mode="r", encoding="utf-8") as f:
        #     data = f.read()
        # data_list = json.loads(data)

        #current_row_count = table_widget.rowCount()  # 获取当前表格有多少行
        # 展示数据文件
        # for row_list in data_list:
        #     table_widget.insertRow(current_row_count)  # 新增一行
        #     # 写数据
        #     for i, ele in enumerate(row_list):
        #         cell = QTableWidgetItem(str(ele))
        #         table_widget.setItem(current_row_count, i, cell)
        #     current_row_count += 1

        table_layout.addWidget(table_widget)
        return table_layout

    def init_table_mid(self):
        # 显示各个维检测时间的表格
        table_mid_layout = QHBoxLayout()
        table_mid_widget = QTableWidget(0, 4)

        item = QTableWidgetItem()
        item.setText("维数")
        table_mid_widget.setHorizontalHeaderItem(0, item)

        item = QTableWidgetItem()
        item.setText("ELBO1")
        table_mid_widget.setHorizontalHeaderItem(1, item)
        table_mid_widget.setColumnWidth(1, 200)

        item = QTableWidgetItem()
        item.setText("ELBO2")
        table_mid_widget.setHorizontalHeaderItem(2, item)
        table_mid_widget.setColumnWidth(2, 200)

        item = QTableWidgetItem()
        item.setText("时间")
        table_mid_widget.setHorizontalHeaderItem(3, item)

        self.table_mid_widget = table_mid_widget

        table_mid_layout.addWidget(table_mid_widget)
        return table_mid_layout
        pass

    def init_footer(self):
        footer_layout = QHBoxLayout()
        # # 4.1 标签
        # label_status = QLabel("未检测", self)
        # footer_layout.addWidget(label_status)
        #
        # # 4.2 按钮
        # btn_reset = QPushButton("重新初始化")
        # btn_reset.clicked.connect(self.event_reset_click)
        # footer_layout.addWidget(btn_reset)
        #
        # footer_layout.addStretch()
        # 4.3 加载图片
        # 图片路径
        # img_path = "ZCEM.jpg"
        # 设置展示控件
        pic_show_label = QLabel("等待显示图片", self)
        # 设置窗口尺寸
        pic_show_label.resize(500, 500)
        # 加载图片,并自定义图片展示尺寸
        # image = QtGui.QPixmap(img_path).scaled(400, 400)
        # 显示图片
        # pic_show_label.setPixmap(image)
        footer_layout.addWidget(pic_show_label)
        self.pic_show_label = pic_show_label

        pic_show_label2 = QLabel("等待显示图片", self)
        pic_show_label2.resize(500, 500)
        footer_layout.addWidget(pic_show_label2)
        self.pic_show_label2 = pic_show_label2

        pic_show_label3 = QLabel("等待显示图片", self)
        pic_show_label3.resize(500, 500)
        footer_layout.addWidget(pic_show_label3)
        self.pic_show_label3 = pic_show_label3

        return footer_layout



    # 点击 开始检测 按钮
    def event_start_click(self):
        # 1.获取输入框中的内容
        #text = self.txt_asin.text()
        #print(text)
        # 2.加入表格中
        #new_row_list = ["1", "2", "3", "4", "5", "6", "7", "8"]
        current_row_count = self.table_widget.rowCount()  # 当前表格有多少行
        self.table_widget.insertRow(current_row_count)  # 在表格下添加一行
        # 写入数据
        #for i, ele in enumerate(new_row_list):
            #cell = QTableWidgetItem(str(ele))
            #self.table_widget.setItem(current_row_count, i, cell)
        self.label_status.setText("检测中")
        # 3.发送请求自动获取标题（爬虫获取数据）
        # 不能在主线程中做爬虫的事，创建一个线程去做爬虫，爬取数据后再更新到窗体应用（信号）
        from utils.threads2 import NewTaskThread
        thread = NewTaskThread(current_row_count, self)
        thread.success.connect(self.init_task_success_callback)
        thread.error.connect(self.init_task_error_callback)
        thread.success_cem.connect(self.init_task_success_cem_callback)
        thread.success_zcem.connect(self.init_task_success_zcem_callback)
        thread.success_cvae.connect(self.init_task_success_cvae_callback)
        thread.success_epoh.connect(self.init_task_success_epoh_callback)
        thread.start()
        pass

    def init_task_success_callback(self, row_index, asin, title, url):
        # 更新窗体显示的数据
        # print(row_index, asin, title, url)

        # 更新标题列
        #cell_title = QTableWidgetItem(title)
        #self.table_widget.setItem(row_index, 1, cell_title)

        # 输入框清空
        self.txt_asin.clear()

        # 将处理后的图片进行展示
        img_path = "ZCEM.jpg"
        image = QtGui.QPixmap(img_path).scaled(400, 400)
        self.pic_show_label.setPixmap(image)

        img_path = "CEMcujaince.jpg"
        image = QtGui.QPixmap(img_path).scaled(400, 400)
        self.pic_show_label2.setPixmap(image)

        img_path = "CEM CVAE.jpg"
        image = QtGui.QPixmap(img_path).scaled(400, 400)
        self.pic_show_label3.setPixmap(image)
        self.label_status.setText("检测完成")

        pass

    def init_task_error_callback(self, row_index, asin, title, url):
        pass

    def init_task_success_cem_callback(self, row_index, name, time, auc):
        cell_title = QTableWidgetItem(name)
        self.table_widget.setItem(row_index, 0, cell_title)
        cell_time = QTableWidgetItem(str(time))
        self.table_widget.setItem(row_index, 1, cell_time)
        cell_auc = QTableWidgetItem(auc)
        self.table_widget.setItem(row_index, 2, cell_auc)
        pass

    def init_task_success_zcem_callback(self, name, time, auc):
        row_index = self.table_widget.rowCount()
        self.table_widget.insertRow(row_index)
        cell_title = QTableWidgetItem(name)
        self.table_widget.setItem(row_index, 0, cell_title)
        cell_time = QTableWidgetItem(str(time))
        self.table_widget.setItem(row_index, 1, cell_time)
        cell_auc = QTableWidgetItem(str(auc))
        self.table_widget.setItem(row_index, 2, cell_auc)
        pass

    def init_task_success_cvae_callback(self, name, time, auc):
        row_index = self.table_widget.rowCount()
        self.table_widget.insertRow(row_index)
        cell_title = QTableWidgetItem(name)
        self.table_widget.setItem(row_index, 0, cell_title)
        cell_time = QTableWidgetItem(str(time))
        self.table_widget.setItem(row_index, 1, cell_time)
        cell_auc = QTableWidgetItem(str(auc))
        self.table_widget.setItem(row_index, 2, cell_auc)
        pass

    def init_task_success_epoh_callback(self, epoh, elco1, elco2, time):
        row_index = self.table_mid_widget.rowCount()
        self.table_mid_widget.insertRow(row_index)
        cell_epoh = QTableWidgetItem(epoh)
        self.table_mid_widget.setItem(row_index, 0, cell_epoh)
        cell_elco1 = QTableWidgetItem(str(elco1))
        self.table_mid_widget.setItem(row_index, 1, cell_elco1)
        cell_elco2 = QTableWidgetItem(str(elco2))
        self.table_mid_widget.setItem(row_index, 2, cell_elco2)
        cell_time = QTableWidgetItem(str(time))
        self.table_mid_widget.setItem(row_index, 3, cell_time)
        pass

    # def event_reset_click(self):
    #     # 1.获取已选中的行
    #     row_list = self.table_widget.selectionModel().selectedRows()
    #     if not row_list:
    #         QMessageBox.warning(self, "错误", "请选择重新初始化的行")
    #     # 2.获取每一行进行重新初始化
    #     for row_object in row_list:
    #         index = row_object.row()
    #         print("选中的行", index)
    #
    #         # 获取型号
    #         asin = self.table_widget.item(index, 0).text().strip()
    #         print("选中的型号", asin)
    #
    #         # 状态重新初始化
    #         cell_status = QTableWidgetItem("重新初始化")
    #         # cell_status.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    #         self.table_widget.setItem(index, 6, cell_status)
    #
    #         # 创建线程去进行
    #         from utils.threads import NewTaskThread
    #         thread = NewTaskThread(index, self)
    #         thread.success.connect(self.init_task_success_callback)
    #         thread.error.connect(self.init_task_error_callback)
    #         thread.start()
    #     pass

    def event_reset_click(self):
        self.txt_asin.clear()
        self.txt_asin1.clear()
        self.table_widget.clear()
        self.table_mid_widget.clear()
        self.pic_show_label.clear()
        self.pic_show_label2.clear()
        self.pic_show_label3.clear()
        self.label_status.setText("待检测")
        pass

    def event_tip_click(self):
        QMessageBox.information(self, "说明书", "请在相应位置输入待检测图片的位置，点击“开始检测”按钮即可开始检测", QMessageBox.Yes)
        pass

    def event_stop_click(self):
        self.isRun = 0
        pass

    def event_add_click(self):
        url = "./fuse_result.mat"
        self.txt_asin.setText(url)
        pass

    def event_add1_click(self):
        url = "./model/checkpoint"
        self.txt_asin1.setText(url)
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())