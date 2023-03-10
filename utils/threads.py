from PyQt5.QtCore import QThread, pyqtSignal


class NewTaskThread(QThread):

    # 信号，触发信号，更新窗体
    success = pyqtSignal(int, str, str, str)
    error = pyqtSignal(int, str, str, str)

    def __init__(self, row_index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_index = row_index

    def run(self):
        # 具体线程应该做的事
        self.success.emit(self.row_index, "xx", "xx", "xx")
        self.error.emit(1, "xx", "xx", "xx")

        pass