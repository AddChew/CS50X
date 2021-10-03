# Import the necessary libraries
import os
import sys
import socket
from config import ICON_PATH
from PySide2 import QtCore, QtWidgets, QtGui, QtWebEngineWidgets


class ApplicationThread(QtCore.QThread):
    def __init__(self, application, port=5000):
        super(ApplicationThread, self).__init__()
        self.application = application
        self.port = port

    def __del__(self):
        self.wait()

    def run(self):
        self.application.run(port=self.port, threaded=True, 
                             debug=True, use_reloader= False)


class WebPage(QtWebEngineWidgets.QWebEnginePage):
    def __init__(self, root_url):
        super(WebPage, self).__init__()
        self.root_url = root_url

    def home(self):
        self.load(QtCore.QUrl(self.root_url))

    def acceptNavigationRequest(self, url, kind, is_main_frame):
        """Open external links in browser and internal links in the webview"""
        ready_url = url.toEncoded().data().decode()
        is_clicked = kind == self.NavigationTypeLinkClicked
        if is_clicked and self.root_url not in ready_url:
            QtGui.QDesktopServices.openUrl(url)
            return False
        return super(WebPage, self).acceptNavigationRequest(url, kind, is_main_frame)


class WebView(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, page, window, icon, zoom_factor = 1.75):
        super(WebView, self).__init__(window)
        self.download_dir = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.DownloadLocation)
        self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.setPage(page)
        self.setZoomFactor(zoom_factor)
        self.page().profile().downloadRequested.connect(
            self.on_download_request
        )
        self.icon = icon
        self.popup_attr = {
            "download": {
                "icon": QtWidgets.QMessageBox.Information,
                "windowTitle": "Download Complete",
                "windowText": "Download completed successfully."
            }
        }

    def check_filename(self, filename):
        count = 0
        basename, ext = os.path.splitext(filename)
        filePath = os.path.join(self.download_dir, filename)
        while os.path.exists(filePath):
            count += 1
            filename = f"{basename} ({count}){ext}"
            filePath = os.path.join(self.download_dir, filename)
        return filePath

    def popup_message(self, type="download"):
        attr = self.popup_attr.get(type)
        if not attr:
            return

        popup = QtWidgets.QMessageBox()
        popup.setWindowIcon(QtGui.QIcon(self.icon))
        popup.setIcon(attr.get("icon"))
        popup.setWindowTitle(attr.get("windowTitle"))
        popup.setText(attr.get("windowText"))
        popup.setStandardButtons(QtWidgets.QMessageBox.Ok)
        popup.exec_()

    @QtCore.Slot("QWebEngineDownloadItem")
    def on_download_request(self, download):
        originalPath = download.url().path()
        fileInfo = QtCore.QFileInfo(originalPath)
        savePath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 
            "Save As",
            self.check_filename(fileInfo.fileName()),
            f"*.{fileInfo.suffix()}"
        )
        if savePath:
            download.finished.connect(lambda: self.popup_message("download"))
            download.setPath(savePath)
            download.accept()


def init_gui(application, port=0, window_title="Top2VecApp", 
             icon=ICON_PATH, argv=None):
    if argv is None:
        argv = sys.argv

    if port == 0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 0))
        port = sock.getsockname()[1]
        sock.close()

    # Application Level
    qtapp = QtWidgets.QApplication(argv)
    webapp = ApplicationThread(application, port)
    webapp.start()
    qtapp.aboutToQuit.connect(webapp.terminate)

    # Main Window Level
    window = QtWidgets.QMainWindow()
    window.showMaximized()
    window.setWindowTitle(window_title)
    window.setWindowIcon(QtGui.QIcon(icon))

    # WebPage Level
    page = WebPage(f"http://localhost:{port}")
    page.home()
    qtapp.aboutToQuit.connect(page.deleteLater)

    # WebView Level
    webView = WebView(page, window, icon)
    window.setCentralWidget(webView)
    window.show()

    return qtapp.exec_()