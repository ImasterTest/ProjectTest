import tkinter as tk
from threading import Thread

from BasicFunctions import *
from TestDisplayWindow import TestDisplayWindow

class TestWindow(tk.Toplevel):
    """
    The test model window.
    """
    def __init__(self, gui, prev_window):
        super().__init__()

        self.gui = gui  # The main gui class
        self.prev_window = prev_window  # The previous tkinter window
        self.__stop_thread = False  # If the thread should be stopped

        self.__menubar = None  # The window's Back menubar
        self.__test_image_btn = None  # The test on image button
        self.__test_video_btn = None  # The test on video file button
        self.__test_camera_btn = None  # The test on live cam button

        self.__setup_window()

    def __setup_window(self):
        """
        Setup the test window.

        :return: None
        """
        self.geometry("200x200")  # define the window's size
        self.title("Test Model")

        self.__test_image_btn = tk.Button(self, bg="Yellow", text="Click here to test on an image",
                                          command=self.__on_test_image)
        self.__test_video_btn = tk.Button(self, bg="Yellow", text="Click here to test on video",
                                          command=self.__on_test_video)
        self.__test_camera_btn = tk.Button(self, bg="Yellow", text="Click here to test on live camera",
                                           command=self.__on_test_live)

        self.__test_image_btn.pack(pady=20)
        self.__test_video_btn.pack()
        self.__test_camera_btn.pack(pady=20)

        self.protocol("WM_DELETE_WINDOW",
                      self.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.resizable(False, False)
        # create a menu bar with an Exit command
        self.__menubar = tk.Menu(self)
        self.__menubar.add_cascade(label='Back', command=combine_functions(self.destroy, self.prev_window.enable, self.prev_window.focus_force))
        self.config(menu=self.__menubar)
        self.focus_force()

    def enable(self):
        """
        Enables the window's buttons.

        :return: None
        """
        self.__test_image_btn.configure(state='normal')
        self.__test_video_btn.configure(state='normal')
        self.__test_camera_btn.configure(state='normal')
        self.__menubar.entryconfigure('Back', state='normal')

    def disable(self):
        """
        Disables the window's buttons

        :return: None
        """
        self.__test_image_btn.configure(state='disable')
        self.__test_video_btn.configure(state='disable')
        self.__test_camera_btn.configure(state='disable')
        self.__menubar.entryconfigure('Back', state='disabled')

    def __on_test_image(self):
        """
        Called when pressing the test on image button.
        Opens the test display window with the test on image function.

        :return: None
        """
        self.disable()
        TestDisplayWindow(self.gui, self, self.gui.model_handler.test_on_image)

    def __on_test_video(self):
        """
        Called when pressing the test on video button.
        Executes the test on video.

        :return: None
        """
        self.disable()
        t = Thread(target=combine_functions(self.gui.model_handler.test_on_video, self.__on_thread_stop))
        t.start()
        self.__wait_thread_finish(self.enable)

    def __on_test_live(self):
        """
        Called when pressing the test on live cam button.
        Open the test display window with the test on live function.

        :return: None
        """
        self.disable()
        TestDisplayWindow(self.gui, self, self.gui.model_handler.test_on_live)

    def __on_thread_stop(self):
        """
        Stop the thread avoiding the tkinter freeze.

        :return: None
        """
        self.__stop_thread = True

    def __wait_thread_finish(self, *funcs):
        """
        Wait until the thread is finished and execute the functions.

        :param funcs: Functions to run when the thread finishes
        :return: None
        """
        if self.__stop_thread:
            self.__stop_thread = False
            for func in funcs:
                func()
        else:
            self.gui.root.after(100, self.__wait_thread_finish, *funcs)
