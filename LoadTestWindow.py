import functools
import tkinter as tk
from threading import Thread

from TestWindow import TestWindow
from TrainTestExportWindow import TrainTestExportWindow
from BasicFunctions import *

class LoadTestWindow(tk.Toplevel):
    """
    The choose load data or test model screen.
    """
    def __init__(self, gui):
        super().__init__()
        self.gui = gui  # The main gui class
        self.__stop_thread = False  # If the thread should be stopped

        self.__menubar = None  # The window's Exit menubar
        self.__data_btn = None  # The load data button
        self.__test_btn = None  # The test a model button

        self.__setup_window()

    def __setup_window(self):
        """
        Setup the load data or test model window.

        :return: None
        """
        self.geometry("200x120")  # define the window's size
        self.title("Choose load or test")

        self.__data_btn = tk.Button(self, bg="Yellow", text="Click here to load the data",
                                    command=self.__on_load_data)
        self.__test_btn = tk.Button(self, bg="Yellow", text="Click here to test a model",
                                    command=self.__on_test)

        self.__data_btn.pack(pady=20)
        self.__test_btn.pack()

        self.protocol("WM_DELETE_WINDOW",
                      self.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.resizable(False, False)
        # create a menu bar with an Exit command
        self.__menubar = tk.Menu(self)
        self.__menubar.add_cascade(label='Exit', command=self.gui.root.destroy)
        self.config(menu=self.__menubar)

    def enable(self):
        """
        Enables the window's buttons.

        :return: None
        """
        self.__data_btn.configure(state='normal')
        self.__test_btn.configure(state='normal')
        self.__menubar.entryconfigure('Exit', state='normal')

    def disable(self):
        """
        Disables the window's buttons.

        :return: None
        """
        self.__data_btn.configure(state='disable')
        self.__test_btn.configure(state='disable')
        self.__menubar.entryconfigure('Exit', state='disabled')

    def __on_load_data(self):
        """
        Called when pressing the load data button,
        Starts the data load process.

        :return: None
        """
        self.disable()
        t = Thread(target=combine_functions(self.gui.data_handler.handle_dataset, self.__on_thread_stop))
        t.start()
        self.__wait_thread_finish(self.destroy, functools.partial(TrainTestExportWindow, self.gui, self))

    def __on_test(self):
        """
        Called when pressing the test model button,
        Starts the load model and then test window.

        :return: None
        """
        self.disable()

        def load():
            result = self.gui.model_handler.load_model()
            if result:
                TestWindow(self.gui, self)
            else:
                self.enable()
                self.focus_force()

        t = Thread(target=load)
        t.start()

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
