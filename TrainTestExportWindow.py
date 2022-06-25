import tkinter as tk
from threading import Thread

from BasicFunctions import *
from TestWindow import TestWindow

class TrainTestExportWindow(tk.Toplevel):
    """
    The choose train, test or export model.
    """
    def __init__(self, gui, prev_window):
        super().__init__()

        self.gui = gui  # The main gui class
        self.prev_window = prev_window  # The previous tkinter window
        self.__stop_thread = False  # If the thread should be stopped

        self.__menubar = None   # The window's Exit menubar
        self.__new_model_btn = None  # The train new model button
        self.__test_existing_btn = None  # The test on exiting model button
        self.__export_model_btn = None  # The export trained model button

        self.__setup_window()

    def __setup_window(self):
        """
        Setup the train, test or export window.

        :return: None
        """
        self.geometry("230x200")  # define the window's size
        self.title("Choose train test or export")

        self.__new_model_btn = tk.Button(self, bg="Yellow", text="Click here to train a new model",
                                         command=self.__on_new_model)
        self.__test_existing_btn = tk.Button(self, bg="Yellow", text="Click here to test an existing model",
                                             command=self.__on_test_existing)
        self.__export_model_btn = tk.Button(self, bg="Yellow", text="Click here to export a model",
                                            command=self.__on_export)

        self.__new_model_btn.pack(pady=20)
        self.__test_existing_btn.pack()
        self.__export_model_btn.pack(pady=20)

        self.protocol("WM_DELETE_WINDOW",
                      self.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.resizable(False, False)
        # create a menu bar with an Exit command
        self.__menubar = tk.Menu(self)
        self.__menubar.add_cascade(label='Exit', command=self.gui.root.destroy)
        self.config(menu=self.__menubar)
        self.focus_force()

    def enable(self):
        """
        Enables the window's buttons

        :return: None
        """
        self.__new_model_btn.configure(state='normal')
        self.__test_existing_btn.configure(state='normal')
        self.__export_model_btn.configure(state='normal')
        self.__menubar.entryconfigure('Exit', state='normal')

    def disable(self):
        """
        Disables the window's buttons

        :return: None
        """
        self.__new_model_btn.configure(state='disable')
        self.__test_existing_btn.configure(state='disable')
        self.__export_model_btn.configure(state='disable')
        self.__menubar.entryconfigure('Exit', state='disabled')

    def __on_new_model(self):
        """
        Called when pressing the train new model button.
        Starts the model training process.

        :return: None
        """
        self.disable()
        t = Thread(target=combine_functions(self.gui.model_handler.train_model, self.__on_thread_stop))
        t.start()

        self.__wait_thread_finish(self.enable)

    def __on_test_existing(self):
        """
        Called when pressing the test on existing model button.
        Starts the load model and testing process.

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

    def __on_export(self):
        """
        Called when pressing the on export model button.
        This window starts the export model process.

        :return: None
        """
        self.disable()
        t = Thread(target=combine_functions(self.gui.model_handler.export_model, self.__on_thread_stop))
        t.start()

        self.__wait_thread_finish(self.enable)

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
