import tkinter as tk
from threading import Thread

from BasicFunctions import *

class TestDisplayWindow(tk.Toplevel):
    """
    The display window for the prediction results.
    """
    def __init__(self, gui, prev_window, run_func):
        super().__init__()

        self.gui = gui  # The main gui class
        self.prev_window = prev_window  # The previous tkinter window
        self.__run_func = run_func  # The function to run when testing
        self.__latest_img = None  # The latest image shown to the user

        self.__image_lbl = None  # The predicted image label

        self.__setup_window()

    def __setup_window(self):
        """
        Setup the test display window.

        :return: None
        """
        self.geometry("800x800")  # define the window's size
        self.title("Image test on model")

        self.__image_lbl = tk.Label(self, text='Loading image...')
        self.__image_lbl.pack()

        self.protocol("WM_DELETE_WINDOW",
                      self.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.resizable(False, False)
        # create a menu bar with an Exit command
        menubar = tk.Menu(self)
        menubar.add_cascade(label='Back',
                            command=combine_functions(self.destroy, self.prev_window.enable, self.prev_window.focus_force))
        self.config(menu=menubar)
        self.focus_force()

        t = Thread(target=self.__do_test)
        t.start()

    def __do_test(self):
        """
        Runs the test of the model based on the test function given to it.
        Shows the output on the label.

        :return: None
        """
        self.__latest_img = self.__run_func(self.__image_lbl)
        if self.__latest_img is not None:
            self.geometry(f'{self.__latest_img.width()}x{self.__latest_img.height()}')
        elif self.winfo_exists():
            self.destroy()
            self.prev_window.enable()
            self.prev_window.focus_force()
