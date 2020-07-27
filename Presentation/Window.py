import Constants
import tkinter as tk
from tkinter import *


label_positions = {}

def increase(*args):
    lbl_value = args[0]
    increase_value = args[1]
    value = int(lbl_value["text"])
    lbl_value["text"] = f"{value + increase_value}"


def decrease(*args):
    lbl_value = args[0]
    decrease_value = args[1]
    value = int(lbl_value["text"])
    lbl_value["text"] = f"{value - decrease_value}"


class Window:
    def __init__(self, controller):
        self.controller = controller
        self.window = tk.Tk()
        self.window.rowconfigure(0, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)
        self.window.rowconfigure([0, 1, 2], minsize=50, weight=1)
        self.initComponents()
        self.window.mainloop()

    def initIterations(self, parentComponent):
        lbl_iterations = tk.Label(master=parentComponent, text="Total iterations")
        lbl_iterations.grid(row=0, column=0)

        lbl_iterations_value = tk.Label(master=parentComponent, text="0")
        lbl_iterations_value.grid(row=0, column=2)

        btn_iterations_decrease = tk.Button(master=parentComponent, text="-")
        btn_iterations_decrease.grid(row=0, column=1, sticky="nsew")
        btn_iterations_decrease['command'] = lambda arg0=lbl_iterations_value, arg1=10: decrease(arg0, arg1)

        btn_iterations_increase = tk.Button(master=parentComponent, text="+", command=increase)
        btn_iterations_increase.grid(row=0, column=3, sticky="nsew")
        btn_iterations_increase['command'] = lambda arg0=lbl_iterations_value, arg1=10: increase(arg0, arg1)

    @staticmethod
    def setStartPosition(labelClicked):
        print(labelClicked)

    @staticmethod
    def setGoalPosition(labelClicked):
        print(labelClicked)

    def initGridWorld(self):
        gridWorldFrame = Frame(self.window)
        gridWorldFrame.pack(side=BOTTOM)
        for i in range(Constants.DEFAULT_DIMENSION):
            gridWorldFrame.columnconfigure(i, weight=1, minsize=75)
            gridWorldFrame.rowconfigure(i, weight=1, minsize=50)

            for j in range(0, Constants.DEFAULT_DIMENSION):
                frame = tk.Frame(
                    master=gridWorldFrame,
                    relief=tk.RAISED,
                    borderwidth=1
                )
                frame.grid(row=i, column=j)

                label = tk.Label(master=frame, text=f"Row {i}\nColumn {j}")
                label_positions[frame, label] = (i, j)

                label.bind("<Button-1>", lambda labelClicked=label: self.setStartPosition(labelClicked))
                label.bind("<Button-2>", lambda labelClicked=label: self.setGoalPosition(labelClicked))
                label.pack(padx=5, pady=5)

    @staticmethod
    def initSamples(parentComponent):
        lbl_samples = tk.Label(master=parentComponent, text="Total samples")
        lbl_samples.grid(row=1, column=0)

        lbl_samples_value = tk.Label(master=parentComponent, text="0")
        lbl_samples_value.grid(row=1, column=2)

        btn_samples_decrease = tk.Button(master=parentComponent, text="-")
        btn_samples_decrease.grid(row=1, column=1, sticky="nsew")
        btn_samples_decrease['command'] = lambda arg0=lbl_samples_value, arg1=1: decrease(arg0, arg1)

        btn_samples_increase = tk.Button(master=parentComponent, text="+", command=increase)
        btn_samples_increase.grid(row=1, column=3, sticky="nsew")
        btn_samples_increase['command'] = lambda arg0=lbl_samples_value, arg1=1: increase(arg0, arg1)

    def initComponents(self):
        topFrame = Frame(self.window)
        topFrame.pack(side=TOP)
        self.initIterations(topFrame)
        self.initSamples(topFrame)
        self.initGridWorld()

    def getDimension(self):
        pass

    def getGoalPosition(self):
        pass

    def getStartPosition(self):
        pass
