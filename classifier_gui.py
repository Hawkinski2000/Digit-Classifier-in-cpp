# This project uses some code inspired by "NeuralNetworkFromScratch" (https://github.com/Bot-Academy/NeuralNetworkFromScratch) by Bot Academy.
import ctypes
from _ctypes import LoadLibrary
import tkinter as tk
from tkinter import Button, PhotoImage 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Classifier:
    def __init__(self):
        root = tk.Tk()
        root.geometry("1000x1000")
        root.resizable(True, True)
        root.configure(bg='white')
        root.title("Classifier")

        self.canvas_size = 28
        self.scale_factor = 20
        self.canvas = tk.Canvas(
            root,
            bg='white',
            width=self.canvas_size * self.scale_factor,
            height=self.canvas_size * self.scale_factor
        )
        self.canvas.pack()
        self.color = 'black'
        self.canvas.bind('<B1-Motion>', self.paint)

        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM)
        button_image_1 = PhotoImage(file=("clear.png"))
        clear_button = Button(
            command=self.clear_canvas,
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            relief="flat"
        )
        clear_button.pack(side=tk.BOTTOM, pady=20)

        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas_plot = self.create_matplotlib_chart(root)

        self.filepath = './libclassifier.dll'
        # Load the shared library
        if ctypes._os.name == 'nt': # Windows
            self.handle = LoadLibrary(self.filepath)
            self.lib = ctypes.CDLL(name = self.filepath, handle = self.handle)
        else: # Linux/macOS
            self.lib = ctypes.CDLL(self.filepath)

        self.lib.classify.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        ]
        self.lib.classify.restype = ctypes.POINTER(ctypes.c_float)

        self.img = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)

        self.lib.train();
        root.mainloop()

    def create_matplotlib_chart(self, parent):
        chart = FigureCanvasTkAgg(self.fig, master=parent)
        self.ax.set_title(f"Predicted Class: ")
        self.ax.set_xlabel("Classes")
        self.ax.set_ylabel("Probability")
        self.ax.set_xticks(range(10))
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.15)
        self.ax.set_xlim(0, 10)
        chart.draw()
        chart.get_tk_widget().pack()
        return chart

    def paint(self, event):
        x = event.x // self.scale_factor
        y = event.y // self.scale_factor

        if (0 <= x < self.canvas_size and
            0 <= y < self.canvas_size and
            0 <= x + 1 < self.canvas_size):
            self.canvas.create_rectangle(
                x * self.scale_factor,
                y * self.scale_factor,
                (x + 2) * self.scale_factor,
                (y + 2) * self.scale_factor,
                fill=self.color,
                outline=self.color
            )
            self.img[y][x] = 1.0
            self.img[y+1][x] = 1.0
            self.img[y][x+1] = 1.0
            self.img[y+1][x+1] = 1.0
            self.classify_image()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.img = np.zeros(
            (self.canvas_size,
             self.canvas_size),
             dtype=np.float32
             )
        self.ax.clear()
        self.ax.set_title(f"Predicted Class: ")
        self.ax.set_xlabel("Classes")
        self.ax.set_ylabel("Probability")
        self.ax.set_xticks(range(10))
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.15)
        self.ax.set_xlim(0, 10)
        self.canvas_plot.draw()

    def classify_image(self):
        self.drawn_image = np.array(self.img).reshape(1, -1)
        flat_image = self.drawn_image.flatten().T
        output_pointer = self.lib.classify(flat_image)
        output_probs = np.ctypeslib.as_array(output_pointer, shape=(10,))
        prediction = np.argmax(output_probs)
        self.ax.clear()
        self.ax.bar(range(10), output_probs, color='#00ffba')
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"Predicted Class: {prediction}")
        self.ax.set_xlabel("Classes")
        self.ax.set_ylabel("Probability")
        self.ax.set_xticks(range(10))
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.15)
        self.ax.set_xlim(0, 10)
        self.canvas_plot.draw()

def main():
    # choice = input(
    #     "Enter 1 to use pre-trained weights or 2 to train from scratch: "
    # )
    # if int(choice) == 1:
    #     use_pretrained_weights = True
    # else:
    #     use_pretrained_weights = False
    
    Classifier()

main()