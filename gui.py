import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from segpredict import SegPredictor

class FastSAMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FastSAM Segmentation Tool")
        self.predictor = None
        self.image_path = None
        self.current_image = None
        self.points = []
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 图片显示区域
        self.canvas = tk.Canvas(self.main_frame, width=600, height=400)
        self.canvas.grid(row=0, column=0, columnspan=2, pady=5)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 控制面板
        control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        # 添加按钮和控件
        ttk.Button(control_frame, text="选择图片", command=self.load_image).grid(row=0, column=0, padx=5)
        
        self.prompt_type = tk.StringVar(value="everything")
        prompt_choices = ["everything", "point", "box", "text"]
        ttk.Label(control_frame, text="Prompt类型:").grid(row=0, column=1, padx=5)
        ttk.OptionMenu(control_frame, self.prompt_type, "everything", *prompt_choices).grid(row=0, column=2, padx=5)
        
        self.text_input = ttk.Entry(control_frame)
        self.text_input.grid(row=0, column=3, padx=5)
        
        ttk.Button(control_frame, text="执行分割", command=self.run_segmentation).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="清除", command=self.clear_canvas).grid(row=0, column=5, padx=5)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if self.image_path:
            self.predictor = SegPredictor(self.image_path)
            self.display_image(self.image_path)
            self.points = []

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((600, 400), Image.LANCZOS)
        self.current_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)

    def on_canvas_click(self, event):
        if self.prompt_type.get() == "point":
            x, y = event.x, event.y
            self.points.append([x, y])
            # 在画布上显示点
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red")

    def clear_canvas(self):
        self.canvas.delete("all")
        if self.current_image:
            self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
        self.points = []

    def run_segmentation(self):
        if not self.predictor:
            return
            
        prompt_type = self.prompt_type.get()
        result = None
        
        if prompt_type == "everything":
            result = self.predictor.segment_everything()
        elif prompt_type == "point" and self.points:
            # 转换坐标比例
            scaled_points = [[int(p[0] * self.predictor.original_width / 600),
                            int(p[1] * self.predictor.original_height / 400)] for p in self.points]
            result = self.predictor.segment_with_points(scaled_points)
        elif prompt_type == "text":
            text = self.text_input.get()
            if text:
                result = self.predictor.segment_with_text(text)
                
        if result is not None:
            result_image = cv2.resize(result, (600, 400))
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(result_image)
            self.current_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = FastSAMGUI(root)
    root.mainloop()
