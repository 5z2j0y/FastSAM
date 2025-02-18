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
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.box = None
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 图片显示区域
        self.canvas = tk.Canvas(self.main_frame, width=600, height=400)
        self.canvas.grid(row=0, column=0, columnspan=2, pady=5)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
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

    def on_mouse_down(self, event):
        if self.prompt_type.get() == "point":
            x, y = event.x, event.y
            self.points.append([x, y])
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red")
        elif self.prompt_type.get() == "box":
            self.start_x = event.x
            self.start_y = event.y

    def on_mouse_drag(self, event):
        if self.prompt_type.get() == "box":
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            self.current_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline="red", width=2
            )

    def on_mouse_release(self, event):
        if self.prompt_type.get() == "box":
            end_x, end_y = event.x, event.y
            # 确保坐标的顺序是 [左上角x, 左上角y, 右下角x, 右下角y]
            x1 = min(self.start_x, end_x)
            y1 = min(self.start_y, end_y)
            x2 = max(self.start_x, end_x)
            y2 = max(self.start_y, end_y)
            self.box = [x1, y1, x2, y2]

    def clear_canvas(self):
        self.canvas.delete("all")
        if self.current_image:
            self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
        self.points = []
        self.box = None
        self.current_rect = None

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
        elif prompt_type == "box" and self.box:
            # 转换坐标比例
            scaled_box = [
                int(self.box[0] * self.predictor.original_width / 600),
                int(self.box[1] * self.predictor.original_height / 400),
                int(self.box[2] * self.predictor.original_width / 600),
                int(self.box[3] * self.predictor.original_height / 400)
            ]
            result = self.predictor.segment_with_box(scaled_box)
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
