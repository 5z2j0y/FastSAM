from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np
import cv2

class SegPredictor:
    def __init__(self, image_path):
        self.model = FastSAM('weights/FastSAM-x.pt')
        self.image_path = image_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" 
                                 if torch.backends.mps.is_available() else "cpu")
        
        # 获取原始图片尺寸
        img = cv2.imread(image_path)
        self.original_height, self.original_width = img.shape[:2]
        
        # 运行模型获取基础结果
        self.everything_results = self.model(
            image_path,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
        )
        self.prompt_process = FastSAMPrompt(image_path, self.everything_results, device=self.device)

    def segment_everything(self):
        ann = self.prompt_process.everything_prompt()
        return self._get_result_image(ann)

    def segment_with_points(self, points, point_labels=None):
        if point_labels is None:
            point_labels = [1] * len(points)
        ann = self.prompt_process.point_prompt(points=points, pointlabel=point_labels)
        return self._get_result_image(ann)

    def segment_with_box(self, box):
        ann = self.prompt_process.box_prompt(bbox=box)
        return self._get_result_image(ann)

    def segment_with_text(self, text):
        ann = self.prompt_process.text_prompt(text=text)
        return self._get_result_image(ann)

    def _get_result_image(self, ann):
        # 创建临时文件路径
        output_path = "temp_output.jpg"
        
        self.prompt_process.plot(
            annotations=ann,
            output_path=output_path,
            mask_random_color=True,
            better_quality=True,
            retina=False,
            withContours=True,
        )
        
        # 读取结果图片
        result_image = cv2.imread(output_path)
        return result_image
