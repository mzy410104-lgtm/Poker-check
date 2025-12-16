
from ultralytics import YOLO
import os
import torch

def main():
    if torch.cuda.is_available():
        print(f"成功检测到 GPU: {torch.cuda.get_device_name(0)}")
        device = 0 
    else:
        print(" 未检测到 GPU，将使用 CPU 训练")
        device = 'cpu'
    
    model = YOLO('yolov8n.pt')

    data_path = './datasets/Playing-Cards-Detection-1/data.yaml'

    if not os.path.exists(data_path):
        print(f" 错误：找不到数据集配置文件！")
        print(f"请下载数据集，并解压到: {os.path.abspath(data_path)}")
        return

    results = model.train(
        data=data_path,  # 使用这个变量
        epochs=50,
        imgsz=640,
        plots=True
    )
    
    model.export(format='tflite')

if __name__ == '__main__':
    main()