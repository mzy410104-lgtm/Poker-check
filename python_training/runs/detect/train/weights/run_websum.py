import cv2
from ultralytics import YOLO

# --- 详细注释版 ---

# 1. 加载你训练好的模型
# 确保 'best.pt' 和这个 .py 脚本在同一个文件夹下
model = YOLO('best.pt')

# 2. 打开你的电脑摄像头
# 0 代表第一个（默认）摄像头。如果你有多个摄像头，可以试试 1, 2...
cap = cv2.VideoCapture(0)

# 3. 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头。请检查摄像头是否被其他程序占用。")
    exit()

print("摄像头已打开。按 'q' 键退出。")

# 4. 循环读取摄像头的每一帧画面
while True:
    # ret 是一个布尔值 (True/False)，表示是否成功读取到一帧
    # frame 是读取到的那张图片（即视频中的一帧）
    ret, frame = cap.read()

    # 如果读取失败（比如摄像头被拔出）
    if not ret:
        print("错误：无法从摄像头读取帧。")
        break

    # 5.【核心】使用 YOLOv8 模型对当前帧进行预测
    # stream=True 是一种优化的模式，适用于视频流
    results = model(frame, stream=True)

    # 6. 遍历检测到的结果
    for r in results:
        boxes = r.boxes  # 获取所有检测到的边界框

        for box in boxes:
            # a. 获取框的坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # b. 在画面上画出框
            # cv2.rectangle(图像, 左上角坐标, 右下角坐标, 颜色(B,G,R), 线条粗细)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2) # 紫色框

            # c. 获取置信度 (0到1) 和类别
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls] # 从模型中获取类别的名字 (例如 'Ks', 'Ac')

            # d. 准备要显示的标签文字
            label = f'{class_name} {confidence:.2f}' # 例如: 'Ks 0.95'

            # e. 把标签文字写在框的上方
            # cv2.putText(图像, 文字, 坐标, 字体, 大小, 颜色, 粗细)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # 7. 显示处理后的图像
    cv2.imshow('YOLOv8 Live Poker Detection', frame)

    # 8. 等待按键，如果按下 'q' 键则退出循环
    # cv2.waitKey(1) 表示等待1毫秒
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. 循环结束，释放资源
cap.release() # 释放摄像头
cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口