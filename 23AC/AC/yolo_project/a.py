file_path = 'D:/yolo_project/yolov5/runs/train/project_yolov5l_sgd_results_1013/weights/best.pt'
try:
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)
except Exception as e:
    print(f"An error occurred while opening the file: {str(e)}")