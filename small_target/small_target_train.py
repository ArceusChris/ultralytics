from ultralytics import YOLO
import yaml

def train_optimized_yolo():
    # 加载自定义模型配置
    model = YOLO('yolo11_small_target.yaml')
    
    # 开始训练，使用优化配置
    results = model.train(
        data='datasets.yaml',
        epochs=150,
        patience=25,
        batch=32,
        imgsz=416,
        
        # 关键优化参数
        single_cls=True,  # 启用单类模式
        max_det=2,        # 限制最大检测数量为2
        
        # 性能优化
        conf=0.25,
        iou=0.45,
        
        # 训练优化
        optimizer='AdamW',
        lr0=0.002,
        lrf=0.01,
        warmup_epochs=3.0,
        
        # 数据增强（针对运动小目标优化）
        degrees=8.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        
        # 损失函数权重调整
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 其他设置
        device='',  # 自动检测设备
        workers=8,
        project='runs/train',
        name='small_target_v1',
        exist_ok=True,
        save=True,
        save_period=50,
        plots=True,
        val=True
    )
    
    return results

if __name__ == "__main__":
    results = train_optimized_yolo()
    print("训练完成！")
    print(f"最佳权重保存位置: {results.save_dir}/weights/best.pt")