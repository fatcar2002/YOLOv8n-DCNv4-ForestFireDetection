import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 🎯 设定数据集配置文件路径
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_Ue_middlesmall_target\UEdataset.yaml"
    
    # 🎯 待验证的权重路径
    # 可以分别替换为 Base 的权重或 Ours 后 60 代微调的最佳权重
    model_weight_path = r"F:\ultralyticsyolo8\runs\detect\runs\detect\stage5_yolov8_UE_base\weights\last.pt"  
    
    print(f"================ 🔍 开始验证模型权重: {model_weight_path} ================")
    
    if not os.path.exists(model_weight_path):
        print(f"⚠️ 提示: 权重文件 {model_weight_path} 尚未生成或路径不正确，请先确保训练已完成。")
    else:
        model = YOLO(model_weight_path)
        
        # 运行验证
        metrics = model.val(data=dataset_yaml)
        
        # 打印综合及单类别详细指标
        print("\n================ 📊 验证结果汇总 ================")
        print(f"全局 mAP50     : {metrics.box.map50:.4f}")
        print(f"全局 mAP50-95  : {metrics.box.map:.4f}")
        
        # 提取并打印纯火焰 (class 0: fire) 的专属数据
        if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
            print(f"🔥 类别 0 (fire / 火焰) AP50     : {metrics.box.maps[0]:.4f}")
        if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0:
            print(f"🔥 类别 0 (fire / 火焰) Precision : {metrics.box.p[0]:.4f}")
        if hasattr(metrics.box, 'r') and len(metrics.box.r) > 0:
            print(f"🔥 类别 0 (fire / 火焰) Recall    : {metrics.box.r[0]:.4f}")
            
        print("==================================================")