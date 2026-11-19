import time
import torch
import numpy as np
from ultralytics import YOLO

# ==================== 配置区：请替换为你的真实权重路径 ====================
BASELINE_WEIGHTS = "yolov8n.pt"          # 基线模型权重路径 (如 exp2_stage2_baseline/weights/best.pt)
OURS_WEIGHTS     = r"F:\ultralyticsyolo8\runs\detect\exp1_stage1_v8_ohid_ours\weights\best.pt"       # 本文完整模型权重路径 (如 exp2_stage2_ours/weights/best.pt)
IMG_SIZE         = 640                   # 输入分辨率 640x640
BATCH_SIZE       = 1                     # 批大小 1
USE_FP16         = True                  # 开启 FP16 半精度推理
WARMUP_ITERS     = 100                   # 预热轮数（消除 GPU 冷启动与 CUDA 初始化误差）
TEST_ITERS       = 1000                  # 正式测速迭代次数（取均值保证稳定性）
# =========================================================================

def measure_model(weights_path: str, model_label: str):
    assert torch.cuda.is_available(), "未检测到 CUDA，请在 RTX 5090 D 环境下运行！"
    device = torch.device("cuda:0")

    # 1. 加载模型并提取标准 Params 与 GFLOPs
    yolo = YOLO(weights_path)
    # 融合 Conv+BN 层以模拟真实部署推理状态
    yolo.fuse()
    model = yolo.model.to(device).eval()

    # 获取参数量 (M) 与计算量 (GFLOPs)
    # 注：Ultralytics 内置 get_flops 默认基于 640x640 计算
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    params_m = get_num_params(model) / 1e6
    gflops = get_flops(model, imgsz=IMG_SIZE)

    # 2. 构造 dummy 输入张量并转换精度
    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)
    if USE_FP16:
        model = model.half()
        dummy_input = dummy_input.half()

    # 3. GPU 预热 (Warm-up)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # 4. 使用 CUDA Event 精确测量纯前向传播时延（不含图像预处理与 NMS 后处理）
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    latencies_ms = []

    with torch.no_grad():
        for _ in range(TEST_ITERS):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()  # 等待当前前向推理彻底完成
            latencies_ms.append(starter.elapsed_time(ender))

    latencies_ms = np.array(latencies_ms)
    # 剔除极端毛刺（取 5%~95% 分位区间均值，更贴近论文稳定测速标准）
    p5, p95 = np.percentile(latencies_ms, [5, 95])
    trimmed = latencies_ms[(latencies_ms >= p5) & (latencies_ms <= p95)]
    avg_latency_ms = float(np.mean(trimmed))
    fps = 1000.0 / avg_latency_ms

    print(f"[{model_label}] 测量完成:")
    print(f"  Params  : {params_m:.2f} M")
    print(f"  GFLOPs  : {gflops:.1f}")
    print(f"  Latency : {avg_latency_ms:.2f} ms (std: {np.std(trimmed):.3f} ms)")
    print(f"  FPS     : {fps:.1f}\n")

    return {
        "params": params_m,
        "gflops": gflops,
        "latency": avg_latency_ms,
        "fps": fps
    }

if __name__ == "__main__":
    print(f"测试硬件: {torch.cuda.get_device_name(0)}")
    print(f"测试条件: 输入={IMG_SIZE}x{IMG_SIZE}, Batch={BATCH_SIZE}, FP16={USE_FP16}, 迭代={TEST_ITERS}次\n")

    base_res = measure_model(BASELINE_WEIGHTS, "YOLOv8n基线")
    ours_res = measure_model(OURS_WEIGHTS, "本文完整模型")

    # 计算相对变化百分比
    d_params  = (ours_res["params"]  - base_res["params"])  / base_res["params"]  * 100
    d_gflops  = (ours_res["gflops"]  - base_res["gflops"])  / base_res["gflops"]  * 100
    d_latency = (ours_res["latency"] - base_res["latency"]) / base_res["latency"] * 100
    d_fps     = (ours_res["fps"]     - base_res["fps"])     / base_res["fps"]     * 100

    print("=" * 65)
    print("表8 真实测量数据汇总（可直接填入论文）：")
    print("=" * 65)
    print(f"{'模型':<14} | {'Params/M':<9} | {'GFLOPs':<8} | {'Latency/ms':<11} | {'FPS':<8}")
    print("-" * 65)
    print(f"{'YOLOv8n基线':<14} | {base_res['params']:<9.2f} | {base_res['gflops']:<8.1f} | {base_res['latency']:<11.2f} | {base_res['fps']:<8.1f}")
    print(f"{'本文完整模型':<14} | {ours_res['params']:<9.2f} | {ours_res['gflops']:<8.1f} | {ours_res['latency']:<11.2f} | {ours_res['fps']:<8.1f}")
    print("-" * 65)
    print(f"{'相对变化':<14} | {d_params:+.1f}%{'':<4} | {d_gflops:+.1f}%{'':<3} | {d_latency:+.1f}%{'':<6} | {d_fps:+.1f}%")
    print("=" * 65)