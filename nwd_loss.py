import torch
import torch.nn.functional as F

def compute_nwd_loss(pred_bboxes, target_bboxes, C=12.8):
    """
    计算预测框与真实框之间的 NWD 损失
    pred_bboxes: 预测框 [x_center, y_center, w, h]
    target_bboxes: 真实框 [x_center, y_center, w, h]
    """
    # 1. 提取中心点和宽高
    p_x, p_y, p_w, p_h = pred_bboxes.unbind(-1)
    t_x, t_y, t_w, t_h = target_bboxes.unbind(-1)

    # 2. 将边界框建模为二维高斯分布的方差 (sigma = width/2, height/2)
    # 为防止除以0或极小值，加一个极小的 eps
    eps = 1e-7
    p_sigma_x, p_sigma_y = (p_w / 2).clamp(min=eps), (p_h / 2).clamp(min=eps)
    t_sigma_x, t_sigma_y = (t_w / 2).clamp(min=eps), (t_h / 2).clamp(min=eps)

    # 3. 计算中心点距离的平方
    center_dist = (p_x - t_x) ** 2 + (p_y - t_y) ** 2

    # 4. 计算 Wasserstein 距离的平方 (针对对角协方差高斯分布的简化闭式解)
    # W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2
    w2_dist = center_dist + (p_sigma_x - t_sigma_x) ** 2 + (p_sigma_y - t_sigma_y) ** 2

    # 5. 通过指数形式归一化到 [0, 1] 之间得到 NWD 相似度
    nwd = torch.exp(-torch.sqrt(w2_dist.clamp(min=eps)) / C)

    # 6. NWD Loss = 1 - NWD
    loss_nwd = 1.0 - nwd
    return loss_nwd.mean()