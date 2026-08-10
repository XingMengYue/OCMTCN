import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score



class AccuracyTools:
    @staticmethod
    def calculate_per_class_accuracy(y_true, y_pred, num_classes, class_names=None, compute_kappa = True, compute_seg_f1 = True):
        """
        计算每个类别的准确率和其他指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            num_classes: 类别数量
            class_names: 类别名称列表（可选）

        Returns:
            dict: 包含每个类别的准确率、召回率、F1分数
        """
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        print("\n" + "="*60)
        print("详细分类报告:")
        print("="*60)
        print(classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0
        ))

        conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        print("混淆矩阵:")
        print(conf_matrix)
        print("="*60)

        per_class_acc = {}
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f"Class_{i}"
            # 尝试用 class_name 或 str(i) 作为键
            key = class_name if class_name in report else str(i)
            if key in report:
                per_class_acc[class_name] = {
                    'precision': report[key]['precision'],
                    'recall': report[key]['recall'],
                    'f1-score': report[key]['f1-score'],
                    'support': report[key]['support']
                }

        # 提取宏平均指标
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        macro_pre = report['macro avg']['precision']

        # 新增：Kappa 系数
        kappa = None
        if compute_kappa:
            kappa = cohen_kappa_score(y_true, y_pred)
            print(f"Cohen's Kappa: {kappa:.4f}")

        # return per_class_acc, report['accuracy'], macro_recall, macro_f1
        return per_class_acc, report['accuracy'], macro_pre, macro_recall, macro_f1, kappa

class LossFuncs:
    @staticmethod
    def temporal_smoothness_loss(logits, method='l2', eps=1e-8):
        """
        计算时间平滑正则化损失
        Args:
            logits: torch.Tensor, shape (batch, seq_len, num_classes)
            method: 'l2' 或 'kl'
            eps: 用于数值稳定
        Returns:
            smooth_loss: scalar tensor
        """
        # 转换为概率分布
        probs = F.softmax(logits, dim=-1)   # (B, L, K)
        # 相邻时间步的分布
        probs_prev = probs[:, :-1, :]       # (B, L-1, K)
        probs_next = probs[:, 1:, :]        # (B, L-1, K)

        if method == 'l2':
            # L2 距离平方
            diff = probs_next - probs_prev   # (B, L-1, K)
            loss = torch.mean(diff ** 2)     # 标量
        elif method == 'kl':
            # 对称 KL 散度
            kl1 = F.kl_div(torch.log(probs_prev + eps), probs_next, reduction='none').sum(-1)   # (B, L-1)
            kl2 = F.kl_div(torch.log(probs_next + eps), probs_prev, reduction='none').sum(-1)
            loss = torch.mean(kl1 + kl2) / 2.0
        else:
            raise ValueError("method must be 'l2' or 'kl'")

        return loss

    @staticmethod
    def inverse_order_loss(logits, order_tensor):
        """
        逆序损失
        logits: (batch, seq_len, num_classes)
        order_tensor: (num_classes,) 例如 tensor([0.,1.,2.,...,13.])
        """
        probs = F.softmax(logits, dim=-1)                       # [B, L, K]
        expected = torch.einsum('blk,k->bl', probs, order_tensor)  # [B, L]
        diff = expected[:, :-1] - expected[:, 1:]               # [B, L-1]
        loss = F.relu(diff).pow(2).mean()
        return loss

    @staticmethod
    def multiscale_temporal_smoothness_loss(
            logits,
            method='l2',
            eps=1e-8,
            scales=[1, 2, 3],
            weights=[0.2, 0.3, 0.5],
            use_gaussian_weight=False
    ):
        """
        多尺度双向滑动窗口时间平滑损失
        Args:
            logits: torch.Tensor, shape (batch, seq_len, num_classes)
            method: 'l2' 或 'kl'，损失计算方法
            eps: 用于数值稳定
            scales: 窗口半宽列表，默认[1,2,3]对应窗口大小3,5,7
            weights: 各尺度的权重，长度与scales相同，和为1
            use_gaussian_weight: 是否使用高斯加权，否则使用简单平均

        Returns:
            total_smooth_loss: 标量张量，多尺度平滑总损失
        """
        batch_size, seq_len, num_classes = logits.shape

        # 转换为概率分布
        probs = F.softmax(logits, dim=-1)  # (B, L, K)

        total_loss = 0.0

        # 遍历每个尺度
        for scale, weight in zip(scales, weights):
            k = scale  # 窗口半宽
            window_size = 2 * k + 1

            # 生成高斯权重（如果启用）
            if use_gaussian_weight:
                sigma = k / 2.0
                distances = torch.arange(window_size, device=logits.device) - k
                gaussian_weights = torch.exp(-distances**2 / (2 * sigma**2))
                gaussian_weights = gaussian_weights / gaussian_weights.sum()  # 归一化
            else:
                gaussian_weights = torch.ones(window_size, device=logits.device) / window_size

            # 计算该尺度的损失
            scale_loss = 0.0
            valid_points = 0

            for t in range(seq_len):
                # 确定窗口的实际范围
                start = max(0, t - k)
                end = min(seq_len, t + k + 1)
                actual_window_size = end - start

                # 获取窗口内的概率分布
                window_probs = probs[:, start:end, :]  # (B, W, K)

                # 获取当前点的概率分布
                current_prob = probs[:, t:t+1, :]  # (B, 1, K)

                # 计算当前点与窗口内所有点的差异
                if method == 'l2':
                    diff = window_probs - current_prob  # (B, W, K)
                    point_loss = torch.mean(diff ** 2, dim=-1)  # (B, W)
                elif method == 'kl':
                    # 对称KL散度
                    kl1 = F.kl_div(
                        torch.log(current_prob + eps),
                        window_probs,
                        reduction='none'
                    ).sum(-1)  # (B, W)
                    kl2 = F.kl_div(
                        torch.log(window_probs + eps),
                        current_prob,
                        reduction='none'
                    ).sum(-1)  # (B, W)
                    point_loss = (kl1 + kl2) / 2.0  # (B, W)
                else:
                    raise ValueError("method must be 'l2' or 'kl'")

                # 应用高斯权重
                # 截取对应实际窗口的权重
                weight_start = k - (t - start)
                weight_end = weight_start + actual_window_size
                current_weights = gaussian_weights[weight_start:weight_end]  # (W,)

                # 加权平均
                weighted_loss = torch.sum(point_loss * current_weights, dim=-1)  # (B,)
                scale_loss += torch.mean(weighted_loss)
                valid_points += 1

            # 平均该尺度的损失
            scale_loss = scale_loss / valid_points

            # 加权到总损失
            total_loss += weight * scale_loss

        return total_loss
    
    @staticmethod
    def multiscale_temporal_smoothness_loss1(
            logits,
            method='l2',
            eps=1e-8,
            scale=3,
            use_gaussian_weight=False
    ):
        """
        多尺度双向滑动窗口时间平滑损失
        Args:
            logits: torch.Tensor, shape (batch, seq_len, num_classes)
            method: 'l2' 或 'kl'，损失计算方法
            eps: 用于数值稳定
            scales: 窗口半宽列表，默认[1,2,3]对应窗口大小3,5,7
            use_gaussian_weight: 是否使用高斯加权，否则使用简单平均

        Returns:
            total_smooth_loss: 标量张量，多尺度平滑总损失
        """
        batch_size, seq_len, num_classes = logits.shape

        # 转换为概率分布
        probs = F.softmax(logits, dim=-1)  # (B, L, K)

        total_loss = 0.0

        # 遍历每个尺度
        k = scale  # 窗口半宽
        window_size = 2 * k + 1

        # 生成高斯权重（如果启用）
        if use_gaussian_weight:
            sigma = k / 2.0
            distances = torch.arange(window_size, device=logits.device) - k
            gaussian_weights = torch.exp(-distances**2 / (2 * sigma**2))
            gaussian_weights = gaussian_weights / gaussian_weights.sum()  # 归一化
        else:
            gaussian_weights = torch.ones(window_size, device=logits.device) / window_size

        # 计算该尺度的损失
        scale_loss = 0.0
        valid_points = 0

        for t in range(seq_len):
            # 确定窗口的实际范围
            start = max(0, t - k)
            end = min(seq_len, t + k + 1)
            actual_window_size = end - start

            # 获取窗口内的概率分布
            window_probs = probs[:, start:end, :]  # (B, W, K)

            # 获取当前点的概率分布
            current_prob = probs[:, t:t+1, :]  # (B, 1, K)

            # 计算当前点与窗口内所有点的差异
            if method == 'l2':
                diff = window_probs - current_prob  # (B, W, K)
                point_loss = torch.mean(diff ** 2, dim=-1)  # (B, W)
            elif method == 'kl':
                # 对称KL散度
                kl1 = F.kl_div(
                    torch.log(current_prob + eps),
                    window_probs,
                    reduction='none'
                ).sum(-1)  # (B, W)
                kl2 = F.kl_div(
                    torch.log(window_probs + eps),
                    current_prob,
                    reduction='none'
                ).sum(-1)  # (B, W)
                point_loss = (kl1 + kl2) / 2.0  # (B, W)
            else:
                raise ValueError("method must be 'l2' or 'kl'")

            # 应用高斯权重
            # 截取对应实际窗口的权重
            weight_start = k - (t - start)
            weight_end = weight_start + actual_window_size
            current_weights = gaussian_weights[weight_start:weight_end]  # (W,)

            # 加权平均
            weighted_loss = torch.sum(point_loss * current_weights, dim=-1)  # (B,)
            scale_loss += torch.mean(weighted_loss)
            valid_points += 1

        # 平均该尺度的损失
        scale_loss = scale_loss / valid_points

        return scale_loss