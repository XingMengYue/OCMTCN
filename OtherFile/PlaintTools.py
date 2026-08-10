
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def plot_segment_comparison(true_labels, pred_labels, class_names, well_name, segment_idx,
                           save_path, segment_accuracy=None):
    """
    绘制单个切片的真实标签与预测标签对比图

    参数:
        true_labels: 真实标签序列 [seq_len]
        pred_labels: 预测标签序列 [seq_len]
        class_names: 类别名称列表
        well_name: 井名
        segment_idx: 片段编号
        save_path: 保存路径
        segment_accuracy: 该切片的准确率（可选）
    """
    seq_len = len(true_labels)
    num_classes = len(class_names)

    # 生成颜色映射
    cmap = plt.cm.get_cmap('tab20', num_classes)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(4, 12))

    # 标题
    if segment_accuracy is not None:
        title = f'{well_name} - Segment {segment_idx}\nAccuracy: {segment_accuracy:.2%}'
    else:
        title = f'{well_name} - Segment {segment_idx}'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

    # 准备颜色数组
    true_colors = [cmap(label) for label in true_labels]
    pred_colors = [cmap(label) for label in pred_labels]

    # 左列：真实标签
    axes[0].set_title('True Labels', fontsize=10, fontweight='bold')
    for i in range(seq_len):
        rect = plt.Rectangle((0, seq_len - i - 1), 1, 1,
                             facecolor=true_colors[i], edgecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, seq_len)
    axes[0].set_xticks([])
    axes[0].set_ylabel('Depth Position', fontsize=9)
    axes[0].tick_params(axis='y', labelsize=7)

    # 右列：预测标签
    axes[1].set_title('Predicted Labels', fontsize=10, fontweight='bold')
    for i in range(seq_len):
        rect = plt.Rectangle((0, seq_len - i - 1), 1, 1,
                             facecolor=pred_colors[i], edgecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, seq_len)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # 添加图例
    legend_patches = [mpatches.Patch(color=cmap(i), label=class_names[i])
                      for i in range(num_classes)]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=min(3, num_classes), fontsize=7, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()