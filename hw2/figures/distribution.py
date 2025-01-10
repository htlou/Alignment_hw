import json
import matplotlib.pyplot as plt
import numpy as np

path_1 = "/data/align-anything/hantao/Alignment_hw/hw2/outputs/rm_score/ours/eval_data_with_score.json"
path_2 = "/data/align-anything/hantao/Alignment_hw/hw2/outputs/rm_score/baseline/eval_data_with_score.json"

with open(path_1, "r") as f:
    data_1 = json.load(f)

with open(path_2, "r") as f:
    data_2 = json.load(f)

# 1. 提取 score 列表
scores_1 = [item["score"] for item in data_1 if "score" in item]
scores_2 = [item["score"] for item in data_2 if "score" in item]

# 2. 将分数限制在 1~30 区间（如果需要）
scores_1 = [s for s in scores_1 if 1 <= s <= 30]
scores_2 = [s for s in scores_2 if 1 <= s <= 30]

# 3. 计算分布：对 10~30（步长=1），统计每个整数分数出现的次数
x_range = np.arange(10, 31, 1)  # 这里是 10 到 30
dist_1 = [scores_1.count(x) for x in x_range]
dist_2 = [scores_2.count(x) for x in x_range]

# 4. 画分布图（以折线图为例）
plt.figure(figsize=(8, 6))
plt.plot(x_range, dist_1, marker='o', label='Ours')
plt.plot(x_range, dist_2, marker='s', label='Baseline')

# 5. 计算均值，并在图上标出竖直线
avg_1 = np.mean(scores_1)
avg_2 = np.mean(scores_2)
plt.axvline(x=avg_1, color='blue', linestyle='--', label=f'Ours Avg={avg_1:.2f}')
plt.axvline(x=avg_2, color='red', linestyle='--', label=f'Baseline Avg={avg_2:.2f}')

# 6. 添加坐标轴、标题等
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Score Distribution (10 ~ 30)')
plt.legend()
plt.grid(True)

# 7. 显示与保存
plt.savefig("outputs/distribution.png")
plt.savefig("outputs/distribution.pdf")
plt.show()