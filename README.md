# 第六届计图挑战赛 · 赛题二（点云降噪）—— Jittor 实现

基于 ASDN（*"You Should Learn to Stop Denoising on Point Clouds in Advance"*）
的**全 Jittor 复现**：噪声自适应地决定编解码器展开的深度，逐 patch 预测位移场，
再把 patch 拼回完整点云。

**不依赖** PyTorch / PyTorch-Lightning / pytorch3d / torch_geometric /
torch_cluster，**不需要编译任何 CUDA 扩展**。所有原本由 CUDA 扩展提供的算子
（`pointops` 的 FPS / 分组 / 插值、`Chamfer3D`、`pytorch3d.ops.knn_points`）
都在 `models/pointops_jt.py` 与 `models/InfoCD.py` 中用纯 Jittor 张量运算重写。

所有权重均在**赛方提供的 ShapeNet 训练集上从零训练**，未使用任何外部数据或
外部预训练权重。

---

## 目录结构

```
code/
├── train_classifier_on_starter.py   # ① 训练 ScaleNet 噪声强度分类器
├── train_on_starter.py              # ② 训练 ASDN 去噪主网络
├── predict_on_starter.py            # ③ 在测试集上推理，产出 denoised.npy
├── check_submission.py              # ④ 打包前自检（点数/NaN/数值范围）
├── make_datalist.py                 # 由数据目录重新生成 datalist（官方 starter 亦提供）
│
├── models/
│   ├── denoiseCD.py        DenoiseNetCD：训练损失 + patch 化推理与拼接
│   ├── feature.py          FeatureExtraction：噪声自适应深度的编码/解码主干
│   ├── blocks.py           StartBlock / Downsampling(MRE+FPS) / Upsampling(交叉注意力)
│   ├── classify.py         Classify：熵比回归的监督损失
│   ├── classifyNet.py      ScaleNet：DGCNN 风格的噪声强度估计网络
│   ├── dynamic_edge_conv.py  EdgeConv（稠密 (B,N,k,C) 邻居张量实现）
│   ├── InfoCD.py           InfoCD 损失 + 纯 Jittor Chamfer 距离
│   ├── pointops_jt.py      FPS / KNN / 邻域分组 / 反距离插值（纯 Jittor）
│   └── utils.py            get_entropy_B：体素密度熵（监督目标）
│
├── bridge/
│   └── data_bridge.py      赛题数据 → ASDN patch 的数据集（采样/归一化/加噪/取块）
│
├── tools/                  本地评测与分析工具（不参与 A 榜结果生成）
│   ├── evaluate.py         复现赛题评分公式（0.5×CD + 0.5×P2S）
│   ├── make_eval_set.py    从**训练集**构造带 GT 的本地验证集
│   ├── analysis.py         逐样本得分统计（按类别 / 按噪声档）
│   ├── estimate_noise.py   无 GT 的噪声强度估计（局部平面拟合残差）
│   ├── visualize_errors.py 误差着色的交互式可视化（Plotly HTML）
│   ├── plot_loss.py        训练损失曲线
│   └── results/            上述工具产出的 CSV 记录
│
└── experiments/            训练得到的 Jittor 权重（.pkl）与训练日志
```

## 数据布局

```
dataset_train/shapenet/<synset>/<model_id>/models/model_normalized.obj
dataset_test_noisy/shapenet/<synset>/<model_id>/noisy.npy
datalist/train.txt      # 每行一个 "shapenet/<synset>/<model_id>"
datalist/test.txt
```

`datalist/` 随官方 starter code 提供；若缺失，可用 `make_datalist.py` 从数据目录重建：

```bash
python make_datalist.py --root ./dataset_train \
    --marker models/model_normalized.obj --out ./datalist/train.txt
python make_datalist.py --root ./dataset_test_noisy \
    --marker noisy.npy --out ./datalist/test.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 完整复现流程

```bash
# ① 噪声强度分类器（ScaleNet），仅用赛方训练数据从零训练
python train_classifier_on_starter.py \
    --data_root ./dataset_train --datalist ./datalist/train.txt \
    --epochs 50 --batch_size 8 --save_dir experiments/classify

# ② ASDN 去噪主网络
python train_on_starter.py \
    --data_root ./dataset_train --datalist ./datalist/train.txt \
    --classify_ckpt experiments/classify/classify-epoch049.pkl \
    --epochs 80 --batch_size 8 --lr 5e-4 --save_dir experiments/asdn

# ③ 推理（A 榜最优配置：epoch039 权重 + niters=2）
python predict_on_starter.py \
    --ckpt experiments/asdn/asdn-epoch039.pkl \
    --data_root ./dataset_test_noisy --datalist ./datalist/test.txt \
    --niters 2 --out_root ./results/dataset_test_noisy

# ④ 自检后打包
python check_submission.py --datalist ./datalist/test.txt \
    --pred_root ./results/dataset_test_noisy --noisy_root ./dataset_test_noisy
cd results/dataset_test_noisy && zip -r ../../result.zip shapenet/
```

## 本地评测（不接触测试集 GT）

测试集没有 GT，因此本地验证集完全由**训练集**构造：从训练网格采样干净点云、
按与测试集相同的分布加拉普拉斯噪声，再用 `tools/evaluate.py` 按赛题公式打分。

```bash
python tools/make_eval_set.py --root ./dataset_train --datalist ./datalist/train.txt \
    --num_eval 30 --num_points 50000 \
    --out_gt ./eval_gt --out_noisy ./eval_noisy --out_mesh ./eval_mesh_normalized

python make_datalist.py --root ./eval_noisy --marker noisy.npy --out ./datalist/eval.txt
python predict_on_starter.py --ckpt experiments/asdn/asdn-epoch039.pkl \
    --data_root ./eval_noisy --datalist ./datalist/eval.txt \
    --niters 2 --out_root ./eval_predict

python tools/evaluate.py --pred_dir ./eval_predict --gt_dir ./eval_gt \
    --noisy_dir ./eval_noisy --mesh_dir ./eval_mesh_normalized \
    --save_csv ./tools/results/per_sample_scores_niters2.csv
```

本地 30 样本验证集上的 `--niters` 消融（`tools/results/*.csv`）：

| niters | CD 得分 | P2S 得分 | 总分 |
|---|---|---|---|
| 1 | 60.12 | 81.94 | 71.03 |
| **2** | **66.47** | **87.02** | **76.74** |
| 3 | 63.39 | 85.71 | 74.55 |

## 移植说明：原实现 → 本实现

| 原依赖 | 本实现的替代 |
|---|---|
| `pytorch_lightning`（Trainer / LightningModule） | `train_*_on_starter.py` 中的普通 Jittor 训练循环；模型是纯 `jt.nn.Module` |
| `pytorch3d.ops.knn_points`、`torch_cluster.fps` | `models/pointops_jt.py` 中的精确暴力 KNN 与精确迭代 FPS |
| `pointops` CUDA 扩展（`furthestsampling`/`queryandgroup`/`interpolation`） | `models/pointops_jt.py`（保持相同的 offset 格式 API） |
| `Chamfer3D` CUDA 扩展（`chamfer_3DDist`） | `models/InfoCD.py` 中基于 KNN 的 `chamfer_dist`，返回值约定相同 |
| `torch_geometric` 的 `MessagePassing`（EdgeConv） | `models/dynamic_edge_conv.py`：稠密 `(B,N,k,C)` 邻居聚合 + `jt.max` |
| `torch.utils.data.DataLoader` + `Dataset` | `jittor.dataset.Dataset`（数据集自带批处理，通过 `set_attrs` 配置） |
| `multiprocessing.Pool` 推理并行 | 顺序循环（fork 后的 Jittor/CUDA 上下文不安全） |

## 已知注意事项

1. **速度**：CUDA 扩展被替换为精确的暴力 Jittor 实现，结果一致但大点云上的
   KNN/FPS 明显更慢。参考耗时：本次训练中去噪主网络约 45 分钟/epoch，
   80 epoch 合计约 2.5 天（依 GPU 与数据规模而变）。
2. **`classifyNet.get_edge_index` 的自环剔除**假设每个点的最近邻是它自己
   （除非存在完全重合的点，否则成立）。
3. **`tools/evaluate.py` 的 P2S**：未安装 `point-cloud-utils` 时会退化为
   "点到网格顶点" 的近似，绝对数值会偏大；安装后使用 BVH 精确计算。
4. **BatchNorm**：Jittor 与 PyTorch 的 momentum 定义等默认值略有差异，
   与原 PyTorch 实现做数值逐位对比时会有小幅漂移。
