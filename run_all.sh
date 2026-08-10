#!/usr/bin/env bash
#
# 第六届计图挑战赛 赛题二（点云降噪）—— 一键复现脚本
#
#   bash run_all.sh infer    仅推理：用随包提供的权重生成 A 榜提交文件（约几十分钟）
#   bash run_all.sh train    完整复现：从零训练分类器 + 去噪网络，再推理（约 2.5 天）
#
# 数据请按下列布局放好（与本脚本同级目录）：
#   dataset_train/shapenet/<synset>/<model_id>/models/model_normalized.obj
#   dataset_test_noisy/shapenet/<synset>/<model_id>/noisy.npy
#   datalist/train.txt, datalist/test.txt        （官方 starter code 提供）
#
# datalist 缺失时本脚本会自动用 make_datalist.py 重建。
#
set -euo pipefail

MODE="${1:-}"

DATA_TRAIN="./dataset_train"
DATA_TEST="./dataset_test_noisy"
DATALIST_TRAIN="./datalist/train.txt"
DATALIST_TEST="./datalist/test.txt"

# A 榜最优配置（与 提交说明文档.pdf 5.3 节一致）
ASDN_CKPT="experiments/asdn/asdn-epoch039.pkl"
CLASSIFY_CKPT="experiments/classify/classify-epoch049.pkl"
NITERS=2
PATCH_SIZE=1000
SEED_K=6
SEED_K_ALPHA=10

OUT_ROOT="./results/dataset_test_noisy"
BACKUP_DIR="./experiments_shipped"
RESULT_ZIP="$(pwd)/result.zip"

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[ -z "$MODE" ] && usage
case "$MODE" in infer|train) ;; *) echo "未知模式: $MODE"; usage ;; esac

step() { echo; echo "=============================================================="; echo "  $*"; echo "=============================================================="; }

# ---------------------------------------------------------------- 数据检查
step "步骤 0 / 检查数据与 datalist"
[ -d "$DATA_TEST" ] || { echo "找不到 $DATA_TEST，请先放好测试数据"; exit 1; }
if [ ! -f "$DATALIST_TEST" ]; then
    echo "datalist/test.txt 不存在，正在重建..."
    python make_datalist.py --root "$DATA_TEST" --marker noisy.npy --out "$DATALIST_TEST"
fi
if [ "$MODE" = "train" ]; then
    [ -d "$DATA_TRAIN" ] || { echo "找不到 $DATA_TRAIN，训练模式需要训练数据"; exit 1; }
    if [ ! -f "$DATALIST_TRAIN" ]; then
        echo "datalist/train.txt 不存在，正在重建..."
        python make_datalist.py --root "$DATA_TRAIN" \
            --marker models/model_normalized.obj --out "$DATALIST_TRAIN"
    fi
fi
echo "数据检查通过。"

# ---------------------------------------------------------------- 训练
if [ "$MODE" = "train" ]; then
    # 训练会按 epoch 编号覆盖同名权重（80 epoch 的第 39 个 epoch 正好写出
    # asdn-epoch039.pkl），这会盖掉随包提供的 A 榜权重。先备份，便于事后对比。
    if [ -f "$ASDN_CKPT" ] && [ ! -d "$BACKUP_DIR" ]; then
        step "步骤 0.5 / 备份随包提供的 A 榜权重到 $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        cp -r experiments/. "$BACKUP_DIR"/
        echo "已备份。训练结束后可用 $BACKUP_DIR/asdn/$(basename "$ASDN_CKPT") 与新权重对比。"
    fi

    step "步骤 1 / 训练 ScaleNet 噪声强度分类器（50 epoch）"
    python train_classifier_on_starter.py \
        --data_root "$DATA_TRAIN" \
        --datalist  "$DATALIST_TRAIN" \
        --epochs 50 --batch_size 8 --lr 5e-4 \
        --save_dir experiments/classify

    step "步骤 2 / 训练 ASDN 去噪主网络（80 epoch，约 2.5 天）"
    python train_on_starter.py \
        --data_root "$DATA_TRAIN" \
        --datalist  "$DATALIST_TRAIN" \
        --classify_ckpt "$CLASSIFY_CKPT" \
        --epochs 80 --batch_size 8 --lr 5e-4 \
        --save_dir experiments/asdn
fi

# ---------------------------------------------------------------- 推理
step "步骤 3 / 推理（$ASDN_CKPT, niters=$NITERS）"
[ -f "$ASDN_CKPT" ] || { echo "找不到权重 $ASDN_CKPT"; exit 1; }
python predict_on_starter.py \
    --ckpt      "$ASDN_CKPT" \
    --data_root "$DATA_TEST" \
    --datalist  "$DATALIST_TEST" \
    --niters "$NITERS" --patch_size "$PATCH_SIZE" \
    --seed_k "$SEED_K" --seed_k_alpha "$SEED_K_ALPHA" \
    --out_root  "$OUT_ROOT"

# ---------------------------------------------------------------- 自检 + 打包
step "步骤 4 / 提交前自检（点数一致性 / NaN / 数值范围）"
python check_submission.py \
    --datalist   "$DATALIST_TEST" \
    --pred_root  "$OUT_ROOT" \
    --noisy_root "$DATA_TEST"

step "步骤 5 / 打包为 A 榜提交格式"
rm -f "$RESULT_ZIP"
if command -v zip >/dev/null 2>&1; then
    ( cd "$OUT_ROOT" && zip -qr "$RESULT_ZIP" shapenet/ )
else
    echo "未找到 zip 命令，改用 Python 打包..."
    python - "$OUT_ROOT" "$RESULT_ZIP" <<'PY'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(os.path.join(src, 'shapenet')):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            zf.write(full, os.path.relpath(full, src))
PY
fi

# 不要盲目报告成功：确认 zip 真的生成了，且内容条目数与样本数一致。
if [ ! -s "$RESULT_ZIP" ]; then
    echo "错误：打包失败，$RESULT_ZIP 不存在或为空。" >&2
    exit 1
fi
counts=$(python - "$RESULT_ZIP" "$DATALIST_TEST" <<'PY'
import sys, zipfile
packed = sum(1 for n in zipfile.ZipFile(sys.argv[1]).namelist() if n.endswith('.npy'))
expected = sum(1 for ln in open(sys.argv[2]) if ln.strip())
print(packed, expected)
PY
)
n_packed=${counts% *}
n_expected=${counts#* }
echo "已生成 $RESULT_ZIP（$n_packed 个 denoised.npy，datalist 共 $n_expected 个样本）"
if [ "$n_packed" != "$n_expected" ]; then
    echo "错误：打包条目数 $n_packed 与样本数 $n_expected 不一致，请勿提交。" >&2
    exit 1
fi

step "完成"
echo "提交文件: $RESULT_ZIP"
if [ -d "$BACKUP_DIR" ]; then
    echo "随包提供的原始 A 榜权重已备份在: $BACKUP_DIR"
fi
