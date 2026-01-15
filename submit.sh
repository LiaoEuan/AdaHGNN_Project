#!/bin/bash

# ===================================================================
# 1. Slurm 资源调度设置 (SBATCH Config)
# ===================================================================
#SBATCH -J AdaHGNN_AVED          # 作业名称 (建议简短明确)
#SBATCH -o logs/%x-%j.out        # 标准输出: logs/作业名-ID.out (推荐把日志归档到logs文件夹)
#SBATCH -e logs/%x-%j.err        # 标准错误: logs/作业名-ID.err
#SBATCH -p gpu2node              # 分区名称 (gpu2node / gpu3node)
#SBATCH -c 8                     # CPU核心数 (对应 DataLoader num_workers)
#SBATCH --gres=gpu:1             # GPU数量
#SBATCH --mem=48G                # 内存 (LOSO 实验数据量大，建议 48G 保险)
#SBATCH -t 2-00:00:00            # 运行时间限制: 3天 (格式 D-HH:MM:SS)

# ===================================================================
# 2. 环境与路径准备 (Setup)
# ===================================================================
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Start:  $(date)"
echo "--------------------------------------------------------"

# [关键步骤] 切换到项目根目录
# 请修改为您实际存放代码的绝对路径
cd /share/home/yuan/LY/AdaHGNN_Project
echo "Working Dir: $(pwd)"

# 创建日志文件夹 (如果不存在)，保持根目录整洁
mkdir -p logs

# [关键步骤] 加载模块
module purge
module load cuda/11.8
module load anaconda3

# [关键步骤] 健壮的 Conda 激活方式
# 1. 获取 conda 基础路径并初始化 shell
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 2. 激活环境 (请修改为您实际的环境名称，如 hgnn 或 ly_torch)
ENV_NAME="hgnn"
conda activate $ENV_NAME

# 检查环境是否激活成功
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment '$ENV_NAME'!"
    exit 1
fi
echo "✅ Environment '$ENV_NAME' activated. Python: $(which python)"

# ===================================================================
# 3. 核心训练任务 (Execution)
# ===================================================================
echo "--------------------------------------------------------"
echo "Starting Training..."

# 运行命令
# -u: 禁用缓冲，让 print 立即输出到日志
# --config: 指定配置文件 (AVED 或 DTU)
/share/home/yuan/.conda/envs/hgnn/bin/python -u train.py --config configs/aved_config.yaml
# python -u train.py --config configs/dtu_config.yaml

# ===================================================================
# 4. 结束 (Finish)
# ===================================================================
echo "--------------------------------------------------------"
echo "Job Finished. Exit Code: $?"
echo "End Time: $(date)"
echo "========================================================"