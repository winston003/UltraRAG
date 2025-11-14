#!/bin/bash

# 一键设置和启动脚本

echo "🚀 UltraRAG 商业案例分析聊天机器人"
echo "=================================================="

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 激活环境
echo "🔧 激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ultrarag

if [ $? -ne 0 ]; then
    echo "❌ 无法激活ultrarag环境"
    echo "💡 请先创建环境: conda create -n ultrarag python=3.10"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖包..."
python check_environment.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 依赖包安装失败"
    echo "💡 尝试修复pip: ./fix_pip_environment.sh"
    exit 1
fi

echo ""
echo "✅ 环境检查完成！"
echo "=================================================="
echo ""

# 启动服务
./start_business_chatbot.sh
