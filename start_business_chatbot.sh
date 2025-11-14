#!/bin/bash

# 商业案例分析聊天机器人启动脚本
# 专为学生设计的RAG聊天机器人系统

echo "🚀 启动商业案例分析聊天机器人..."
echo "📊 目标用户: 学生群体"
echo "📚 内容领域: 商业案例拆解"
echo "=================================================="

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 激活ultrarag环境
echo "🔧 激活conda环境: ultrarag"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ultrarag

if [ $? -ne 0 ]; then
    echo "❌ 无法激活ultrarag环境，请确保环境已创建"
    echo "💡 创建环境命令: conda create -n ultrarag python=3.10"
    exit 1
fi

# 检查并安装必要的Python包
echo "📦 检查并安装必要的Python包..."
required_packages=("streamlit" "psutil" "pyyaml")

for package in "${required_packages[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        echo "⚠️  缺少依赖包: $package，正在安装..."
        pip install $package
        if [ $? -eq 0 ]; then
            echo "✅ $package 安装成功"
        else
            echo "❌ $package 安装失败"
            exit 1
        fi
    else
        echo "✅ $package 已安装"
    fi
done

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python环境异常，请检查conda环境"
    exit 1
fi

echo "✅ Python环境: $(python --version)"

# 检查必要的依赖包
echo "📦 检查依赖包..."
required_packages=("streamlit" "ultrarag" "asyncio")

for package in "${required_packages[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        echo "⚠️  缺少依赖包: $package"
        echo "💡 请运行: pip install $package"
    else
        echo "✅ $package 已安装"
    fi
done

# 检查索引是否存在
echo "🔍 检查索引文件..."
if [ -d "data/lancedb" ]; then
    echo "✅ 找到LanceDB索引目录"
    if [ -d "data/lancedb/documents.lance" ]; then
        echo "✅ 找到documents表"
    else
        echo "⚠️  未找到documents表，请检查索引是否正确构建"
    fi
else
    echo "❌ 未找到索引目录，请先构建索引"
    echo "💡 索引路径: data/lancedb"
    exit 1
fi

# 检查配置文件
echo "📋 检查配置文件..."
if [ -f "config/chatbot.yaml" ]; then
    echo "✅ 找到聊天机器人配置文件"
else
    echo "❌ 未找到配置文件: config/chatbot.yaml"
    exit 1
fi

# 检查提示词模板
if [ -f "prompt/qa_rag_multiround.jinja" ]; then
    echo "✅ 找到提示词模板"
else
    echo "❌ 未找到提示词模板: prompt/qa_rag_multiround.jinja"
    exit 1
fi

# 检查自定义服务
if [ -f "servers/custom/src/chatbot_custom.py" ]; then
    echo "✅ 找到自定义服务组件"
else
    echo "❌ 未找到自定义服务: servers/custom/src/chatbot_custom.py"
    exit 1
fi

# 检查端口并杀死占用进程的函数
check_and_kill_port() {
    local port=$1
    local service_name=$2
    
    # 检查端口是否被占用
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  端口 $port 已被占用，正在查找占用进程..."
        # 获取占用端口的进程PID
        local pid=$(lsof -ti :$port)
        if [ ! -z "$pid" ]; then
            echo "🔧 正在终止占用端口 $port 的进程 (PID: $pid)..."
            kill -9 $pid 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ 成功终止占用端口 $port 的进程"
                # 等待进程完全终止
                sleep 2
            else
                echo "❌ 无法终止占用端口 $port 的进程"
                return 1
            fi
        fi
    else
        echo "✅ 端口 $port 未被占用"
    fi
    return 0
}

echo "=================================================="
echo "🎯 所有检查完成，准备启动服务..."
echo "=================================================="

# Ensure logs directory exists
mkdir -p logs

# 检查并清理可能占用的端口 (只检查实际使用的端口)
echo "🔍 检查端口占用情况..."
check_and_kill_port 8501 "Streamlit应用"

# 启动后台服务
echo "🔧 启动后台MCP服务..."

# Use conda run to ensure correct environment; redirect logs and record PIDs

# 设置必要的环境变量
export LLM_API_KEY="${LLM_API_KEY:-your-api-key-here}"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
echo "🔑 设置API密钥环境变量"
echo "📂 设置Python路径: $PYTHONPATH"

# 启动检索服务
echo "📚 启动检索服务..."
nohup conda run -n ultrarag --no-capture-output env LLM_API_KEY="$LLM_API_KEY" PYTHONPATH="$(pwd)/src:$(pwd)/servers/retriever/src:$PYTHONPATH" python -m servers.retriever.src.retriever > logs/retriever.log 2>&1 &
RETRIEVER_PID=$!
echo $RETRIEVER_PID > logs/retriever.pid
echo "✅ 检索服务已启动 (PID: $RETRIEVER_PID), 日志: logs/retriever.log"

# 启动提示词服务
echo "💭 启动提示词服务..."
nohup conda run -n ultrarag --no-capture-output env LLM_API_KEY="$LLM_API_KEY" PYTHONPATH="$(pwd)/src:$PYTHONPATH" python -m servers.prompt.src.prompt > logs/prompt.log 2>&1 &
PROMPT_PID=$!
echo $PROMPT_PID > logs/prompt.pid
echo "✅ 提示词服务已启动 (PID: $PROMPT_PID), 日志: logs/prompt.log"

# 启动生成服务
echo "🤖 启动生成服务..."
nohup conda run -n ultrarag --no-capture-output env LLM_API_KEY="$LLM_API_KEY" PYTHONPATH="$(pwd)/src:$PYTHONPATH" python -m servers.generation.src.generation > logs/generation.log 2>&1 &
GENERATION_PID=$!
echo $GENERATION_PID > logs/generation.pid
echo "✅ 生成服务已启动 (PID: $GENERATION_PID), 日志: logs/generation.log"

# 启动自定义服务
echo "⚙️  启动自定义服务..."
# 修复 anyio 兼容性问题，通过指定 Python 路径来确保使用正确的库版本
nohup conda run -n ultrarag --no-capture-output env LLM_API_KEY="$LLM_API_KEY" PYTHONPATH="$(pwd)/src:$PYTHONPATH" python -m servers.custom.src.chatbot_custom > logs/custom.log 2>&1 &
CUSTOM_PID=$!
echo $CUSTOM_PID > logs/custom.pid
echo "✅ 自定义服务已启动 (PID: $CUSTOM_PID), 日志: logs/custom.log"

# 等待服务启动
echo "⏳ 等待服务初始化..."
sleep 8

# 启动Streamlit（用 conda run 确保环境）
echo "🌟 启动商业案例分析聊天机器人界面..."
echo "🔗 访问地址: http://localhost:8501"
# 修复移动端访问地址显示问题
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname 2>/dev/null)
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="localhost"
fi
echo "📱 移动端访问: http://$LOCAL_IP:8501"
echo "=================================================="
echo "🎓 欢迎使用商业案例分析助手！"
echo "📊 专为学生设计的智能学习平台"
echo "💡 按 Ctrl+C 停止所有服务"
echo "=================================================="

# 清理过大的日志文件（保留最后1000行）
if [ -f "logs/streamlit.log" ] && [ $(stat -f%z "logs/streamlit.log" 2>/dev/null || stat -c%s "logs/streamlit.log" 2>/dev/null) -gt 10485760 ]; then
    tail -1000 logs/streamlit.log > logs/streamlit.log.tmp
    mv logs/streamlit.log.tmp logs/streamlit.log
fi

nohup conda run -n ultrarag --no-capture-output env LLM_API_KEY="$LLM_API_KEY" PYTHONPATH="$(pwd)/src:$PYTHONPATH" python -m streamlit run chatbot_app.py --server.port 8501 --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo $STREAMLIT_PID > logs/streamlit.pid

echo "✅ Streamlit 已启动 (PID: $STREAMLIT_PID), 日志: logs/streamlit.log"

# 清理函数
cleanup() {
    echo ""
    echo "🛑 正在停止所有服务..."
    echo "📚 停止检索服务..."
    kill $RETRIEVER_PID 2>/dev/null || true
    echo "💭 停止提示词服务..."
    kill $PROMPT_PID 2>/dev/null || true
    echo "🤖 停止生成服务..."
    kill $GENERATION_PID 2>/dev/null || true
    echo "⚙️  停止自定义服务..."
    kill $CUSTOM_PID 2>/dev/null || true
    echo "🌟 停止Streamlit应用..."
    kill $STREAMLIT_PID 2>/dev/null || true
    echo "✅ 所有服务已停止"
    echo "👋 感谢使用商业案例分析助手！"
    exit 0
}

# 注册清理函数
trap cleanup SIGINT SIGTERM

# 保持脚本运行
echo "🔄 服务运行中，按 Ctrl+C 停止..."
wait