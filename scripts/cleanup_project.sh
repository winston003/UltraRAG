#!/bin/bash
# UltraRAG 项目清理脚本
# 基于项目文件清理分析报告执行清理操作

set -e

echo "🧹 UltraRAG 项目清理脚本"
echo "=================================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 确认执行
read -p "⚠️  此操作将删除和移动文件，是否继续？(y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "❌ 操作已取消"
    exit 0
fi

echo ""
echo "📋 开始执行清理..."

# ============================================
# 阶段1: 删除系统临时文件和备份文件
# ============================================
echo ""
echo "🔴 阶段1: 删除临时文件和备份文件..."

# 删除 .DS_Store 文件
echo "  删除 .DS_Store 文件..."
find . -name ".DS_Store" -type f -delete 2>/dev/null || true
echo "  ${GREEN}✅ .DS_Store 文件已删除${NC}"

# 删除备份文件
if [ -f "servers/prompt/parameter.yaml_bak" ]; then
    rm servers/prompt/parameter.yaml_bak
    echo "  ${GREEN}✅ 删除备份文件: servers/prompt/parameter.yaml_bak${NC}"
fi

if [ -f "servers/custom/src/custom_backup.py" ]; then
    rm servers/custom/src/custom_backup.py
    echo "  ${GREEN}✅ 删除备份文件: servers/custom/src/custom_backup.py${NC}"
fi

# 更新 .gitignore
if [ -f ".gitignore" ]; then
    if ! grep -q "__pycache__/" .gitignore; then
        echo "__pycache__/" >> .gitignore
        echo "  ${GREEN}✅ 已更新 .gitignore${NC}"
    fi
    if ! grep -q ".DS_Store" .gitignore; then
        echo ".DS_Store" >> .gitignore
        echo "  ${GREEN}✅ 已更新 .gitignore${NC}"
    fi
fi

# ============================================
# 阶段2: 创建文档目录结构
# ============================================
echo ""
echo "📦 阶段2: 创建文档目录结构..."

mkdir -p docs/archive/{work-records,project-summaries,old-proposals}
mkdir -p docs/technical
mkdir -p docs/guides
mkdir -p docs/api
mkdir -p examples/tools

echo "  ${GREEN}✅ 文档目录结构已创建${NC}"

# ============================================
# 阶段3: 归档工作文档
# ============================================
echo ""
echo "📚 阶段3: 归档工作文档..."

# 移动工作记录
for file in work_todo.md working.md 代码整理.md 代码整理使用说明.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/archive/work-records/
        echo "  ${GREEN}✅ 已归档: $file${NC}"
    fi
done

# 移动旧方案
for file in 优化001.md 方案1.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/archive/old-proposals/
        echo "  ${GREEN}✅ 已归档: $file${NC}"
    fi
done

# 移动项目总结
for file in 项目完成总结.md 优化实施指南.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/archive/project-summaries/
        echo "  ${GREEN}✅ 已归档: $file${NC}"
    fi
done

# ============================================
# 阶段4: 整理技术文档
# ============================================
echo ""
echo "📋 阶段4: 整理技术文档..."

# 移动技术分析文档
if [ -f "技术分析报告.md" ]; then
    mv 技术分析报告.md docs/technical/
    echo "  ${GREEN}✅ 已移动: 技术分析报告.md${NC}"
fi

# 移动优化代码示例中的文档
if [ -d "优化代码示例" ]; then
    for file in 优化代码示例/*.md; do
        if [ -f "$file" ]; then
            mv "$file" docs/technical/
            echo "  ${GREEN}✅ 已移动: $(basename $file)${NC}"
        fi
    done
fi

# 移动docs目录下的技术文档
for file in docs/上线前优化方案.md docs/优化设想.md docs/可扩展索引规划.md docs/元查询+稀疏查询.md docs/清洗数据指南.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/technical/
        echo "  ${GREEN}✅ 已移动: $(basename $file)${NC}"
    fi
done

# ============================================
# 阶段5: 整理使用指南
# ============================================
echo ""
echo "📖 阶段5: 整理使用指南..."

if [ -f "DashScope使用指南.md" ]; then
    mv DashScope使用指南.md docs/guides/
    echo "  ${GREEN}✅ 已移动: DashScope使用指南.md${NC}"
fi

if [ -f "CODEBUDDY.md" ]; then
    mv CODEBUDDY.md docs/guides/development.md
    echo "  ${GREEN}✅ 已移动: CODEBUDDY.md -> docs/guides/development.md${NC}"
fi

if [ -f "docs/MVP版本部署.md" ]; then
    mv docs/MVP版本部署.md docs/guides/deployment.md
    echo "  ${GREEN}✅ 已移动: docs/MVP版本部署.md -> docs/guides/deployment.md${NC}"
fi

# ============================================
# 阶段6: 整理工具脚本
# ============================================
echo ""
echo "🔧 阶段6: 整理工具脚本..."

# 移动工具脚本
for file in process_dashscope.py benchmark_dashscope.py run_evaluation.py; do
    if [ -f "$file" ]; then
        mv "$file" examples/tools/
        echo "  ${GREEN}✅ 已移动: $file -> examples/tools/${NC}"
    fi
done

# 创建工具README
if [ ! -f "examples/tools/README.md" ]; then
    cat > examples/tools/README.md << 'EOF'
# UltraRAG 工具脚本

本目录包含UltraRAG项目的辅助工具脚本。

## 工具列表

- `process_dashscope.py` - DashScope文本处理和索引构建工具
- `benchmark_dashscope.py` - DashScope模型性能测试
- `run_evaluation.py` - RAG评估脚本

## 使用方法

详见各脚本的文档注释。

## 注意事项

这些工具脚本是辅助性的，不是核心应用运行所必需的。
如果需要使用，请确保已安装相关依赖。
EOF
    echo "  ${GREEN}✅ 已创建: examples/tools/README.md${NC}"
fi

# ============================================
# 完成
# ============================================
echo ""
echo "=================================================="
echo "${GREEN}✅ 清理完成！${NC}"
echo ""
echo "📊 清理摘要:"
echo "  - 已删除临时文件和备份文件"
echo "  - 已归档工作文档到 docs/archive/"
echo "  - 已整理技术文档到 docs/technical/"
echo "  - 已整理使用指南到 docs/guides/"
echo "  - 已整理工具脚本到 examples/tools/"
echo ""
echo "📝 下一步建议:"
echo "  1. 检查清理结果: git status"
echo "  2. 查看清理报告: docs/项目文件清理分析报告.md"
echo "  3. 更新文档链接（如需要）"
echo "  4. 测试应用启动: ./start_business_chatbot.sh"
echo ""
echo "⚠️  注意: 请确认所有文件移动正确后，再提交到版本控制"
echo "=================================================="

