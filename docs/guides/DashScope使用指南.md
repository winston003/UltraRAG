# UltraRAG DashScope 使用指南

## 🎯 概述

本指南介绍如何使用UltraRAG框架配合阿里云DashScope API进行文本处理和向量索引构建。我们提供了多个工具来满足不同的使用场景。

## 📁 文件说明

### 核心脚本
- **`process_dashscope.py`** - 单文件处理脚本（推荐）
- **`batch_process_dashscope.py`** - 批量处理脚本
- **`benchmark_dashscope.py`** - 性能测试脚本

### 配置文件
- **`examples/dashscope_example.yaml`** - 配置示例
- **`README-text-processing.md`** - 详细文档

### 辅助工具
- **`process_simple.py`** - 简化版本（需要OpenAI兼容API）
- **`process_and_index_text.py`** - 标准版本（需要OpenAI兼容API）
- **`clean_index.py`** - 索引清理工具

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活UltraRAG环境
conda activate ultrarag

# 设置阿里云API密钥
export ALI_EMBEDDING_API_KEY="your_aliyun_api_key_here"
```

### 2. 单文件处理

```bash
# 基本用法
python process_dashscope.py --input_file data/your_text.txt --overwrite

# 自定义参数
python process_dashscope.py \
  --input_file data/your_text.txt \
  --chunk_size 800 \
  --model text-embedding-v1 \
  --tokenizer bert-base-chinese \
  --overwrite
```

### 3. 批量处理

```bash
# 处理目录中的所有文本文件
python batch_process_dashscope.py --input_dir data/corpus --file_pattern "*.txt"

# 限制处理文件数量
python batch_process_dashscope.py \
  --input_dir data/corpus \
  --file_pattern "*.txt" \
  --max_files 10 \
  --chunk_size 800 \
  --model text-embedding-v1
```

### 4. 性能测试

```bash
# 测试不同模型的性能
python benchmark_dashscope.py \
  --input_file data/word_chunk.txt \
  --models text-embedding-v1 text-embedding-v2 text-embedding-v3 \
  --chunk_sizes 400 800 1200 \
  --iterations 3
```

## 🔧 参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_file` | str | 必需 | 输入文本文件路径 |
| `--output_dir` | str | data/processed | 输出目录 |
| `--chunk_size` | int | 800 | 分块大小（字符数） |
| `--chunk_strategy` | str | recursive | 分块策略 |
| `--model` | str | text-embedding-v1 | DashScope模型 |
| `--tokenizer` | str | bert-base-chinese | 分词器 |
| `--overwrite` | flag | False | 覆盖现有文件 |

### 批量处理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_dir` | str | 必需 | 输入目录路径 |
| `--file_pattern` | str | *.txt | 文件匹配模式 |
| `--max_files` | int | None | 最大处理文件数 |

### 性能测试参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--models` | list | v1,v2,v3 | 测试的模型列表 |
| `--chunk_sizes` | list | 400,800,1200 | 测试的分块大小 |
| `--iterations` | int | 3 | 每个测试的重复次数 |
| `--output_file` | str | benchmark_results.json | 结果输出文件 |

## 📊 支持的模型

| 模型 | 向量维度 | 特点 | 推荐场景 |
|------|----------|------|----------|
| `text-embedding-v1` | 1536 | 通用性强，性能稳定 | 推荐使用 |
| `text-embedding-v2` | 1536 | 优化版本，质量更高 | 高质量要求 |
| `text-embedding-v3` | 1024 | 轻量级，速度快 | 大规模处理 |

## 📈 性能优化建议

### 1. 分块大小选择
- **小文档（<1000字符）**: 400-600字符
- **中等文档（1000-5000字符）**: 800-1200字符
- **大文档（>5000字符）**: 1200-2000字符

### 2. 批量处理优化
- 使用`--max_files`限制并发文件数
- 根据内存大小调整`--chunk_size`
- 考虑使用`text-embedding-v3`提高处理速度

### 3. 内存管理
- 大文件处理时监控内存使用
- 必要时分批处理
- 使用SSD存储提高I/O性能

## 🔍 输出文件说明

### 单文件处理输出
```
data/processed/
├── {filename}_chunks.jsonl    # 分块结果
embedding/
├── embedding_{filename}.npy   # 嵌入向量
index/
├── index_{filename}.index     # Faiss索引
```

### 批量处理输出
```
data/processed/
├── file1_chunks.jsonl
├── file2_chunks.jsonl
├── ...
embedding/
├── embedding_file1.npy
├── embedding_file2.npy
├── ...
index/
├── index_file1.index
├── index_file2.index
├── ...
```

## 🛠️ 故障排除

### 常见问题

1. **404错误**
   - 检查API密钥是否正确
   - 确认模型名称是否正确
   - 验证网络连接

2. **内存不足**
   - 减小`--chunk_size`
   - 使用`--max_files`限制并发
   - 考虑使用`text-embedding-v3`

3. **处理速度慢**
   - 使用`text-embedding-v3`模型
   - 调整分块大小
   - 检查网络延迟

4. **文件权限错误**
   - 检查输出目录权限
   - 确保有写入权限

### 调试技巧

1. **启用详细日志**
   ```bash
   export ULTRA_RAG_LOG_LEVEL=DEBUG
   ```

2. **测试API连接**
   ```bash
   python -c "
   import os
   from process_dashscope import DashScopeEmbedding
   client = DashScopeEmbedding(os.getenv('ALI_EMBEDDING_API_KEY'))
   result = await client.embed_texts(['测试'])
   print('API连接正常')
   "
   ```

3. **检查输出文件**
   ```bash
   # 检查分块文件
   head -5 data/processed/*_chunks.jsonl
   
   # 检查向量文件
   python -c "import numpy as np; print(np.load('embedding/embedding_word_chunk.npy').shape)"
   ```

## 📚 高级用法

### 1. 自定义分块策略

```python
# 在process_dashscope.py中修改
chunk_result = await ToolCall.corpus.chunk_documents(
    chunk_strategy="semantic",  # 使用语义分块
    chunk_size=800,
    raw_data=raw_data['raw_data'],
    output_path=chunks_path,
    tokenizer_name_or_path="bert-base-chinese"
)
```

### 2. 集成到现有项目

```python
from process_dashscope import DashScopeEmbedding
import asyncio

async def process_text(text: str, model: str = "text-embedding-v1"):
    client = DashScopeEmbedding(
        api_key=os.getenv("ALI_EMBEDDING_API_KEY"),
        model=model
    )
    embeddings = await client.embed_texts([text])
    return embeddings[0]

# 使用示例
embedding = asyncio.run(process_text("你的文本内容"))
```

### 3. 性能监控

```python
import time
import psutil

def monitor_performance():
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # 执行处理...
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"处理时间: {end_time - start_time:.2f}s")
    print(f"内存使用: {end_memory - start_memory:.2f}MB")
```

## 📞 技术支持

如果遇到问题，请：

1. 查看错误日志
2. 检查环境配置
3. 参考故障排除部分
4. 提交Issue到项目仓库

## 📄 许可证

本项目遵循MIT许可证，详见LICENSE文件。
