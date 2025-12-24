# UltraRAG 项目中的数据清洗提示词总结

## ⭐ 最终使用的清洗提示词

**当前生产环境实际使用的提示词** (位于 `/servers/custom/src/cleaner_optimized.py` 第140-150行):

```python
prompt = f"""请清洗以下文本，要求：
1. 移除无关内容和噪音
2. 修正明显的错误
3. 保持原意不变
4. 保持专业术语
5. 只返回清洗后的文本，不要解释

原文本：
{text}

清洗后的文本："""
```

**调用参数**:
- **模型**: `qwen-plus`
- **Temperature**: `0.1` (低温度保证输出稳定性)
- **Max Tokens**: `2000`
- **API端点**: `https://dashscope.aliyuncs.com/compatible-mode/v1`

**函数**: `_clean_with_llm()` 在 `cleaner_optimized.py` 中被 `clean_text_chunks()` 调用

---

## 📍 提示词位置索引

### 1. 主要指南文档（参考用，非生产环境）
**文件位置**: `/docs/technical/清洗数据指南.md`

该文档包含完整的文本清洗策略和LLM提示词设计模板。**注意：这是设计文档中的示例提示词，专门针对录音转文本场景，实际生产环境使用的是 cleaner_optimized.py 中的版本。**

#### 核心提示词模板（来自指南文档）：

```jinja
你是一个专业的文本清洗专家，请对以下录音转文本内容进行清洗：

清洗要求：
1. 修正语音识别错误（同音字、断句错误等）
2. 将口语化表达转为书面语
3. 去除重复内容和无意义的填充词
4. 修正标点符号和段落结构
5. 保持原意不变，提升可读性

原始文本：
{{text}}

请输出清洗后的文本：
```

**用途**: 设计参考文档，针对录音转文本的特定场景

### 2. 实际代码实现中的提示词（生产环境）

#### 2.1 优化版清洗模块 ⭐（当前使用）
**文件位置**: `/servers/custom/src/cleaner_optimized.py` (第140-150行)

```python
prompt = f"""请清洗以下文本，要求：
1. 移除无关内容和噪音
2. 修正明显的错误
3. 保持原意不变
4. 保持专业术语
5. 只返回清洗后的文本，不要解释

原文本：
{text}

清洗后的文本："""
```

**使用的LLM模型**: `qwen-plus`
**API端点**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
**参数设置**:
- `temperature=0.1` (低温度保证稳定性)
- `max_tokens=2000` (最大输出2000个token)

#### 2.2 基础清洗模块
**文件位置**: `/servers/custom/src/cleaner.py`

该文件提供了基础的清洗框架，但LLM清洗部分是占位符（第103行），实际清洗逻辑需要集成。

## 📂 相关配置文件

### Pipeline配置文件

1. **基础清洗Pipeline**: `/examples/text_cleaning.yaml`
   - 完整的清洗流程配置
   - 包含加载、分块、清洗、合并、保存等步骤

2. **简单分块清洗**: `/examples/text_cleaning_simple_chunked.yaml`
   - 简化版的分块清洗流程

3. **高级分块清洗**: `/examples/text_cleaning_chunked.yaml`
   - 标准分块清洗配置

4. **递归分块清洗**: `/examples/text_cleaning_recursive_chunked.yaml`
   - 递归分块策略

5. **基础测试**: `/examples/test_basic_clean.yaml`
   - 清洗功能测试配置

6. **数据处理Pipeline**: `/examples/data_processing_pipeline.yaml`
   - 综合数据处理流程

### 批处理脚本

**文件位置**: `/script/batch_clean.py`

用于批量清洗文本文件的Python脚本，支持：
- 批量处理多个文件
- 自定义块大小
- 多种文本格式（.txt, .md, .json）

## 🔧 清洗策略详解

### 1. 分块处理策略
- **推荐块大小**: 1000-2000字/块
- **块间重叠**: 100-200字
- **目的**: 避免LLM上下文限制，保证语义连续性

### 2. 清洗重点
1. 语音识别错误修正
2. 口语化表达规范化
3. 重复内容去除
4. 标点符号修正
5. 逻辑结构整理

### 3. 工具函数

**优化版模块** (`cleaner_optimized.py`) 提供：

- `_clean_with_rules(text)` - 基于规则的文本清洗
- `_clean_with_llm(text, api_key, api_base)` - 使用LLM清洗文本
- `chunk_text(text_content, chunk_size, chunk_overlap)` - 文本分块
- `clean_text_chunks(chunks, use_llm)` - 批量清洗文本块
- `merge_chunks(chunks)` - 合并文本块
- `save_text_file(text, file_path)` - 保存清洗结果
- `load_text_file(file_path)` - 加载文本文件

## 📖 使用示例

### 1. 使用UltraRAG Pipeline清洗文本

```bash
# 运行基础清洗Pipeline
ultrarag run examples/text_cleaning.yaml

# 运行分块清洗
ultrarag run examples/text_cleaning_chunked.yaml
```

### 2. 批量清洗文本文件

```bash
python script/batch_clean.py \
    --input_dir /path/to/raw_texts \
    --output_dir /path/to/cleaned_texts \
    --chunk_size 1500
```

### 3. 在Python代码中使用

```python
from servers.custom.src.cleaner_optimized import _clean_with_llm

# 使用LLM清洗文本
cleaned_text = _clean_with_llm(
    text="要清洗的文本内容",
    api_key="your-api-key",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

## 🎯 提示词设计原则

根据项目中的实现，数据清洗提示词遵循以下原则：

1. **明确任务目标**: 清晰说明"清洗"的定义
2. **具体要求列表**: 使用编号列表列出5个具体要求
3. **简洁输出**: 要求只返回清洗后的文本，不要解释
4. **保持原意**: 强调保持专业术语和原意不变
5. **低温度参数**: 使用temperature=0.1确保输出稳定性

## 📝 关键配置参数

### LLM清洗参数 (cleaner_optimized.py)
```python
{
    "model": "qwen-plus",
    "temperature": 0.1,
    "max_tokens": 2000,
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"
}
```

### 分块参数推荐
```yaml
chunk_size: 1000-2000      # 每块字符数
chunk_overlap: 100-200     # 块间重叠字符数
chunk_strategy: "sentence" # 按句子分块
```

## 🔗 相关文档链接

- 清洗数据指南: `/docs/technical/清洗数据指南.md`
- 项目完成总结: `/docs/archive/project-summaries/项目完成总结.md`
- 主README: `/README.md`

## 💡 最佳实践

1. **大文本处理**: 对于超过3万字的文本，使用分块策略
2. **错误处理**: 使用降级机制，LLM失败时回退到规则清洗
3. **性能优化**: 
   - 并行处理多个块
   - 使用GPU加速
   - 实现缓存机制
4. **质量保证**: 清洗后进行评估验证

---

**最后更新**: 2025-12-24
**维护者**: UltraRAG Team
