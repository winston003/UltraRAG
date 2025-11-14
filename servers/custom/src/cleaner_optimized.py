"""
优化后的数据清洗模块
包含完整的文本清洗、分块、验证功能
"""
import json
import logging
import os
import re
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ultrarag.server import UltraRAG_MCP_Server
except ImportError:
    class UltraRAG_MCP_Server:
        def __init__(self, name: str):
            self.name = name
        def tool(self, output=None):
            def decorator(func):
                return func
            return decorator
        def run(self, transport="stdio"):
            pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = UltraRAG_MCP_Server("custom")

# ==================== 工具函数 ====================

def retry_on_error(max_retries=3, delay=1):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} 失败（已重试{max_retries}次）: {e}")
                        raise
                    logger.warning(f"{func.__name__} 失败，{delay}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(delay)
        return wrapper
    return decorator

def validate_text(text: str) -> Tuple[bool, str]:
    """验证文本质量
    
    Returns:
        (is_valid, error_message)
    """
    if not text:
        return False, "文本为空"
    
    if len(text) < 10:
        return False, "文本过短（少于10字符）"
    
    if len(text) > 10000000:  # 10MB
        return False, "文本过长（超过10M字符）"
    
    # 检查是否包含有效内容
    if not re.search(r'[\u4e00-\u9fff\w]', text):
        return False, "文本不包含有效字符"
    
    # 检查特殊字符比例
    special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff。，！？；：""''（）、]', text))
    if len(text) > 0 and special_chars / len(text) > 0.5:
        return False, "特殊字符比例过高"
    
    return True, ""

def preprocess_text(text: str) -> str:
    """预处理文本"""
    # 1. 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. 移除零宽字符
    text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
    
    # 3. 统一空格
    text = re.sub(r'[\t\xa0]', ' ', text)
    
    # 4. 移除多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

# ==================== 清洗功能 ====================

def _clean_with_rules(text: str) -> str:
    """基于规则的文本清洗"""
    # 1. 移除多余空白
    text = re.sub(r' +', ' ', text)
    
    # 2. 移除特殊字符（保留中文、英文、数字、常用标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：""''（）、\-\.]', '', text)
    
    # 3. 移除重复标点
    text = re.sub(r'([。！？])\1+', r'\1', text)
    text = re.sub(r'([，；：])\1+', r'\1', text)
    
    # 4. 统一标点符号
    text = text.replace('，，', '，')
    text = text.replace('。。', '。')
    
    # 5. 移除行首行尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    # 6. 移除首尾空白
    text = text.strip()
    
    return text

@retry_on_error(max_retries=3, delay=2)
def _clean_with_llm(text: str, api_key: str, api_base: str = None) -> str:
    """使用LLM清洗文本"""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("OpenAI库未安装，降级到规则清洗")
        return _clean_with_rules(text)
    
    if not api_base:
        api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    prompt = f"""请清洗以下文本，要求：
1. 移除无关内容和噪音
2. 修正明显的错误
3. 保持原意不变
4. 保持专业术语
5. 只返回清洗后的文本，不要解释

原文本：
{text}

清洗后的文本："""
    
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        
        cleaned = response.choices[0].message.content.strip()
        return cleaned
    except Exception as e:
        logger.error(f"LLM清洗失败: {e}")
        # 降级到规则清洗
        return _clean_with_rules(text)

# ==================== 分块功能 ====================

def _chunk_by_size(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """按大小分块（改进版）"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 如果不是最后一块，寻找合适的分割点
        if end < len(text):
            # 优先级：段落 > 句子 > 短语 > 词语
            split_patterns = [
                ('\n\n', 200),  # 段落
                ('。', 100),     # 句子
                ('！', 100),
                ('？', 100),
                ('，', 50),      # 短语
                ('；', 50),
                (' ', 20)        # 词语
            ]
            
            found = False
            for pattern, search_range in split_patterns:
                for i in range(min(search_range, chunk_size // 4)):
                    if end - i <= start:
                        break
                    if text[end - i:end - i + len(pattern)] == pattern:
                        end = end - i + len(pattern)
                        found = True
                        break
                if found:
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 计算下一个块的起始位置
        start = end - chunk_overlap if chunk_overlap > 0 else end
        
        if start >= len(text):
            break
    
    return chunks

def _chunk_by_paragraphs(
    paragraphs: List[str],
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """按段落分块"""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para)
        
        # 如果单个段落超过chunk_size，需要拆分
        if para_size > chunk_size:
            # 先保存当前块
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # 拆分大段落
            sub_chunks = _chunk_by_size(para, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
        
        # 如果加上这个段落会超过chunk_size
        elif current_size + para_size > chunk_size:
            # 保存当前块
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            # 开始新块（可能包含重叠）
            if chunk_overlap > 0 and current_chunk:
                overlap_paras = []
                overlap_size = 0
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= chunk_overlap:
                        overlap_paras.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                current_chunk = overlap_paras
                current_size = overlap_size
            else:
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        else:
            # 添加到当前块
            current_chunk.append(para)
            current_size += para_size
    
    # 保存最后一个块
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

# ==================== MCP工具方法 ====================

@app.tool(output="text_content->text_chunks")
def chunk_text(
    text_content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    respect_paragraphs: bool = True
) -> Dict[str, List[str]]:
    """改进的文本分块
    
    Args:
        text_content: 输入文本内容
        chunk_size: 每块的字符数（默认1000）
        chunk_overlap: 块之间的重叠字符数（默认150）
        respect_paragraphs: 是否尊重段落边界（默认True）
    
    Returns:
        包含文本块列表的字典
    """
    if not text_content or not text_content.strip():
        logger.warning("输入文本为空")
        return {"text_chunks": []}
    
    # 预处理
    text = preprocess_text(text_content.strip())
    
    # 验证
    is_valid, error = validate_text(text)
    if not is_valid:
        logger.error(f"文本验证失败: {error}")
        return {"text_chunks": []}
    
    # 如果文本长度小于chunk_size，直接返回
    if len(text) <= chunk_size:
        logger.info(f"文本长度({len(text)})小于分块大小，返回单个块")
        return {"text_chunks": [text]}
    
    # 分块
    if respect_paragraphs:
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = _chunk_by_paragraphs(paragraphs, chunk_size, chunk_overlap)
    else:
        chunks = _chunk_by_size(text, chunk_size, chunk_overlap)
    
    logger.info(f"文本分块完成，共生成 {len(chunks)} 个块")
    return {"text_chunks": chunks}

@app.tool(output="chunks->cleaned_chunks")
def clean_text_chunks(
    chunks: List[str],
    use_llm: bool = False,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
) -> Dict[str, List[str]]:
    """对文本块进行清洗
    
    Args:
        chunks: 文本块列表
        use_llm: 是否使用LLM清洗（默认False）
        api_key: LLM API密钥
        api_base: LLM API地址
    
    Returns:
        包含清洗后文本块的字典
    """
    if not chunks:
        logger.warning("没有文本块需要清洗")
        return {"cleaned_chunks": []}
    
    cleaned_chunks = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"正在清洗第 {i+1}/{len(chunks)} 个文本块")
        
        try:
            if use_llm and api_key:
                cleaned_chunk = _clean_with_llm(chunk, api_key, api_base)
            else:
                cleaned_chunk = _clean_with_rules(chunk)
            
            cleaned_chunks.append(cleaned_chunk)
        except Exception as e:
            logger.error(f"清洗第 {i+1} 个块失败: {e}")
            # 失败时使用原文本
            cleaned_chunks.append(chunk)
    
    logger.info(f"完成 {len(cleaned_chunks)} 个文本块的清洗")
    return {"cleaned_chunks": cleaned_chunks}

@app.tool(output="chunks->final_text")
def merge_chunks(chunks: List[str]) -> Dict[str, str]:
    """合并文本块
    
    Args:
        chunks: 要合并的文本块列表
    
    Returns:
        包含合并后文本的字典
    """
    if not chunks:
        logger.warning("没有文本块需要合并")
        return {"final_text": ""}
    
    # 合并文本块，使用双换行分隔
    merged_text = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    
    logger.info(f"成功合并 {len(chunks)} 个文本块，总长度: {len(merged_text)} 字符")
    return {"final_text": merged_text}

@app.tool(output="text->save_result")
def save_text_file(text: str, file_path: str) -> Dict[str, str]:
    """保存文本到文件
    
    Args:
        text: 要保存的文本内容
        file_path: 保存路径
    
    Returns:
        保存结果
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"文本已保存到: {file_path}")
        return {"save_result": f"文本已成功保存到 {file_path}，大小: {len(text)} 字符"}
    except Exception as e:
        error_msg = f"保存文件失败: {str(e)}"
        logger.error(error_msg)
        return {"save_result": error_msg}

@app.tool(output="file_path->text_content")
def load_text_file(file_path: str) -> Dict[str, str]:
    """加载文本文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        包含文本内容的字典
    """
    try:
        if not os.path.exists(file_path):
            return {"text_content": f"错误: 文件不存在 {file_path}"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"成功加载文件: {file_path}，大小: {len(content)} 字符")
        return {"text_content": content}
    except Exception as e:
        error_msg = f"加载文件失败: {str(e)}"
        logger.error(error_msg)
        return {"text_content": error_msg}

@app.tool()
def build(parameter_file: str) -> Dict[str, str]:
    """构建服务器配置文件"""
    try:
        with open(parameter_file, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        server_config = {
            "path": params.get("path", os.path.join(os.path.dirname(parameter_file), "src/cleaner_optimized.py")),
            "tools": {
                "chunk_text": {
                    "input": {
                        "text_content": "$text",
                        "chunk_size": "$chunk_size",
                        "chunk_overlap": "$chunk_overlap",
                        "respect_paragraphs": "$respect_paragraphs"
                    },
                    "output": ["text_chunks"]
                },
                "clean_text_chunks": {
                    "input": {
                        "chunks": "$text_chunks",
                        "use_llm": "$use_llm",
                        "api_key": "$api_key",
                        "api_base": "$api_base"
                    },
                    "output": ["cleaned_chunks"]
                },
                "merge_chunks": {
                    "input": {
                        "chunks": "$cleaned_chunks"
                    },
                    "output": ["final_text"]
                },
                "save_text_file": {
                    "input": {
                        "text": "$final_text",
                        "file_path": "$output_path"
                    },
                    "output": ["save_result"]
                },
                "load_text_file": {
                    "input": {
                        "file_path": "$file_path"
                    },
                    "output": ["text_content"]
                }
            }
        }
        
        server_file = parameter_file.replace("parameter.yaml", "server.yaml")
        with open(server_file, 'w', encoding='utf-8') as f:
            yaml.dump(server_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"服务器配置文件已生成: {server_file}")
        return {"result": f"Server configuration built successfully: {server_file}"}
        
    except Exception as e:
        error_msg = f"构建服务器配置失败: {str(e)}"
        logger.error(error_msg)
        return {"result": error_msg}

if __name__ == "__main__":
    app.run(transport="stdio")
