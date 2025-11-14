import json
import logging
import os
import re
import yaml

from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from ultrarag.server import UltraRAG_MCP_Server
except ImportError:
    # 如果无法导入UltraRAG_MCP_Server，使用替代方案
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = UltraRAG_MCP_Server("custom")

@app.tool(output="text_content->text_chunks")
def chunk_text(text_content: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> Dict[str, List[str]]:
    """将文本分块处理
    
    Args:
        text_content: 输入文本内容
        chunk_size: 每块的字符数（默认1000）
        chunk_overlap: 块之间的重叠字符数（默认150）
    
    Returns:
        包含文本块列表的字典
    """
    if not text_content or not text_content.strip():
        logger.warning("输入文本为空")
        return {"text_chunks": []}
    
    text = text_content.strip()
    chunks = []
    
    # 如果文本长度小于chunk_size，直接返回整个文本
    if len(text) <= chunk_size:
        chunks.append(text)
        logger.info(f"文本长度({len(text)})小于分块大小，返回单个块")
        return {"text_chunks": chunks}
    
    # 分块处理
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # 如果不是最后一块，尝试在句号、换行符等位置分割
        if end < len(text):
            # 寻找最近的句号、换行符或空格
            for i in range(min(100, chunk_size // 4)):  # 在后1/4范围内寻找
                if end - i <= start:
                    break
                if text[end - i] in '。！？\n ':
                    end = end - i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 计算下一个块的起始位置（考虑重叠）
        start = end - chunk_overlap if chunk_overlap > 0 else end
        
        # 避免无限循环
        if start >= len(text):
            break
    
    logger.info(f"文本分块完成，共生成 {len(chunks)} 个块")
    return {"text_chunks": chunks}

@app.tool(output="chunks->cleaned_chunks")
def clean_text_chunks(chunks: List[str]) -> Dict[str, List[str]]:
    """对文本块列表进行LLM清洗
    
    Args:
        chunks: 文本块列表
    
    Returns:
        包含清洗后文本块的字典
    """
    cleaned_chunks = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"正在清洗第 {i+1}/{len(chunks)} 个文本块")
        
        # 这里应该调用LLM进行清洗，暂时返回原文本
        # 实际实现中需要集成LLM API
        cleaned_chunk = chunk  # 占位符，实际需要LLM清洗
        cleaned_chunks.append(cleaned_chunk)
    
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"文本已保存到: {file_path}")
        return {"save_result": f"文本已成功保存到 {file_path}"}
    except Exception as e:
        error_msg = f"保存文件失败: {str(e)}"
        logger.error(error_msg)
        return {"save_result": error_msg}

@app.tool(output="file_path->text_content")
def load_text_file(file_path: str) -> Dict[str, str]:
    """加载文本文件
    
    从指定路径加载文本文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"text_content": content}
    except Exception as e:
        return {"text_content": f"Error loading file: {str(e)}"}

@app.tool()
def build(parameter_file: str) -> Dict[str, str]:
    """构建服务器配置文件
    
    根据parameter.yaml生成server.yaml配置文件
    """
    try:
        # 读取parameter.yaml
        with open(parameter_file, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        # 构建server.yaml配置
        server_config = {
            "path": params.get("path", os.path.join(os.path.dirname(parameter_file), "src/cleaner.py")),
            "tools": {
                "chunk_text": {
                    "input": {
                        "text_content": "$text",
                        "chunk_size": "$chunk_size",
                        "chunk_overlap": "$chunk_overlap"
                    },
                    "output": ["text_chunks"]
                },
                "clean_text_chunks": {
                    "input": {
                        "chunks": "$text_chunks"
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
                },
                "process_file": {
                    "input": {
                        "input_file": "$input_file",
                        "output_file": "$output_file",
                        "chunk_size": "$chunk_size",
                        "chunk_overlap": "$chunk_overlap"
                    },
                    "output": ["result"]
                }
            }
        }
        
        # 保存server.yaml
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