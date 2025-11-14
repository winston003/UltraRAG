#!/usr/bin/env python3
"""
批量文本清洗脚本

使用UltraRAG框架批量清洗文本文件，支持多种格式的文本文件。
适用于录音转文本、OCR文本等需要清洗的文本内容。

使用方法:
    python script/batch_clean.py --input_dir /path/to/raw_texts --output_dir /path/to/cleaned_texts
"""

import os
import glob
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Optional

def find_text_files(input_dir: Path, extensions: List[str] = None) -> List[Path]:
    """查找指定目录下的文本文件
    
    Args:
        input_dir: 输入目录路径
        extensions: 支持的文件扩展名列表
    
    Returns:
        文本文件路径列表
    """
    if extensions is None:
        extensions = ["*.txt", "*.md", "*.json", "*.jsonl"]
    
    text_files = []
    for ext in extensions:
        pattern = str(input_dir / ext)
        text_files.extend(glob.glob(pattern))
    
    return [Path(f) for f in text_files]

def batch_clean_text_files(
    input_dir: str, 
    output_dir: str, 
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    chunk_strategy: str = "sentence",
    model_name: str = "gpt-3.5-turbo",
    pipeline_config: str = "examples/text_cleaning.yaml",
    extensions: Optional[List[str]] = None,
    enable_evaluation: bool = False
) -> None:
    """批量清洗文本文件
    
    基于UltraRAG框架的大规模文本清洗方案，采用分块+LLM清洗策略。
    
    清洗策略：
    - 分块大小：1000-2000字/块，避免LLM上下文限制
    - 重叠处理：块间重叠100-200字，保证语义连续性
    - 清洗重点：语音识别错误修正、口语化表达规范化、重复内容去除、
                标点符号修正、逻辑结构整理
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        chunk_size: 文本分块大小
        chunk_overlap: 分块重叠大小
        model_name: 使用的LLM模型名称
        pipeline_config: Pipeline配置文件路径
        extensions: 支持的文件扩展名
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找文本文件
    text_files = find_text_files(input_dir, extensions)
    
    if not text_files:
        print(f"在目录 {input_dir} 中未找到任何文本文件")
        return
    
    print(f"找到 {len(text_files)} 个文本文件，开始批量清洗...")
    
    # 处理结果统计
    success_count = 0
    error_count = 0
    processing_log = []
    
    for file_path in text_files:
        file_name = file_path.stem
        output_file = output_dir / f"{file_name}_cleaned.txt"
        
        print(f"正在处理: {file_path}")
        
        # 构建清洗命令
        cmd = [
            "ultrarag", "run", pipeline_config,
            "--set", f"input_file={str(file_path)}",
            "--set", f"output_file={str(output_file)}",
            "--set", f"chunk_size={chunk_size}",
            "--set", f"chunk_overlap={chunk_overlap}",
            "--set", f"model_name={model_name}"
        ]
        
        try:
            # 执行清洗命令
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5分钟超时
            )
            
            success_count += 1
            print(f"✓ 完成: {output_file}")
            
            # 记录处理日志
            processing_log.append({
                "input_file": str(file_path),
                "output_file": str(output_file),
                "status": "success",
                "file_size": file_path.stat().st_size,
                "output_size": output_file.stat().st_size if output_file.exists() else 0
            })
            
        except subprocess.CalledProcessError as e:
            error_count += 1
            error_msg = f"处理失败: {e.stderr if e.stderr else str(e)}"
            print(f"✗ {file_path}: {error_msg}")
            
            # 记录错误日志
            processing_log.append({
                "input_file": str(file_path),
                "output_file": str(output_file),
                "status": "error",
                "error": error_msg
            })
            
        except subprocess.TimeoutExpired:
            error_count += 1
            print(f"✗ {file_path}: 处理超时")
            
            processing_log.append({
                "input_file": str(file_path),
                "output_file": str(output_file),
                "status": "timeout",
                "error": "Processing timeout"
            })
    
    # 保存处理日志
    log_file = output_dir / "batch_cleaning_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_files": len(text_files),
            "success_count": success_count,
            "error_count": error_count,
            "processing_details": processing_log
        }, f, ensure_ascii=False, indent=2)
    
    # 输出处理结果统计
    print(f"\n批量清洗完成!")
    print(f"总文件数: {len(text_files)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {error_count}")
    print(f"处理日志已保存到: {log_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量清洗文本文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python script/batch_clean.py --input_dir data/raw_texts --output_dir data/cleaned_texts
  python script/batch_clean.py --input_dir data/raw_texts --output_dir data/cleaned_texts --chunk_size 2000 --model gpt-4
        """
    )
    
    parser.add_argument(
        "--input_dir", 
        required=True, 
        help="输入目录路径，包含需要清洗的文本文件"
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="输出目录路径，清洗后的文件将保存在此目录"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1500, 
        help="文本分块大小（字符数），默认1500"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=200, 
        help="分块重叠大小（字符数），默认200"
    )
    parser.add_argument(
        "--model", 
        default="gpt-3.5-turbo", 
        help="使用的LLM模型名称，默认gpt-3.5-turbo"
    )
    parser.add_argument(
        "--pipeline", 
        default="examples/text_cleaning.yaml", 
        help="Pipeline配置文件路径，默认examples/text_cleaning.yaml"
    )
    parser.add_argument(
        "--extensions", 
        nargs="+", 
        default=["*.txt", "*.md", "*.json", "*.jsonl"], 
        help="支持的文件扩展名，默认支持txt、md、json、jsonl"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not Path(args.input_dir).exists():
        print(f"错误: 输入目录 {args.input_dir} 不存在")
        return 1
    
    # 检查Pipeline配置文件是否存在
    if not Path(args.pipeline).exists():
        print(f"错误: Pipeline配置文件 {args.pipeline} 不存在")
        return 1
    
    try:
        batch_clean_text_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=args.model,
            pipeline_config=args.pipeline,
            extensions=args.extensions
        )
        return 0
    except Exception as e:
        print(f"批量清洗过程中发生错误: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())