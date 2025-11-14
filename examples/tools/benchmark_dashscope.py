#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DashScope模型性能测试脚本

测试不同模型和参数的性能表现
"""

import os
import sys
import asyncio
import time
import json
import argparse
from typing import Dict, List, Any

# 确保能导入UltraRAG库
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ultrarag.client import initialize, ToolCall
except ImportError:
    print("错误: 找不到UltraRAG库，请确保已安装或在正确的目录中执行")
    sys.exit(1)

from process_dashscope import DashScopeEmbedding

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DashScope模型性能测试")
    
    parser.add_argument("--input_file", type=str, default="data/word_chunk.txt",
                        help="测试文件路径")
    parser.add_argument("--models", nargs="+", 
                        default=["text-embedding-v1", "text-embedding-v2", "text-embedding-v3"],
                        help="要测试的模型列表")
    parser.add_argument("--chunk_sizes", nargs="+", type=int,
                        default=[400, 800, 1200],
                        help="要测试的分块大小列表")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json",
                        help="结果输出文件")
    parser.add_argument("--iterations", type=int, default=3,
                        help="每个测试的重复次数")
    
    return parser.parse_args()

async def test_model_performance(
    model: str,
    chunk_size: int,
    input_file: str,
    api_key: str,
    iterations: int = 3
) -> Dict[str, Any]:
    """测试单个模型的性能"""
    
    print(f"\n测试模型: {model}, 分块大小: {chunk_size}")
    
    results = {
        "model": model,
        "chunk_size": chunk_size,
        "iterations": iterations,
        "timings": [],
        "errors": []
    }
    
    for i in range(iterations):
        print(f"  第 {i+1}/{iterations} 次测试...")
        
        try:
            start_time = time.time()
            
            # 初始化服务
            servers = ["corpus"]
            initialize(servers, "servers")
            
            # 加载文本
            load_start = time.time()
            raw_data = await ToolCall.corpus.parse_documents(file_path=input_file)
            load_time = time.time() - load_start
            
            # 文本分块
            chunk_start = time.time()
            chunk_result = await ToolCall.corpus.chunk_documents(
                chunk_strategy="recursive",
                chunk_size=chunk_size,
                raw_data=raw_data['raw_data'],
                output_path=f"temp_chunks_{model}_{chunk_size}_{i}.jsonl",
                tokenizer_name_or_path="bert-base-chinese"
            )
            chunk_time = time.time() - chunk_start
            
            # 读取分块数据
            import jsonlines
            with jsonlines.open(f"temp_chunks_{model}_{chunk_size}_{i}.jsonl", mode="r") as reader:
                chunk_contents = [item["contents"] for item in reader]
            
            # 生成嵌入向量
            embed_start = time.time()
            dashscope_client = DashScopeEmbedding(api_key=api_key, model=model)
            embeddings = await dashscope_client.embed_texts(chunk_contents)
            embed_time = time.time() - embed_start
            
            # 构建索引
            index_start = time.time()
            import faiss
            import numpy as np
            embeddings = np.array(embeddings, dtype=np.float16)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype(np.float32))
            index_time = time.time() - index_start
            
            total_time = time.time() - start_time
            
            # 清理临时文件
            os.remove(f"temp_chunks_{model}_{chunk_size}_{i}.jsonl")
            
            timing = {
                "iteration": i + 1,
                "total_time": total_time,
                "load_time": load_time,
                "chunk_time": chunk_time,
                "embed_time": embed_time,
                "index_time": index_time,
                "text_length": len(raw_data['raw_data']),
                "chunk_count": len(chunk_contents),
                "embedding_dim": embeddings.shape[1],
                "success": True
            }
            
            results["timings"].append(timing)
            print(f"    总时间: {total_time:.2f}s, 嵌入时间: {embed_time:.2f}s, 向量维度: {embeddings.shape[1]}")
            
        except Exception as e:
            error = {
                "iteration": i + 1,
                "error": str(e),
                "success": False
            }
            results["errors"].append(error)
            print(f"    错误: {e}")
    
    return results

def calculate_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """计算统计信息"""
    
    if not results["timings"]:
        return {"error": "没有成功的测试结果"}
    
    timings = results["timings"]
    
    # 计算平均值
    avg_total = sum(t["total_time"] for t in timings) / len(timings)
    avg_embed = sum(t["embed_time"] for t in timings) / len(timings)
    avg_chunk = sum(t["chunk_time"] for t in timings) / len(timings)
    
    # 计算标准差
    import statistics
    std_total = statistics.stdev([t["total_time"] for t in timings]) if len(timings) > 1 else 0
    std_embed = statistics.stdev([t["embed_time"] for t in timings]) if len(timings) > 1 else 0
    
    # 计算吞吐量
    total_chunks = sum(t["chunk_count"] for t in timings)
    total_text_length = sum(t["text_length"] for t in timings)
    chunks_per_second = total_chunks / sum(t["total_time"] for t in timings)
    chars_per_second = total_text_length / sum(t["total_time"] for t in timings)
    
    return {
        "success_rate": len(timings) / (len(timings) + len(results["errors"])),
        "avg_total_time": avg_total,
        "avg_embed_time": avg_embed,
        "avg_chunk_time": avg_chunk,
        "std_total_time": std_total,
        "std_embed_time": std_embed,
        "chunks_per_second": chunks_per_second,
        "chars_per_second": chars_per_second,
        "total_chunks": total_chunks,
        "avg_chunk_count": sum(t["chunk_count"] for t in timings) / len(timings),
        "avg_embedding_dim": timings[0]["embedding_dim"] if timings else 0
    }

async def main():
    """主函数"""
    args = parse_args()
    
    # 检查API密钥
    api_key = os.getenv("ALI_EMBEDDING_API_KEY")
    if not api_key:
        print("错误: 未设置ALI_EMBEDDING_API_KEY环境变量")
        return 1
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return 1
    
    print("=" * 60)
    print("DashScope模型性能测试")
    print("=" * 60)
    print(f"测试文件: {args.input_file}")
    print(f"测试模型: {args.models}")
    print(f"分块大小: {args.chunk_sizes}")
    print(f"重复次数: {args.iterations}")
    print("=" * 60)
    
    all_results = {
        "test_config": {
            "input_file": args.input_file,
            "models": args.models,
            "chunk_sizes": args.chunk_sizes,
            "iterations": args.iterations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": []
    }
    
    # 测试所有模型和分块大小的组合
    for model in args.models:
        for chunk_size in args.chunk_sizes:
            print(f"\n{'='*40}")
            print(f"测试: {model} + {chunk_size}")
            print(f"{'='*40}")
            
            result = await test_model_performance(
                model=model,
                chunk_size=chunk_size,
                input_file=args.input_file,
                api_key=api_key,
                iterations=args.iterations
            )
            
            # 计算统计信息
            stats = calculate_statistics(result)
            result["statistics"] = stats
            
            all_results["results"].append(result)
    
    # 保存结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 输出总结
    print(f"\n{'='*60}")
    print("测试结果总结")
    print(f"{'='*60}")
    
    for result in all_results["results"]:
        model = result["model"]
        chunk_size = result["chunk_size"]
        stats = result["statistics"]
        
        if "error" not in stats:
            print(f"\n{model} (分块大小: {chunk_size}):")
            print(f"  成功率: {stats['success_rate']:.1%}")
            print(f"  平均总时间: {stats['avg_total_time']:.2f}s")
            print(f"  平均嵌入时间: {stats['avg_embed_time']:.2f}s")
            print(f"  分块速度: {stats['chunks_per_second']:.2f} 块/秒")
            print(f"  字符速度: {stats['chars_per_second']:.0f} 字符/秒")
            print(f"  向量维度: {stats['avg_embedding_dim']}")
        else:
            print(f"\n{model} (分块大小: {chunk_size}): 测试失败")
    
    print(f"\n详细结果已保存到: {args.output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
