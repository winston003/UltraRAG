#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
支持阿里云DashScope的文本处理和索引构建工具
"""

import argparse
import os
import sys
import asyncio
import aiohttp
import numpy as np
import json
from pathlib import Path
from typing import List
import jsonlines
import faiss
from dotenv import load_dotenv

# 确保能导入UltraRAG库
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ultrarag.client import initialize, ToolCall
except ImportError:
    print("错误: 找不到UltraRAG库，请确保已安装或在正确的目录中执行")
    sys.exit(1)

class DashScopeEmbedding:
    """阿里云DashScope嵌入服务"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-v3"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        
    async def embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """批量嵌入文本，支持分批处理"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            payload = {
                "model": self.model,
                "input": {
                    "texts": batch_texts
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        batch_embeddings = []
                        for item in result['output']['embeddings']:
                            batch_embeddings.append(item['embedding'])
                        all_embeddings.extend(batch_embeddings)
                    else:
                        error_text = await response.text()
                        raise Exception(f"DashScope API错误: {response.status} - {error_text}")
        
        return all_embeddings

def ensure_dir_exists(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="支持DashScope的文本分段和索引构建工具")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入文本文件路径")
    parser.add_argument("--output_dir", type=str, default="data/chunks",
                        help="输出目录，默认为data/chunks")
    parser.add_argument("--chunk_size", type=int, default=400,
                        help="分块大小，默认800")
    parser.add_argument("--chunk_strategy", type=str, default="recursive",
                        choices=["recursive", "token", "word", "sentence"],
                        help="分块策略，默认递归分块")
    parser.add_argument("--model", type=str, default="text-embedding-v3",
                        help="DashScope嵌入模型，默认text-embedding-v3")
    parser.add_argument("--tokenizer", type=str, default="bert-base-chinese",
                        help="分词器，默认bert-base-chinese")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖现有文件")
    parser.add_argument("--index_type", type=str, default="lancedb",
                        choices=["faiss", "lancedb"],
                        help="索引类型，默认lancedb")
    parser.add_argument("--lancedb_path", type=str, default="data/lancedb",
                        help="LanceDB路径，默认data/lancedb")
    parser.add_argument("--table_name", type=str, default="documents",
                        help="LanceDB表名，默认为documents")
    parser.add_argument("--append_mode", action="store_true",
                        help="追加模式，将向量追加到现有表中而不是覆盖")

    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 获取输入文件名和课程编号
    input_file_name = os.path.basename(args.input_file)
    input_file_stem = os.path.splitext(input_file_name)[0]
    lesson_number = input_file_name.split(" ")[0].replace("第", "").replace("课", "")
    
    # 使用指定的表名或默认表名
    table_name = args.table_name
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return 1
    
    # 从.env文件中获取API密钥
    # 指定.env文件的绝对路径
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("ALI_EMBEDDING_API_KEY")
    if not api_key:
        print(f"错误: 未设置ALI_EMBEDDING_API_KEY环境变量，请在{env_path}文件中配置")
        return 1
        
    # 准备输出目录和文件名
    ensure_dir_exists(args.output_dir)
    
    chunks_path = os.path.join(args.output_dir, f"{input_file_stem}_chunks.jsonl")
    
    print("=" * 60)
    print("UltraRAG文本分段和索引构建工具 (DashScope版本)")
    print("=" * 60)
    print(f"输入文件: {args.input_file}")
    print(f"分块大小: {args.chunk_size}")
    print(f"分块策略: {args.chunk_strategy}")
    print(f"嵌入模型: {args.model}")
    print(f"分块输出: {chunks_path}")
    print(f"索引类型: {args.index_type}")
    if args.index_type == "faiss":
        index_path = f"index/index_{input_file_stem}.index"
        print(f"索引文件: {index_path}")
    else:
        print(f"LanceDB路径: {args.lancedb_path}")
        print(f"表名: {table_name}")
    print("=" * 60)
    
    # 步骤1: 加载原始文本
    print("\n步骤1: 加载文本文件...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        print(f"文本加载成功，共 {len(raw_text)} 字符")
        print(f"文本内容预览: {raw_text[:100]}...")
    except Exception as e:
        print(f"加载文件失败: {e}")
        return 1
    
    # 步骤2: 文本分块（使用本地chonkie库）
    print("\n步骤2: 文本分块...")
    try:
        from chonkie import RecursiveChunker
        
        # 创建递归分块器
        chunker = RecursiveChunker(
            chunk_size=args.chunk_size
        )
        
        # 执行分块
        chunks = chunker.chunk(raw_text)
        
        # 保存分块结果
        chunked_documents = []
        with jsonlines.open(chunks_path, mode="w") as writer:
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "id": str(i),
                    "contents": chunk.text,
                    "metadata": {
                        "chunk_index": i,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "source_file": input_file_name
                    }
                }
                writer.write(chunk_data)
                chunked_documents.append(chunk_data)
                
        print(f"文本分块成功，生成 {len(chunked_documents)} 个分块")
    except ImportError:
        print("错误: chonkie库未安装，请运行: pip install chonkie")
        return 1
    except Exception as e:
        print(f"文本分块失败: {e}")
        return 1
    
    # 步骤3: 使用DashScope生成嵌入向量
    print("\n步骤3: 生成文本嵌入 (DashScope)...")
    try:
        # 读取分块数据
        with jsonlines.open(chunks_path, mode="r") as reader:
            chunk_contents = [item["contents"] for item in reader]
        
        # 使用DashScope API
        dashscope_client = DashScopeEmbedding(
            api_key=api_key,
            model=args.model
        )
        
        embeddings = await dashscope_client.embed_texts(chunk_contents)
        embeddings = np.array(embeddings, dtype=np.float16)
        print(f"文本嵌入生成成功，向量维度: {embeddings.shape}")
        
    except Exception as e:
        print(f"生成文本嵌入失败: {e}")
        return 1
    
    # 步骤4: 构建索引
    print(f"\n步骤4: 构建{args.index_type}索引...")
    try:
        if args.index_type == "faiss":
            # 创建Faiss索引
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # 内积索引
            
            # 添加向量到索引
            index.add(embeddings.astype(np.float32))
            
            # 保存索引
            ensure_dir_exists("index")
            faiss.write_index(index, index_path)
            print(f"Faiss向量索引构建成功，包含 {index.ntotal} 个向量")
        else:
            try:
                import lancedb
                import pandas as pd
                
                # 准备数据
                ids = [str(i) for i in range(len(embeddings))]
                data = []
                for i, (text, embedding) in enumerate(zip(chunk_contents, embeddings)):
                    data.append({
                        'id': ids[i],
                        'text': text,
                        'vector': embedding.tolist(),
                        'source_file': input_file_name,  # 添加源文件信息
                        'lesson_number': lesson_number   # 添加课程编号
                    })
                
                # 创建LanceDB表
                ensure_dir_exists(args.lancedb_path)
                db = lancedb.connect(args.lancedb_path)
                
                if args.append_mode and table_name in db.table_names():
                    # 追加模式：分批处理避免内存溢出
                    table = db.open_table(table_name)
                    start_id = len(table)
                    
                    # 分批处理，每批50个向量
                    batch_size = 50
                    total_vectors = len(embeddings)
                    
                    for i in range(0, total_vectors, batch_size):
                        end_idx = min(i + batch_size, total_vectors)
                        batch_embeddings = embeddings[i:end_idx]
                        batch_contents = chunk_contents[i:end_idx]
                        
                        batch_data = []
                        for j, (text, embedding) in enumerate(zip(batch_contents, batch_embeddings)):
                            batch_data.append({
                                'id': str(start_id + i + j),
                                'text': text,
                                'vector': embedding.tolist(),
                                'source_file': input_file_name,
                                'lesson_number': lesson_number
                            })
                        
                        # 分批写入
                        table.add(batch_data)
                        print(f"已处理 {end_idx}/{total_vectors} 个向量")
                        
                        # 释放批次内存
                        del batch_data, batch_embeddings, batch_contents
                    
                    print(f"LanceDB索引追加成功，新增 {total_vectors} 个向量到表: {table_name}")
                else:
                    # 覆盖模式：使用分批处理避免内存溢出
                    batch_size = 50
                    total_vectors = len(embeddings)
                    
                    # 使用第一批数据创建表
                    first_batch_size = min(batch_size, total_vectors)
                    first_batch_embeddings = embeddings[:first_batch_size]
                    first_batch_contents = chunk_contents[:first_batch_size]
                    
                    # 准备第一批数据
                    first_batch_data = []
                    for j, (text, embedding) in enumerate(zip(first_batch_contents, first_batch_embeddings)):
                        first_batch_data.append({
                            'id': str(j),
                            'text': text,
                            'vector': embedding.tolist(),
                            'source_file': input_file_name,
                            'lesson_number': lesson_number
                        })
                    
                    # 创建表
                    table = db.create_table(table_name, data=first_batch_data, mode="overwrite")
                    print(f"已处理 {first_batch_size}/{total_vectors} 个向量")
                    
                    # 释放第一批内存
                    del first_batch_data, first_batch_embeddings, first_batch_contents
                    
                    # 处理剩余批次
                    for i in range(first_batch_size, total_vectors, batch_size):
                        end_idx = min(i + batch_size, total_vectors)
                        batch_embeddings = embeddings[i:end_idx]
                        batch_contents = chunk_contents[i:end_idx]
                        
                        batch_data = []
                        for j, (text, embedding) in enumerate(zip(batch_contents, batch_embeddings)):
                            batch_data.append({
                                'id': str(i + j),
                                'text': text,
                                'vector': embedding.tolist(),
                                'source_file': input_file_name,
                                'lesson_number': lesson_number
                            })
                        
                        # 分批写入
                        table.add(batch_data)
                        print(f"已处理 {end_idx}/{total_vectors} 个向量")
                        
                        # 释放批次内存
                        del batch_data, batch_embeddings, batch_contents
                    
                    print(f"LanceDB索引构建成功，包含 {total_vectors} 个向量，表名: {table_name}")
                
            except ImportError as e:
                print(f"错误: LanceDB未安装，请运行: pip install lancedb")
                return 1
            except Exception as e:
                print(f"构建LanceDB索引失败: {e}")
                return 1

    except Exception as e:
        print(f"构建向量索引失败: {e}")
        return 1
    
    # 步骤5: 测试检索功能
    print("\n步骤5: 测试检索功能...")
    try:
        # 创建测试查询
        test_query = "什么是人工智能？"
        
        # 生成查询向量
        query_embeddings = await dashscope_client.embed_texts([test_query])
        query_vector = np.array(query_embeddings[0], dtype=np.float32).reshape(1, -1)
        
        if args.index_type == "faiss":
            # 执行Faiss检索
            scores, indices = index.search(query_vector, k=3)
            
            print("Faiss检索测试成功")
            print(f"检索到 {len(indices[0])} 个相关文档")
            print("最相关的文档:")
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(chunk_contents):
                    print(f"  {i+1}. 相似度: {score:.4f}")
                    print(f"     内容: {chunk_contents[idx][:100]}...")
        else:
            # LanceDB测试需要在实际检索服务中进行
            
            print("LanceDB索引构建完成，测试将在检索服务中进行")
        
    except Exception as e:
        print(f"检索测试失败: {e}")
        return 1
    
    print("\n处理完成！")
    print("=" * 60)
    print(f"分块文件: {chunks_path}")
    if args.index_type == "faiss":
        print(f"索引文件: {index_path}")
    else:
        print(f"LanceDB路径: {args.lancedb_path}")
        print(f"表名: {table_name}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))