#!/usr/bin/env python3

import asyncio
import logging
from ultrarag.client import run
from ultrarag.mcp_logging import get_logger
import ultrarag.client

def main():
    # 初始化全局logger
    logger = get_logger("ultrarag", "info")
    ultrarag.client.logger = logger  # 设置全局logger
    
    # 运行RAG评估
    try:
        asyncio.run(run('examples/rag_88_evaluation.yaml'))
        print("\n✅ RAG评估完成！")
    except Exception as e:
        print(f"❌ RAG评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()