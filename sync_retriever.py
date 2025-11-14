"""
同步版本的检索器，避免异步嵌套问题
直接在Streamlit应用中使用，不通过MCP协议
"""
import os
import asyncio
import jsonlines
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SyncRetriever:
    """同步检索器，专为Streamlit环境设计"""
    
    def __init__(self):
        self.client = None
        self.openai_model = None
        self.lancedb_table = None
        self.contents = []
        
    def init_openai(
        self,
        corpus_path: str = None,
        openai_model: str = None,
        api_base: str = None,
        api_key: str = None,
        **kwargs  # 忽略其他参数
    ):
        """初始化OpenAI客户端（同步版本）"""
        try:
            from openai import OpenAI  # 使用同步版本
            
            # 记录接收到的额外参数（用于调试）
            if kwargs:
                logger.debug(f"init_openai 忽略的额外参数: {list(kwargs.keys())}")
            
            if not api_key or api_key == "your_api_key_here":
                raise ValueError("无效的API密钥")
            
            # 只传递 OpenAI 客户端支持的参数
            client_params = {
                'api_key': api_key,
                'base_url': api_base,
                'timeout': 30.0,
                'max_retries': 2
            }
            
            logger.info(f"创建 OpenAI 客户端，参数: base_url={api_base}, model={openai_model}")
            logger.debug(f"client_params keys: {list(client_params.keys())}")
            
            try:
                self.client = OpenAI(**client_params)
            except TypeError as te:
                logger.error(f"OpenAI 客户端初始化参数错误: {te}")
                logger.error(f"尝试的参数: {client_params}")
                raise
            self.openai_model = openai_model
            
            # 加载语料库
            self.contents = []
            if os.path.exists(corpus_path):
                with jsonlines.open(corpus_path, mode="r") as reader:
                    self.contents = [item["contents"] for item in reader]
                logger.info(f"已加载 {len(self.contents)} 个文档片段")
            else:
                logger.warning(f"语料库文件不存在: {corpus_path}")
            
            # 测试连接（使用更短的测试文本）
            try:
                response = self.client.embeddings.create(
                    input="test",
                    model=self.openai_model
                )
                logger.info(f"OpenAI客户端初始化成功 (模型: {self.openai_model})")
            except Exception as test_error:
                logger.warning(f"连接测试失败，但客户端已创建: {test_error}")
            
            return True
            
        except ImportError:
            logger.error("未安装openai库，请运行: pip install openai")
            raise
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {e}")
            raise
    
    def search_lancedb(
        self,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = "",
        use_openai: bool = True,
        lancedb_path: str = "",
        table_name: str = "documents",
        filter_expr: Optional[str] = None,
        **kwargs  # 忽略其他参数
    ) -> Dict[str, List[List[str]]]:
        """LanceDB检索（同步版本）"""
        # 记录接收到的额外参数（用于调试）
        if kwargs:
            logger.debug(f"search_lancedb 忽略的额外参数: {list(kwargs.keys())}")
        try:
            import lancedb
        except ImportError as e:
            logger.error(f"lancedb 导入失败: {e}")
            logger.info("使用简单的文本匹配作为降级方案")
            return self._fallback_search(query_list, top_k, query_instruction)
        
        # 确保query_list不为None
        if query_list is None:
            query_list = []
        if isinstance(query_list, str):
            query_list = [query_list]
        
        queries = [f"{query_instruction}{query}" for query in query_list]
        
        if use_openai and self.client:
            # 使用同步OpenAI客户端
            query_embeddings = []
            for text in queries:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.openai_model
                )
                query_embeddings.append(response.data[0].embedding)
        else:
            raise ValueError("OpenAI客户端未初始化或use_openai=False但未实现其他嵌入方法")
        
        query_embeddings = np.array(query_embeddings, dtype=np.float32)
        logger.info("查询向量化完成")
        
        # 连接LanceDB
        if not lancedb_path:
            raise ValueError("`lancedb_path` must be provided.")
        
        db = lancedb.connect(lancedb_path)
        table = db.open_table(table_name)
        
        results = []
        for i, query_vec in enumerate(query_embeddings):
            search_result = table.search(query_vec).limit(top_k)
            
            if filter_expr:
                search_result = search_result.where(filter_expr)
            
            # 执行搜索 - LanceDB 0.3.0 返回列表而不是 DataFrame
            try:
                # 尝试新 API (0.3.0+)
                search_results_list = search_result.to_list()
                
                # 提取结果
                query_results = []
                for item in search_results_list:
                    text = item.get('text', '') or item.get('contents', '')
                    distance = item.get('_distance', 0.0)
                    query_results.append([text, str(distance)])
                    
            except AttributeError:
                # 降级到旧 API
                try:
                    search_df = search_result.to_pandas()
                    query_results = []
                    for _, row in search_df.iterrows():
                        query_results.append([
                            row.get('text', '') or row.get('contents', ''),
                            str(row.get('_distance', 0.0))
                        ])
                except Exception as e:
                    logger.error(f"搜索失败: {e}")
                    query_results = []
            
            results.append(query_results)
            logger.info(f"查询 '{queries[i]}' 找到 {len(query_results)} 个结果")
        
        return {"ret_psg": results}
    
    def _fallback_search(
        self,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = ""
    ) -> Dict[str, List[List[str]]]:
        """降级搜索方案：使用简单的文本匹配"""
        logger.warning("使用降级搜索方案（简单文本匹配）")
        
        if not self.contents:
            logger.error("没有加载文档内容")
            return {"ret_psg": [[] for _ in query_list]}
        
        results = []
        for query in query_list:
            # 简单的关键词匹配
            query_lower = query.lower()
            scored_docs = []
            
            for content in self.contents:
                content_lower = content.lower()
                # 计算简单的匹配分数（关键词出现次数）
                score = sum(1 for word in query_lower.split() if word in content_lower)
                if score > 0:
                    scored_docs.append((content, score))
            
            # 按分数排序并取top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            query_results = [[doc, str(score)] for doc, score in scored_docs[:top_k]]
            
            results.append(query_results)
            logger.info(f"降级搜索找到 {len(query_results)} 个结果")
        
        return {"ret_psg": results}
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self.client is not None