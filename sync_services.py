"""
同步版本的所有服务，避免MCP异步调用
包含：检索、上下文格式化、提示词生成、LLM生成、答案提取
"""
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from jinja2 import Template

logger = logging.getLogger(__name__)

class SyncServices:
    """统一的同步服务类"""
    
    def __init__(self):
        self.retriever = None
        self.openai_client = None
        
    def init_retriever(self, **config):
        """初始化检索服务"""
        from sync_retriever import SyncRetriever
        if not self.retriever:
            self.retriever = SyncRetriever()
        
        if not self.retriever.is_initialized():
            self.retriever.init_openai(**config)
        return True
    
    def search_documents(self, query_list: List[str], **config) -> Dict[str, Any]:
        """搜索文档"""
        if not self.retriever or not self.retriever.is_initialized():
            raise ValueError("检索服务未初始化")
        
        return self.retriever.search_lancedb(query_list=query_list, **config)
    
    def format_context_with_history(self, ret_psg: List[List[List[str]]], chat_history: List[Dict] = None) -> str:
        """格式化上下文和历史记录"""
        if chat_history is None:
            chat_history = []
        
        # 格式化检索结果
        context_parts = []
        if ret_psg and len(ret_psg) > 0:
            for i, query_results in enumerate(ret_psg):
                if query_results:
                    context_parts.append(f"## 相关内容 {i+1}")
                    for j, (content, score) in enumerate(query_results):
                        if content.strip():  # 只添加非空内容
                            context_parts.append(f"### 片段 {j+1} (相似度: {score})")
                            context_parts.append(content.strip())
                            context_parts.append("")  # 空行分隔
        
        # 格式化历史记录
        history_parts = []
        if chat_history:
            history_parts.append("## 对话历史")
            for i, msg in enumerate(chat_history[-5:]):  # 只保留最近5轮对话
                if msg.get('role') == 'user':
                    history_parts.append(f"**用户**: {msg.get('content', '')}")
                elif msg.get('role') == 'assistant':
                    history_parts.append(f"**助手**: {msg.get('content', '')}")
            history_parts.append("")
        
        # 组合结果
        formatted_context = "\n".join(history_parts + context_parts)
        logger.info(f"格式化上下文完成，长度: {len(formatted_context)} 字符")
        return formatted_context
    
    def generate_prompt(self, formatted_context: str, q_ls: List[str], template_path: str) -> List[str]:
        """生成提示词"""
        try:
            # 读取模板文件
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"模板文件不存在: {template_path}")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            template = Template(template_content)
            
            # 为每个查询生成提示词
            prompts = []
            for query in q_ls:
                # 构造模板期望的格式化上下文结构
                context_obj = {
                    'history': '',  # 历史记录部分（如果有的话）
                    'documents': formatted_context  # 文档内容
                }
                
                prompt = template.render(
                    formatted_context=context_obj,
                    question=query,
                    query=query
                )
                prompts.append(prompt)
            
            logger.info(f"生成了 {len(prompts)} 个提示词")
            return prompts
            
        except Exception as e:
            logger.error(f"生成提示词失败: {e}")
            raise
    
    def init_openai_client(self, api_key: str, api_base: str, model: str):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )
            self.openai_model = model
            
            # 测试连接
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=10
            )
            logger.info("OpenAI生成客户端初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"OpenAI生成客户端初始化失败: {e}")
            raise
    
    def generate_response(self, prompt_ls: List[str], **config) -> List[str]:
        """生成回答"""
        if not self.openai_client:
            raise ValueError("OpenAI客户端未初始化")
        
        responses = []
        for prompt in prompt_ls:
            try:
                response = self.openai_client.chat.completions.create(
                    model=config.get('model', self.openai_model),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.get('max_tokens', 2000),
                    temperature=config.get('temperature', 0.7)
                )
                
                content = response.choices[0].message.content
                responses.append(content)
                
            except Exception as e:
                logger.error(f"生成回答失败: {e}")
                responses.append(f"生成回答时出错: {str(e)}")
        
        logger.info(f"生成了 {len(responses)} 个回答")
        return responses
    
    def generate_response_stream(self, prompt_ls: List[str], **config):
        """流式生成回答"""
        if not self.openai_client:
            raise ValueError("OpenAI客户端未初始化")
        
        # 只处理第一个prompt进行流式生成
        if not prompt_ls:
            yield "没有提供有效的提示词"
            return
            
        prompt = prompt_ls[0]
        try:
            stream = self.openai_client.chat.completions.create(
                model=config.get('model', self.openai_model),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.get('max_tokens', 2000),
                temperature=config.get('temperature', 0.7),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"流式生成回答失败: {e}")
            yield f"生成回答时出错: {str(e)}"
    
    def extract_answer(self, ans_ls: List[str]) -> str:
        """提取和清理答案"""
        if not ans_ls:
            return "抱歉，没有生成有效的回答。"
        
        # 取第一个非空回答
        for answer in ans_ls:
            if answer and answer.strip():
                # 简单的清理逻辑
                cleaned = answer.strip()
                
                # 移除可能的系统提示
                if cleaned.startswith("作为"):
                    lines = cleaned.split('\n')
                    if len(lines) > 1:
                        cleaned = '\n'.join(lines[1:]).strip()
                
                logger.info(f"提取答案完成，长度: {len(cleaned)} 字符")
                return cleaned
        
        return "抱歉，生成的回答为空。"