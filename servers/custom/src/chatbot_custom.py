import json
from typing import List, Dict, Any, Optional

# 简化 anyio 兼容性处理，避免版本检查可能引发的问题
try:
    import anyio
    # 确保 create_memory_object_stream 可以正确导入
    try:
        # 尝试新版本导入方式
        from anyio import create_memory_object_stream
    except ImportError:
        # 尝试旧版本导入方式
        try:
            from anyio.streams.memory import create_memory_object_stream
        except ImportError:
            # 如果都失败了，设置一个占位符
            create_memory_object_stream = None
except ImportError:
    # anyio 未安装，设置占位符
    create_memory_object_stream = None

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

app = UltraRAG_MCP_Server("custom")

@app.tool(output="ret_psg,chat_history->formatted_context")
def format_context_with_history(
    ret_psg: List[List[str]], 
    chat_history: List[Dict[str, str]]
) -> Dict[str, str]:
    """格式化检索文档和对话历史，专门针对商业案例分析场景"""
  
    # 确保ret_psg不为None
    if ret_psg is None:
        ret_psg = []
  
    # 格式化检索到的商业案例文档
    documents = []
    for i, psg_list in enumerate(ret_psg):
        # 确保psg_list不为None
        if psg_list is None:
            psg_list = []
        for j, psg in enumerate(psg_list):
            # 为商业案例添加更清晰的标识
            documents.append(f"📊 案例资料 {i+1}-{j+1}:\n{psg}")
  
    documents_text = "\n\n".join(documents)
  
    # 格式化历史对话
    history_text = ""
    # 确保chat_history不为None
    if chat_history is not None:
        history_items = []
        # 保留最近6轮对话，适合学生学习场景
        for turn in chat_history[-6:]:
            # 确保turn不为None且包含必要的键
            if turn and "user" in turn and "assistant" in turn:
                history_items.append(f"🎓 学生问题: {turn['user']}")
                history_items.append(f"📚 助手回答: {turn['assistant']}")
        history_text = "\n".join(history_items)
    else:
        history_text = "这是我们的第一次对话。"
  
    # 返回包含JSON字符串的字典
    context_dict = {
        "documents": documents_text,
        "history": history_text
    }
    return {
        "formatted_context": json.dumps(context_dict, ensure_ascii=False)
    }

@app.tool(output="ans_ls->clean_answer")
def extract_answer(ans_ls: List[str]) -> Dict[str, str]:
    """提取和清理答案，针对学生学习优化"""
    if not ans_ls:
        return {"clean_answer": "抱歉，我暂时无法为您分析这个商业案例。请尝试换个角度提问，或者提供更具体的问题。"}
  
    answer = ans_ls[0]
  
    # 移除可能的格式标记
    import re
    answer = re.sub(r'\\boxed\{([^}]*)\}', r'\1', answer)
    answer = answer.strip()
    
    # 如果答案过短，提供更有帮助的回复
    if len(answer) < 20:
        answer += "\n\n💡 如果您需要更详细的分析，请告诉我您想了解这个案例的哪个具体方面，比如：\n- 商业模式分析\n- 市场策略\n- 财务表现\n- 竞争优势\n- 风险因素"
  
    return {"clean_answer": answer}

@app.tool(output="question->enhanced_question")
def enhance_student_question(question: str) -> Dict[str, str]:
    """增强学生问题，使其更适合商业案例分析"""
    
    # 检测问题类型并提供引导
    question_lower = question.lower()
    
    enhanced = question
    
    # 如果问题过于简单，提供引导
    simple_patterns = ["是什么", "怎么样", "好不好", "what", "how"]
    if any(pattern in question_lower for pattern in simple_patterns):
        if len(question) < 10:
            enhanced += "（请从商业案例分析的角度详细说明）"
    
    return {"enhanced_question": enhanced}

if __name__ == "__main__":
    app.run(transport="stdio")