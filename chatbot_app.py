import asyncio
import json
import logging
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到 Python 路径
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import psutil
except ImportError:
    print("错误: 无法导入 psutil。请运行: pip install psutil")
    raise

try:
    import yaml
except ImportError:
    print("错误: 无法导入 yaml。请运行: pip install pyyaml")
    raise

try:
    import streamlit as st
except ImportError:
    print("错误: 无法导入 streamlit。请运行: pip install streamlit")
    raise

# 修改导入语句 - 确保src在路径中
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 注意：我们使用 sync_services 而不是 ultrarag.client
# 因此不需要导入 ToolCall 和 initialize
# try:
#     from ultrarag.client import ToolCall, initialize
#     logger_temp = logging.getLogger(__name__)
#     logger_temp.info("✅ 成功导入 ultrarag.client")
# except ImportError as e:
#     print(f"错误: 无法导入 ultrarag.client: {e}")
#     print(f"Python路径: {sys.path}")
#     print(f"当前目录: {os.getcwd()}")
#     raise

try:
    from sync_services import SyncServices
except ImportError:
    print("错误: 无法导入 sync_services。请确保相关模块已正确安装。")
    raise

# 导入配置管理器
try:
    from config_manager import config_manager
except ImportError:
    print("错误: 无法导入配置管理器。")
    raise

# 导入智能日志管理器
try:
    from log_manager import log_manager, get_smart_logger
    logger = get_smart_logger("chatbot_app")
    logger.info("智能日志管理器已启用")
except ImportError:
    # 降级到基本日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.warning("智能日志管理器不可用，使用基本日志配置")

# 全局同步服务实例
sync_services = SyncServices()

# 性能监控装饰器
def monitor_performance(func_name: str):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"⏱️ {func_name} 执行耗时: {duration:.3f}秒")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ {func_name} 执行失败 (耗时: {duration:.3f}秒): {e}")
                raise
        return wrapper
    return decorator

def _sync_result(val, timeout: float | None = None):
    """如果 val 是 asyncio.Task/Future 或 coroutine, 在同步上下文中等待其完成并返回结果。
    采用线程安全的方式处理异步调用，避免事件循环嵌套问题。
    """
    if val is None:
        return None
    try:
        # 已经是普通值
        if not (asyncio.iscoroutine(val) or isinstance(val, asyncio.Future) or isinstance(val, asyncio.Task)):
            return val
    except Exception:
        return val

    # 使用线程安全的方式处理异步调用
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在运行的事件循环中，使用线程池执行
            import concurrent.futures
            
            async def _run_async():
                if asyncio.iscoroutine(val):
                    return await val
                elif isinstance(val, (asyncio.Task, asyncio.Future)):
                    return await val
                else:
                    return val
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run_async())
                if timeout:
                    return future.result(timeout=timeout)
                else:
                    return future.result()
        else:
            # 事件循环没有运行，直接运行协程
            if asyncio.iscoroutine(val):
                return asyncio.run(val)
            elif isinstance(val, (asyncio.Task, asyncio.Future)):
                return asyncio.run(asyncio.create_task(val))
            else:
                return val
    except RuntimeError:
        # 没有可用事件循环，直接运行
        if asyncio.iscoroutine(val):
            return asyncio.run(val)
        elif isinstance(val, (asyncio.Task, asyncio.Future)):
            return asyncio.run(asyncio.create_task(val))
        else:
            return val

def _exec_step(name: str, call_func, snapshot_vars: dict | None = None, timeout: float | None = None, raise_on_error: bool = False):
    """执行单步调用并捕获异常/堆栈与变量快照。
    返回步骤的实际结果（如果失败返回 None 并记录详细日志）。
    如果 raise_on_error=True，则在失败时抛出异常而不是返回 None。
    """
    step_start = time.time()
    logger.info(f"开始执行步骤: {name}")
    
    # 记录系统状态
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.debug(f"系统状态 - 内存使用: {mem_info.rss / 1024 / 1024:.2f}MB, CPU使用率: {process.cpu_percent()}%")
    except Exception as e:
        logger.warning(f"获取系统状态失败: {e}")

    # 记录上下文变量
    if snapshot_vars:
        logger.debug(f"步骤 {name} 上下文变量: {snapshot_vars}")
    
    try:
        logger.debug(f"开始调用函数: {call_func.__name__ if hasattr(call_func, '__name__') else str(call_func)}")
        raw = call_func()
        res = _sync_result(raw, timeout=timeout)
        
        # 记录执行时间
        duration = time.time() - step_start
        logger.info(f"步骤 {name} 成功完成，耗时: {duration:.3f}秒")
        
        # 记录返回结果摘要
        if res is not None:
            try:
                if isinstance(res, (dict, list)):
                    result_summary = {
                        "type": type(res).__name__,
                        "size": len(res),
                        "sample": str(res)[:200] + "..." if len(str(res)) > 200 else str(res)
                    }
                else:
                    result_summary = {
                        "type": type(res).__name__,
                        "value": str(res)[:200] + "..." if len(str(res)) > 200 else str(res)
                    }
                logger.debug(f"步骤 {name} 返回结果摘要: {result_summary}")
            except Exception as e:
                logger.warning(f"记录结果摘要时出错: {e}")
        
        return res
        
    except TimeoutError as e:
        logger.error(f"步骤 {name} 执行超时: {str(e)}")
        error_type = "timeout"
        error_msg = f"执行超时 (>{timeout}秒)" if timeout else "执行超时"
        tb = traceback.format_exc()
    except ConnectionError as e:
        logger.error(f"步骤 {name} 网络连接错误: {str(e)}")
        error_type = "connection"
        error_msg = f"网络连接错误: {str(e)}"
        tb = traceback.format_exc()
    except Exception as e:
        logger.error(f"步骤 {name} 执行错误: {str(e)}")
        error_type = "general"
        error_msg = str(e)
        tb = traceback.format_exc()

    # 错误处理和日志记录
    info = {
        "step": name,
        "error_type": error_type,
        "error": error_msg,
        "traceback": tb,
        "snapshot": {},
        "system_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": time.time() - step_start
        }
    }

    # 添加系统状态信息
    try:
        process = psutil.Process()
        info["system_info"].update({
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "threads": len(process.threads())
        })
    except Exception as e:
        logger.warning(f"获取系统信息失败: {e}")

    if snapshot_vars:
        for k, v in snapshot_vars.items():
            try:
                info["snapshot"][k] = repr(v)
            except Exception:
                info["snapshot"][k] = "<unserializable>"

    # 保存详细错误日志
    try:
        os.makedirs("logs/error_details", exist_ok=True)
        log_path = os.path.join("logs/error_details", f"{name}_{int(time.time())}.json")
        with open(log_path, "w", encoding="utf-8") as lf:
            json.dump(info, lf, ensure_ascii=False, indent=2)
        logger.info(f"已保存详细错误日志到: {log_path}")
    except Exception as e:
        logger.error(f"保存错误日志失败: {e}")
        log_path = None

    # 显示错误消息
    try:
        last_line = tb.splitlines()[-1] if tb else error_msg
        error_display = f"步骤 '{name}' {error_type}错误: {error_msg}"
        if log_path:
            error_display += f"。详细日志: {log_path}"
        else:
            error_display += f"。错误信息: {last_line}"
        st.error(error_display)
        logger.error(error_display)
    except Exception as e:
        logger.error(f"显示错误消息失败: {e}")

    # 如果要求抛出异常，则抛出而不是返回 None
    if raise_on_error:
        if error_type == "timeout":
            raise TimeoutError(error_msg)
        elif error_type == "connection":
            raise ConnectionError(error_msg)
        else:
            raise RuntimeError(error_msg)
    
    return None

class BusinessCaseRAGChatbot:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._initialized = False
        
        # 检查并创建环境变量文件
        config_manager.create_env_file_if_missing()
        
        # 验证配置完整性
        is_valid, errors = config_manager.validate_config()
        if not is_valid:
            error_msg = "配置验证失败:\n" + "\n".join(f"  - {err}" for err in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 读取配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

        # 注意：我们使用 sync_services 而不是 MCP 服务器
        # 因此不需要调用 initialize()
        logger.info("使用同步服务模式，跳过 MCP 服务器初始化")
        self._initialized = True

    def chat_stream(self, question: str, chat_history: List[Dict]):
        """流式聊天方法"""
        try:
            # 验证配置文件是否存在
            if not os.path.exists(self.config_path):
                yield "系统配置文件缺失，请检查配置文件是否存在。"
                return
            
            # 读取基础参数配置
            base_param_path = "config/parameter/chatbot_parameter.yaml"
            if not os.path.exists(base_param_path):
                yield "系统参数配置文件缺失，请检查配置文件是否存在。"
                return
                
            with open(base_param_path, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)
                if not isinstance(params, dict):
                    params = {}

            # 如果初始化时未加载 params，则使用这里的
            if not hasattr(self, 'params') or getattr(self, 'params', None) is None:
                self.params = params
            if not isinstance(getattr(self, 'params', None), dict):
                self.params = {}

            # 设置用户问题和聊天历史
            if 'global_vars' not in params:
                params['global_vars'] = {}
            params['global_vars']['query'] = question
            params['global_vars']['chat_history'] = chat_history if chat_history is not None else []

            q_ls = [question]

            # 1) 初始化检索服务
            retriever_cfg = self.params.get('retriever', {})
            
            try:
                # 安全获取API密钥
                api_key = config_manager.get_config_with_fallback(
                    retriever_cfg, 'api_key', 'dashscope'
                )
                
                if not api_key or not config_manager.validate_api_key(api_key):
                    yield "系统初始化失败: API密钥无效或未配置。请检查 .env 文件中的 ALI_EMBEDDING_API_KEY 配置。"
                    return
                
                sync_services.init_retriever(
                    corpus_path=retriever_cfg.get('corpus_path', 'data/processed/88_chunks.jsonl'),
                    openai_model=retriever_cfg.get('openai_model', 'text-embedding-v3'),
                    api_base=retriever_cfg.get('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
                    api_key=api_key
                )
            except Exception as e:
                yield f"系统初始化失败: {str(e)}"
                return
            
            # 2) 执行检索
            try:
                ret = sync_services.search_documents(
                    query_list=q_ls,
                    top_k=retriever_cfg.get('top_k', 5),
                    query_instruction=retriever_cfg.get('query_instruction', 'Query: '),
                    use_openai=retriever_cfg.get('use_openai', True),
                    lancedb_path=retriever_cfg.get('lancedb_path', 'data/lancedb'),
                    table_name=retriever_cfg.get('table_name', 'documents'),
                    filter_expr=retriever_cfg.get('filter_expr', ''),
                )
            except Exception as e:
                yield f"检索服务出错: {str(e)}"
                return

            if ret is None:
                yield "系统调用检索服务时出错。"
                return

            # 取回检索结果
            ret_psg = None
            if isinstance(ret, dict):
                ret_psg = ret.get('ret_psg') or ret.get('passages') or ret.get('results')
            else:
                ret_psg = ret

            if ret_psg is None:
                ret_psg = []

            # 3) 格式化上下文
            try:
                safe_chat_history = chat_history if chat_history is not None else []
                formatted_context = sync_services.format_context_with_history(
                    ret_psg=ret_psg, 
                    chat_history=safe_chat_history
                )
            except Exception as e:
                yield f"系统格式化上下文时出错: {str(e)}"
                return

            # 4) 生成 prompt
            try:
                prompt_cfg = self.params.get('prompt', {}) if isinstance(self.params, dict) else {}
                template_path = prompt_cfg.get('template') if isinstance(prompt_cfg, dict) else None
                if not template_path or not isinstance(template_path, str):
                    template_path = "prompt/qa_rag_multiround.jinja"
                else:
                    if not os.path.exists(template_path):
                        alt_path = template_path.replace("prompts/", "prompt/")
                        if os.path.exists(alt_path):
                            template_path = alt_path
                        else:
                            template_path = "prompt/qa_rag_multiround.jinja"

                prompt_ls = sync_services.generate_prompt(
                    formatted_context=formatted_context, 
                    q_ls=q_ls, 
                    template_path=template_path
                )
            except Exception as e:
                yield f"系统生成提示词时出错: {str(e)}"
                return

            # 5) 初始化生成客户端
            gen_cfg = self.params.get('generation', {})
            model_name = gen_cfg.get('model_name', self.params.get('model_name') if isinstance(self.params, dict) else None)
            base_url = gen_cfg.get('base_url', self.params.get('base_url') if isinstance(self.params, dict) else None)
            sampling_params = gen_cfg.get('sampling_params', self.params.get('sampling_params') if isinstance(self.params, dict) else None)
            
            if model_name is None:
                model_name = ""
            if base_url is None:
                base_url = ""

            try:
                if not sync_services.openai_client:
                    # 安全获取API密钥
                    api_key = config_manager.get_config_with_fallback(
                        gen_cfg, 'api_key', 'dashscope'
                    )
                    
                    if not api_key or not config_manager.validate_api_key(api_key):
                        yield "系统初始化失败: 生成服务API密钥无效或未配置。请检查 .env 文件配置。"
                        return
                    
                    sync_services.init_openai_client(
                        api_key=api_key,
                        api_base=base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                        model=model_name or 'qwen-plus'
                    )
                
                # 6) 流式生成回答
                for chunk in sync_services.generate_response_stream(
                    prompt_ls=prompt_ls,
                    model=model_name or 'qwen-plus',
                    max_tokens=sampling_params.get('max_tokens', 2000) if sampling_params else 2000,
                    temperature=sampling_params.get('temperature', 0.7) if sampling_params else 0.7
                ):
                    yield chunk
                    
            except Exception as e:
                yield f"系统生成回答时出错: {str(e)}"
                return

        except Exception as e:
            yield f"系统遇到问题: {str(e)}"
            return
    
    def chat(self, question: str, chat_history: List[Dict]) -> str:
        """同步聊天方法"""
        try:

            # 验证配置文件是否存在
            if not os.path.exists(self.config_path):
                st.error(f"❌ 配置文件不存在: {self.config_path}")
                return "系统配置文件缺失，请检查配置文件是否存在。"
            
            # 读取基础参数配置
            base_param_path = "config/parameter/chatbot_parameter.yaml"
            if not os.path.exists(base_param_path):
                st.error(f"❌ 参数配置文件不存在: {base_param_path}")
                return "系统参数配置文件缺失，请检查配置文件是否存在。"
                
            t0 = time.time()
            with open(base_param_path, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)
                # 防御性处理：当 YAML 为空或解析为非字典时，使用默认空字典
                if not isinstance(params, dict):
                    params = {}
            t1 = time.time()
            st.info(f"⏱️ 读取参数耗时: {t1 - t0:.3f}s")

            # 如果初始化时未加载 params，则使用这里的
            if not hasattr(self, 'params') or getattr(self, 'params', None) is None:
                self.params = params
            # 确保 self.params 一定为字典
            if not isinstance(getattr(self, 'params', None), dict):
                self.params = {}

            # 设置用户问题和聊天历史
            if 'global_vars' not in params:
                params['global_vars'] = {}
            params['global_vars']['query'] = question
            # 确保chat_history不为None
            params['global_vars']['chat_history'] = chat_history if chat_history is not None else []

            # 创建临时参数文件（保留以兼容现有 pipeline 逻辑）
            t2 = time.time()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as temp_file:
                yaml.dump(params, temp_file, default_flow_style=False, allow_unicode=True)
                temp_param_path = temp_file.name
            t3 = time.time()
            st.info(f"⏱️ 写入临时参数文件耗时: {t3 - t2:.3f}s, 路径={temp_param_path}")

            try:
                # 使用长连接的 ToolCall 顺序执行业务流水线，避免每次重启 MCP client
                st.info(f"🔧 调试信息: 配置文件={self.config_path}, 参数文件={temp_param_path}")
                st.info(f"🔧 调试信息: 用户问题={question}")

                q_ls = [question]

                # 1) 初始化检索服务（确保OpenAI客户端已初始化）
                retriever_cfg = self.params.get('retriever', {})
                
                # 记录检索请求详情
                logger.info(f"开始检索请求，查询列表: {q_ls}")
                logger.debug(f"检索配置: {json.dumps(retriever_cfg, ensure_ascii=False, indent=2)}")
                
                # 记录检索资源状态
                try:
                    db_path = retriever_cfg.get('lancedb_path', 'data/lancedb')
                    if os.path.exists(db_path):
                        db_size = sum(f.stat().st_size for f in Path(db_path).rglob('*') if f.is_file())
                        logger.debug(f"检索数据库状态 - 路径: {db_path}, 大小: {db_size/1024/1024:.2f}MB")
                except Exception as e:
                    logger.warning(f"获取数据库状态失败: {e}")
                
                # 首先初始化同步服务（如果尚未初始化）
                try:
                    # 安全获取API密钥
                    api_key = config_manager.get_config_with_fallback(
                        retriever_cfg, 'api_key', 'dashscope'
                    )
                    
                    if not api_key or not config_manager.validate_api_key(api_key):
                        error_msg = "API密钥无效或未配置。请检查 .env 文件中的 ALI_EMBEDDING_API_KEY 配置。"
                        logger.error(f"❌ {error_msg}")
                        return f"系统初始化失败: {error_msg}"
                    
                    sync_services.init_retriever(
                        corpus_path=retriever_cfg.get('corpus_path', 'data/processed/88_chunks.jsonl'),
                        openai_model=retriever_cfg.get('openai_model', 'text-embedding-v3'),
                        api_base=retriever_cfg.get('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
                        api_key=api_key
                    )
                    logger.info("✅ 检索服务初始化成功")
                except Exception as e:
                    logger.error(f"❌ 检索服务初始化失败: {e}")
                    return f"系统初始化失败: {str(e)}"
                
                # 2) 执行检索（使用同步版本）
                t_retr0 = time.time()
                try:
                    ret = sync_services.search_documents(
                        query_list=q_ls,
                        top_k=retriever_cfg.get('top_k', 5),
                        query_instruction=retriever_cfg.get('query_instruction', 'Query: '),
                        use_openai=retriever_cfg.get('use_openai', True),
                        lancedb_path=retriever_cfg.get('lancedb_path', 'data/lancedb'),
                        table_name=retriever_cfg.get('table_name', 'documents'),
                        filter_expr=retriever_cfg.get('filter_expr', ''),
                    )
                    logger.info(f"✅ 检索完成，找到 {len(ret.get('ret_psg', []))} 个查询的结果")
                except Exception as e:
                    logger.error(f"❌ 检索失败: {e}")
                    return "检索服务出错，请稍后重试。"
                t_retr1 = time.time()
                retrieval_time = t_retr1 - t_retr0
                
                # 记录检索结果统计
                if ret is not None:
                    try:
                        if isinstance(ret, dict):
                            ret_stats = {
                                "total_results": len(ret.get('ret_psg', []) or ret.get('passages', []) or ret.get('results', [])),
                                "result_format": "dict",
                                "available_keys": list(ret.keys())
                            }
                        elif isinstance(ret, list):
                            ret_stats = {
                                "total_results": len(ret),
                                "result_format": "list"
                            }
                        else:
                            ret_stats = {
                                "result_format": type(ret).__name__
                            }
                        logger.info(f"检索完成 - 耗时: {retrieval_time:.3f}s, 统计信息: {ret_stats}")
                    except Exception as e:
                        logger.warning(f"统计检索结果失败: {e}")
                
                st.info(f"⏱️ 检索耗时: {retrieval_time:.3f}s")

                if ret is None:
                    return "系统调用检索服务时出错，已记录详细日志。"

                # 取回检索结果（兼容多种返回格式）
                ret_psg = None
                if isinstance(ret, dict):
                    ret_psg = ret.get('ret_psg') or ret.get('passages') or ret.get('results')
                else:
                    ret_psg = ret

                # 确保ret_psg不为None，提供默认空列表
                if ret_psg is None:
                    ret_psg = []

                # 2) 格式化上下文（使用同步版本）
                t_fmt0 = time.time()
                # 确保chat_history不为None，避免NoneType迭代错误
                safe_chat_history = chat_history if chat_history is not None else []
                try:
                    formatted_context = sync_services.format_context_with_history(
                        ret_psg=ret_psg, 
                        chat_history=safe_chat_history
                    )
                    logger.info("✅ 上下文格式化成功")
                except Exception as e:
                    logger.error(f"❌ 上下文格式化失败: {e}")
                    return "系统格式化上下文时出错，请稍后重试。"
                t_fmt1 = time.time()
                st.info(f"⏱️ 格式化上下文耗时: {t_fmt1 - t_fmt0:.3f}s")

                # 3) 生成 prompt
                t_p0 = time.time()
                # 读取模板路径（从 prompt.template），并进行兜底与路径修正
                prompt_cfg = self.params.get('prompt', {}) if isinstance(self.params, dict) else {}
                template_path = prompt_cfg.get('template') if isinstance(prompt_cfg, dict) else None
                if not template_path or not isinstance(template_path, str):
                    template_path = "prompt/qa_rag_multiround.jinja"
                else:
                    # 若路径不存在，尝试修正 prompts -> prompt，并提供兜底
                    if not os.path.exists(template_path):
                        alt_path = template_path.replace("prompts/", "prompt/")
                        if os.path.exists(alt_path):
                            st.warning(f"提示: 未找到模板 {template_path}，自动使用 {alt_path}")
                            template_path = alt_path
                        else:
                            st.warning(f"提示: 未找到模板 {template_path}，自动回退为默认模板 prompt/qa_rag_multiround.jinja")
                            template_path = "prompt/qa_rag_multiround.jinja"

                try:
                    prompt_ls = sync_services.generate_prompt(
                        formatted_context=formatted_context, 
                        q_ls=q_ls, 
                        template_path=template_path
                    )
                    logger.info("✅ 提示词生成成功")
                except Exception as e:
                    logger.error(f"❌ 提示词生成失败: {e}")
                    return "系统生成提示词时出错，请稍后重试。"
                t_p1 = time.time()
                st.info(f"⏱️ prompt 生成耗时: {t_p1 - t_p0:.3f}s")

                # 4) 调用生成模型（使用同步版本）
                t_g0 = time.time()
                gen_cfg = self.params.get('generation', {})
                # 安全读取生成配置，避免 None 传播
                model_name = gen_cfg.get('model_name', self.params.get('model_name') if isinstance(self.params, dict) else None)
                base_url = gen_cfg.get('base_url', self.params.get('base_url') if isinstance(self.params, dict) else None)
                sampling_params = gen_cfg.get('sampling_params', self.params.get('sampling_params') if isinstance(self.params, dict) else None)
                # 将 None 转为空字符串以避免在下游执行 `in` 判断时报 NoneType 错误
                if model_name is None:
                    model_name = ""
                if base_url is None:
                    base_url = ""

                try:
                    # 初始化生成客户端（如果需要）
                    if not sync_services.openai_client:
                        # 安全获取API密钥
                        api_key = config_manager.get_config_with_fallback(
                            gen_cfg, 'api_key', 'dashscope'
                        )
                        
                        if not api_key or not config_manager.validate_api_key(api_key):
                            error_msg = "生成服务API密钥无效或未配置。请检查 .env 文件配置。"
                            logger.error(f"❌ {error_msg}")
                            return f"系统初始化失败: {error_msg}"
                        
                        sync_services.init_openai_client(
                            api_key=api_key,
                            api_base=base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                            model=model_name or 'qwen-plus'
                        )
                    
                    ans_ls = sync_services.generate_response(
                        prompt_ls=prompt_ls,
                        model=model_name or 'qwen-plus',
                        max_tokens=sampling_params.get('max_tokens', 2000) if sampling_params else 2000,
                        temperature=sampling_params.get('temperature', 0.7) if sampling_params else 0.7
                    )
                    logger.info("✅ 回答生成成功")
                except Exception as e:
                    logger.error(f"❌ 回答生成失败: {e}")
                    return "系统生成回答时出错，请稍后重试。"
                t_g1 = time.time()
                st.info(f"⏱️ 生成调用耗时: {t_g1 - t_g0:.3f}s")

                # 5) 提取最终回答（使用同步版本）
                t_e0 = time.time()
                try:
                    final_answer = sync_services.extract_answer(ans_ls=ans_ls)
                    logger.info("✅ 答案提取成功")
                except Exception as e:
                    logger.error(f"❌ 答案提取失败: {e}")
                    return "系统提取答案时出错，请稍后重试。"
                t_e1 = time.time()
                st.info(f"⏱️ 答案提取耗时: {t_e1 - t_e0:.3f}s")

                # 返回最终答案
                return final_answer

            finally:
                # 清理临时文件
                if os.path.exists(temp_param_path):
                    os.unlink(temp_param_path)
                    st.info(f"🧹 已清理临时参数文件: {temp_param_path}")

        except FileNotFoundError as e:
            error_msg = f"文件未找到错误: {str(e)}"
            st.error(f"❌ {error_msg}")
            return f"系统配置文件缺失: {error_msg}。请检查相关文件是否存在。"
        except yaml.YAMLError as e:
            error_msg = f"YAML配置文件格式错误: {str(e)}"
            st.error(f"❌ {error_msg}")
            return f"配置文件格式错误: {error_msg}。请检查配置文件格式。"
        except ImportError as e:
            error_msg = f"模块导入错误: {str(e)}"
            st.error(f"❌ {error_msg}")
            return f"系统模块缺失: {error_msg}。请检查依赖是否正确安装。"
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            st.error(f"❌ 分析过程中遇到问题: {error_msg}")
            # 打印详细的错误堆栈信息用于调试
            import traceback
            st.error(f"🔧 详细错误信息: {traceback.format_exc()}")
            return f"系统遇到问题: {error_msg}。请稍后重试或联系技术支持。"

def init_streamlit():
    """初始化Streamlit页面"""
    st.set_page_config(
        page_title="高老师分身智能问答系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
  
    # 自定义CSS样式
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .student-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
  
    st.markdown('<h1 class="main-header">📊 商业案例拆解</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🎓 陆续更新中... ...</p>', unsafe_allow_html=True)
    st.markdown("---")

def display_chat_history():
    """显示聊天历史"""
    # 确保st.session_state.messages存在且不为None
    if not hasattr(st.session_state, 'messages') or st.session_state.messages is None:
        st.session_state.messages = []
        
    if len(st.session_state.messages) == 0:
        # 添加欢迎消息
        welcome_msg = """
        👋 **欢迎使用商业案例分析助手！**
        
        我是您的专属商业学习导师，可以帮助您：
        
        📈 **案例分析**: 深入拆解商业案例的关键要素
        💡 **概念解释**: 用通俗易懂的方式解释商业概念
        🔍 **多维思考**: 从财务、市场、运营等多角度分析
        🎯 **学习指导**: 提供结构化的学习建议
        
        **💭 您可以这样提问：**
        - "请分析这个公司的商业模式"
        - "这个案例中的关键成功因素是什么？"
        - "从财务角度如何评估这个项目？"
        - "这个策略的风险和机遇在哪里？"
        
        现在就开始您的商业案例学习之旅吧！🚀
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
  
    # 显示历史消息
    for message in st.session_state.messages:
        # 确保message不为None
        if message is not None:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(f"🎓 **学生提问**: {message['content']}")
                else:
                    st.markdown(message["content"])

def create_sidebar():
    """创建侧边栏"""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.header("🛠️ 学习工具")
        
        # 重置对话按钮
        if st.button("🗑️ 开始新的学习会话", use_container_width=True):
            # 保存当前会话到localStorage（通过JavaScript）
            if hasattr(st.session_state, 'messages') and st.session_state.messages is not None and len(st.session_state.messages) > 1:
                save_session_js = f"""
                <script>
                const sessionData = {{
                    timestamp: new Date().toISOString(),
                    messages: {json.dumps(st.session_state.messages)}
                }};
                const sessions = JSON.parse(localStorage.getItem('ultrarag_sessions') || '[]');
                sessions.push(sessionData);
                // 只保留最近10个会话
                if (sessions.length > 10) {{
                    sessions.shift();
                }}
                localStorage.setItem('ultrarag_sessions', JSON.stringify(sessions));
                </script>
                """
                if hasattr(st.components, 'v1'):
                    st.components.v1.html(save_session_js, height=0)
            
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # 显示统计信息
        if hasattr(st.session_state, 'messages') and st.session_state.messages is not None and len(st.session_state.messages) > 1:
            total_messages = len(st.session_state.messages) - 1  # 减去欢迎消息
            questions_asked = (total_messages) // 2
            st.metric("📝 提问次数", questions_asked)
            st.metric("💬 对话轮数", total_messages)
        
        st.markdown("---")
        
        # 学习提示
        st.markdown("### 💡 学习小贴士")
        tips = [
            "🔍 尝试从不同角度分析同一个案例",
            "📊 关注数据背后的商业逻辑",
            "🤔 多问'为什么'和'如何'",
            "📈 将理论与实际案例相结合",
            "🎯 总结关键学习要点"
        ]
        
        for tip in tips:
            st.markdown(f"- {tip}")
        
        st.markdown("---")
        
        # 快速问题模板
        st.markdown("### 🚀 快速提问模板")
        
        question_templates = {
            "商业模式分析": "请分析这个案例中公司的商业模式，包括价值主张、收入来源和关键资源",
            "竞争优势分析": "这个案例中公司的核心竞争优势是什么？如何构建和维持的？",
            "财务表现评估": "从财务角度如何评估这个案例中公司的表现？",
            "市场策略分析": "请分析这个案例中的市场进入策略和定位策略",
            "风险机遇评估": "这个案例中存在哪些主要风险和机遇？"
        }
        
        for template_name, template_text in question_templates.items():
            if st.button(f"📋 {template_name}", use_container_width=True):
                st.session_state.template_question = template_text
        
        st.markdown("---")
        
        # 历史会话管理
        st.markdown("### 📚 历史会话")
        
        # 显示历史会话加载按钮
        if st.button("📖 查看历史会话", use_container_width=True):
            load_sessions_js = """
            <script>
            const sessions = JSON.parse(localStorage.getItem('ultrarag_sessions') || '[]');
            if (sessions.length > 0) {
                const sessionList = sessions.map((session, index) => {
                    const date = new Date(session.timestamp).toLocaleString();
                    const messageCount = session.messages.length - 1; // 减去欢迎消息
                    return `${index + 1}. ${date} (${messageCount}条对话)`;
                }).join('\n');
                alert('历史会话:\n' + sessionList + '\n\n注：历史会话功能正在完善中');
            } else {
                alert('暂无历史会话记录');
            }
            </script>
            """
            if hasattr(st.components, 'v1'):
                st.components.v1.html(load_sessions_js, height=0)
        
        # 清空历史会话按钮
        if st.button("🗑️ 清空历史记录", use_container_width=True):
            clear_sessions_js = """
            <script>
            if (confirm('确定要清空所有历史会话记录吗？此操作不可恢复。')) {
                localStorage.removeItem('ultrarag_sessions');
                alert('历史记录已清空');
            }
            </script>
            """
            if hasattr(st.components, 'v1'):
                st.components.v1.html(clear_sessions_js, height=0)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    init_streamlit()
    
    # 创建侧边栏
    create_sidebar()
  
    # 初始化聊天机器人
    if "chatbot" not in st.session_state:
        try:
            with st.spinner("🔧 正在初始化系统..."):
                st.session_state.chatbot = BusinessCaseRAGChatbot("config/chatbot.yaml")
            st.success("✅ 系统初始化成功！")
        except ValueError as e:
            st.error(f"💥 配置错误: {str(e)}")
            with st.expander("📋 配置检查清单"):
                st.markdown("""
                请检查以下配置：
                1. ✅ `.env` 文件是否存在
                2. ✅ `ALI_EMBEDDING_API_KEY` 是否已配置
                3. ✅ `data/lancedb` 目录是否存在
                4. ✅ 配置文件格式是否正确
                """)
            st.stop()
        except Exception as e:
            st.error(f"💥 系统初始化失败: {str(e)}")
            st.info("请确保配置文件存在且格式正确，或联系技术支持。")
            st.stop()
  
    # 显示聊天历史
    display_chat_history()
  
    # 初始化 session state
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # 处理模板问题
    if "template_question" in st.session_state and not st.session_state.processing:
        prompt = st.session_state.template_question
        del st.session_state.template_question
    else:
        prompt = st.chat_input("💭 请输入您的商业案例问题...")
    
    # 只有在不处理中且有新问题时才处理
    if prompt and not st.session_state.processing:
        # 设置处理标记
        st.session_state.processing = True
        
        # 记录用户问题（轻量级）
        try:
            session_id = st.session_state.get('session_id', 'unknown')
            log_manager.log_user_question(
                question=prompt,
                session_id=session_id,
                metadata={"timestamp": time.time()}
            )
        except Exception as e:
            logger.warning(f"记录用户问题失败: {e}")
        
        # 添加用户消息
        if not hasattr(st.session_state, 'messages') or st.session_state.messages is None:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"🎓 **学生提问**: {prompt}")
      
        # 生成回答
        with st.chat_message("assistant"):
            # 构建对话历史（排除欢迎消息）
            chat_history = []
            # 安全地获取消息列表，确保不为None
            messages = getattr(st.session_state, 'messages', []) or []
            if len(messages) > 2:  # 至少有欢迎消息、用户问题和一个回答
                messages_without_welcome = messages[1:-1]  # 排除欢迎消息和当前问题
                
                # 确保messages_without_welcome不为None
                if messages_without_welcome is not None:
                    for i in range(0, len(messages_without_welcome), 2):
                        if i + 1 < len(messages_without_welcome):
                            user_msg = messages_without_welcome[i]
                            assistant_msg = messages_without_welcome[i + 1]
                            # 确保消息对象不为None且包含content键
                            if (user_msg and assistant_msg and 
                                "content" in user_msg and "content" in assistant_msg):
                                chat_history.append({
                                    "user": user_msg["content"],
                                    "assistant": assistant_msg["content"]
                                })
          
            # 流式生成回答
            # 临时解决方案：避免使用 st.write_stream 以绕过 pyarrow 依赖问题
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.chatbot.chat_stream(prompt, chat_history):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                response = full_response
            except Exception as e:
                logger.error(f"流式生成失败: {e}")
                # 降级到非流式
                response = "抱歉，生成回答时出现问题。请稍后重试。"
                response_placeholder.markdown(response)
      
        # 添加助手消息
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 清除处理标记，允许下一次输入
        st.session_state.processing = False

if __name__ == "__main__":
    main()