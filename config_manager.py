"""
配置管理器
用于管理API密钥和配置文件
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 配置日志
logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self):
        """初始化配置管理器"""
        self.env_file = Path(".env")
        self._load_env_file()
    
    def _load_env_file(self):
        """加载.env文件"""
        if self.env_file.exists():
            load_dotenv(dotenv_path=self.env_file)
            logger.info(f"已加载环境变量文件: {self.env_file}")
    
    def create_env_file_if_missing(self):
        """如果.env文件不存在，创建它"""
        if not self.env_file.exists():
            logger.info(f"创建.env文件: {self.env_file}")
            # 创建基本的.env文件模板
            env_template = """# UltraRAG 环境变量配置
# 阿里云 DashScope API 密钥（用于嵌入和生成）
ALI_EMBEDDING_API_KEY=your_api_key_here

# OpenAI API 密钥（可选，如果使用OpenAI服务）
OPENAI_API_KEY=your_openai_key_here
"""
            try:
                with open(self.env_file, 'w', encoding='utf-8') as f:
                    f.write(env_template)
                logger.info(f"已创建.env文件模板: {self.env_file}")
            except Exception as e:
                logger.error(f"创建.env文件失败: {e}")
                raise
        else:
            # 即使文件存在，也重新加载以确保最新值
            self._load_env_file()
    
    def check_required_keys(self) -> Dict[str, bool]:
        """检查必需的API密钥是否已配置
        
        Returns:
            Dict[str, bool]: 服务名称和是否可用的映射
        """
        key_status = {}
        
        # 检查 DashScope API 密钥
        dashscope_key = os.getenv("ALI_EMBEDDING_API_KEY")
        key_status["dashscope"] = bool(dashscope_key and dashscope_key != "your_api_key_here")
        
        # 检查 OpenAI API 密钥（可选）
        openai_key = os.getenv("OPENAI_API_KEY")
        key_status["openai"] = bool(openai_key and openai_key != "your_openai_key_here")
        
        return key_status
    
    def get_config_with_fallback(
        self, 
        config: Dict[str, Any], 
        key: str, 
        service: str
    ) -> Optional[str]:
        """从配置中获取值，如果不存在则从环境变量获取
        
        Args:
            config: 配置字典
            key: 配置键名
            service: 服务名称（用于环境变量回退）
            
        Returns:
            str: 配置值或环境变量值，如果都不存在则返回None
        """
        # 首先尝试从配置中获取
        if config and isinstance(config, dict):
            value = config.get(key)
            if value and isinstance(value, str) and value.strip():
                # 如果值是环境变量引用格式 ${VAR_NAME}，则解析它
                if value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_value = os.getenv(env_var)
                    if env_value:
                        return env_value
                else:
                    return value
        
        # 如果配置中没有，尝试从环境变量获取
        # 根据服务类型构建环境变量名
        if service == "dashscope":
            env_key = "ALI_EMBEDDING_API_KEY"
        elif service == "openai":
            env_key = "OPENAI_API_KEY"
        else:
            # 默认使用服务名构建环境变量
            env_key = f"{service.upper()}_API_KEY"
        
        env_value = os.getenv(env_key)
        if env_value and env_value != "your_api_key_here":
            return env_value
        
        return None
    
    def validate_api_key(self, api_key: Optional[str]) -> bool:
        """验证API密钥是否有效
        
        Args:
            api_key: API密钥字符串
            
        Returns:
            bool: 如果密钥有效返回True，否则返回False
        """
        if not api_key:
            return False
        
        if not isinstance(api_key, str):
            return False
        
        # 去除首尾空格
        api_key = api_key.strip()
        
        # 检查是否为空或占位符
        if not api_key or api_key in ["your_api_key_here", "your_openai_key_here"]:
            return False
        
        # 基本格式检查（DashScope API密钥通常以sk-开头）
        if api_key.startswith("sk-") and len(api_key) > 10:
            return True
        
        # OpenAI API密钥也以sk-开头
        if api_key.startswith("sk-") and len(api_key) > 20:
            return True
        
        # 如果长度合理，也认为有效（允许其他格式）
        if len(api_key) >= 10:
            return True
        
        return False
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置信息（不包含敏感信息）"""
        return {
            "env_file_exists": self.env_file.exists(),
            "key_status": self.check_required_keys(),
            "env_vars": {
                "LANCEDB_PATH": os.getenv("LANCEDB_PATH", "data/lancedb"),
                "LANCEDB_TABLE": os.getenv("LANCEDB_TABLE", "documents"),
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
            }
        }
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """验证配置完整性
        
        Returns:
            tuple: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查.env文件
        if not self.env_file.exists():
            errors.append("缺少.env配置文件")
        
        # 检查必需的API密钥
        key_status = self.check_required_keys()
        if not key_status.get("dashscope"):
            errors.append("缺少DashScope API密钥 (ALI_EMBEDDING_API_KEY)")
        
        # 检查数据目录
        lancedb_path = os.getenv("LANCEDB_PATH", "data/lancedb")
        if not os.path.exists(lancedb_path):
            errors.append(f"LanceDB数据库目录不存在: {lancedb_path}")
        
        return len(errors) == 0, errors

# 创建全局配置管理器实例
config_manager = ConfigManager()

