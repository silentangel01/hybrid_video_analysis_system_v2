# backend/config/qwen_vl_config.py
"""
Qwen-VL API 配置管理
Qwen-VL API Configuration Management
"""

import json
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class QwenVLAPIConfig:
    """Qwen-VL API 配置类 | Qwen-VL API Configuration Class"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "qwen_vl_config.json"
        )
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件 | Load configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"⚠️ Qwen-VL config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen-VL config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置 | Get default configuration"""
        return {
            "qwen_vl": {
                "api_url": os.getenv("QWEN_VL_API_URL", ""),
                "api_key": os.getenv("QWEN_VL_API_KEY", ""),
                "model_name": os.getenv("QWEN_VL_MODEL_NAME", "qwen-vl-plus"),
                "timeout": int(os.getenv("QWEN_VL_TIMEOUT", "30")),
                "max_retries": 3,
                "verify_prompt": "请仔细分析这张图片中是否有烟雾或火焰。只回答'是'或'否'，不要解释。",
                "temperature": 0.1,
                "max_tokens": 10
            }
        }

    def get_api_url(self) -> str:
        return self._config["qwen_vl"]["api_url"]

    def get_api_key(self) -> str:
        return self._config["qwen_vl"]["api_key"]

    def get_model_name(self) -> str:
        return self._config["qwen_vl"]["model_name"]

    def get_timeout(self) -> int:
        return self._config["qwen_vl"]["timeout"]

    def get_verify_prompt(self) -> str:
        return self._config["qwen_vl"]["verify_prompt"]

    def is_configured(self) -> bool:
        """检查配置是否完整 | Check if configuration is complete"""
        return bool(self.get_api_url() and self.get_api_key())


# 全局配置实例 | Global config instance
qwen_vl_api_config = QwenVLAPIConfig()