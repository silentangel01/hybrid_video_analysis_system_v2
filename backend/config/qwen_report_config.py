"""Qwen text report configuration for common-space summarisation."""

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class QwenReportAPIConfig:
    """Configuration loader for common-space report LLM calls."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__),
            "qwen_report_config.json",
        )
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            logger.warning("Qwen report config file not found: %s", self.config_path)
            return self._get_default_config()
        except Exception as exc:
            logger.error("Failed to load Qwen report config: %s", exc)
            return self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        return {
            "qwen_report": {
                "api_url": os.getenv("QWEN_REPORT_API_URL", os.getenv("QWEN_VL_API_URL", "")),
                "api_key": os.getenv("QWEN_REPORT_API_KEY", os.getenv("QWEN_VL_API_KEY", "")),
                "model_name": os.getenv("QWEN_REPORT_MODEL_NAME", "qwen-plus"),
                "timeout": int(os.getenv("QWEN_REPORT_TIMEOUT", "30")),
                "temperature": float(os.getenv("QWEN_REPORT_TEMPERATURE", "0.2")),
                "max_tokens": int(os.getenv("QWEN_REPORT_MAX_TOKENS", "700")),
                "language": os.getenv("QWEN_REPORT_LANGUAGE", "zh-CN"),
            }
        }

    def is_configured(self) -> bool:
        section = self._config["qwen_report"]
        return bool(section.get("api_url") and section.get("api_key"))

    def get_api_url(self) -> str:
        return self._config["qwen_report"]["api_url"]

    def get_api_key(self) -> str:
        return self._config["qwen_report"]["api_key"]

    def get_model_name(self) -> str:
        return self._config["qwen_report"]["model_name"]

    def get_timeout(self) -> int:
        return self._config["qwen_report"]["timeout"]

    def get_temperature(self) -> float:
        return self._config["qwen_report"]["temperature"]

    def get_max_tokens(self) -> int:
        return self._config["qwen_report"]["max_tokens"]

    def get_language(self) -> str:
        return self._config["qwen_report"]["language"]


qwen_report_api_config = QwenReportAPIConfig()
