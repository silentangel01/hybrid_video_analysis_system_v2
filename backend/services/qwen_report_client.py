"""Tongyi Qwen text client for common-space report summarisation."""

import json
import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class QwenReportClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "qwen-plus",
        timeout: int = 30,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def summarize_common_space_report(
        self,
        report: Dict[str, Any],
        language: str = "en-US",
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt(language)
        user_prompt = self._build_user_prompt(report, language)
        payload = self._build_payload(system_prompt, user_prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        summary = self._extract_text(response.json()).strip()
        return {
            "summary": summary,
            "provider": "tongyi_qwen",
            "model": self.model_name,
            "language": language,
        }

    @staticmethod
    def _build_system_prompt(language: str) -> str:
        if language.lower().startswith("zh"):
            return (
                "You are a public-space operations analysis assistant. "
                "The user prefers Chinese, so write an accurate, formal, and concise Chinese summary. "
                "Use only the supplied report data and do not invent facts."
            )

        return (
            "You are a public-space operations analysis assistant. "
            "Write a concise, accurate, and professional English summary from the provided structured report. "
            "Your response must be in English only. "
            "Do not invent facts beyond the supplied data."
        )

    @staticmethod
    def _build_user_prompt(report: Dict[str, Any], language: str) -> str:
        report_payload = {
            "report_key": report.get("report_key", {}),
            "window": report.get("window", {}),
            "stats": report.get("stats", {}),
            "highlights": report.get("highlights", {}),
            "sample_events": report.get("sample_events", []),
            "rule_based_narrative": report.get("narrative", ""),
        }

        if language.lower().startswith("zh"):
            instruction = (
                "Generate the final answer in Chinese. "
                "Use 1 to 2 short paragraphs of natural language. "
                "Mention overall occupancy, people flow, major activities, and safety observations. "
                "If there are no events, say so clearly. "
                "Do not output JSON or bullet points."
            )
        else:
            instruction = (
                "Generate a concise English summary from the following common-space report. "
                "Use 1 to 2 short paragraphs of natural language. "
                "Mention overall occupancy, people flow, major activities, and safety observations. "
                "If there are no events, say so clearly. "
                "Do not output JSON, bullet points, or any non-English text."
            )

        return f"{instruction}\n\nReport JSON:\n{json.dumps(report_payload, ensure_ascii=False, indent=2)}"

    def _build_payload(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if "openai" in self.api_url or "v1/chat/completions" in self.api_url:
            return {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

        if "dashscope" in self.api_url:
            return {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                },
                "parameters": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            }

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, list):
            if not data:
                return ""
            first = data[0]
            if isinstance(first, dict):
                for key in ("content", "text", "message", "result"):
                    if key in first:
                        value = first[key]
                        return self._extract_text(value) if isinstance(value, (dict, list)) else str(value).strip()
            if isinstance(first, str):
                return " ".join(str(item).strip() for item in data)
            return ""

        if not isinstance(data, dict):
            return str(data).strip()

        if "choices" in data:
            content = data["choices"][0].get("message", {}).get("content", "")
            return self._extract_text(content) if isinstance(content, (dict, list)) else str(content).strip()

        if "output" in data and "choices" in data["output"]:
            content = data["output"]["choices"][0].get("message", {}).get("content", "")
            return self._extract_text(content) if isinstance(content, (dict, list)) else str(content).strip()

        for key in ("content", "text", "result", "message"):
            if key in data:
                value = data[key]
                return self._extract_text(value) if isinstance(value, (dict, list)) else str(value).strip()

        logger.warning("Unrecognised report API response format: %s", data)
        return str(data).strip()
