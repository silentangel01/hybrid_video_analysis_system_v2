# scripts/debug_qwen_vl.py
"""
Qwen-VL 配置诊断脚本
"""

import os
import sys
import logging

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_qwen_vl_config():
    """诊断Qwen-VL配置"""
    print("🔍 Qwen-VL Configuration Diagnosis")
    print("=" * 50)

    # 检查环境变量
    print("\n📋 Environment Variables:")
    env_vars = [
        "QWEN_VL_API_URL",
        "QWEN_VL_API_KEY",
        "QWEN_VL_MODEL_NAME",
        "QWEN_VL_TIMEOUT"
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            # 对API_KEY进行部分隐藏
            if "API_KEY" in var and len(value) > 8:
                masked_value = value[:4] + "*" * 8 + value[-4:]
            else:
                masked_value = value
            print(f"   ✅ {var}: {masked_value}")
        else:
            print(f"   ❌ {var}: Not set")

    # 检查配置文件
    print("\n📁 Config File Status:")
    try:
        from backend.config.qwen_vl_config import qwen_vl_api_config
        config_path = qwen_vl_api_config.config_path
        if os.path.exists(config_path):
            print(f"   ✅ Config file exists: {config_path}")
            # 读取配置内容
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"   📊 Config content: {config_data}")
        else:
            print(f"   ❌ Config file not found: {config_path}")
    except Exception as e:
        print(f"   ❌ Error reading config: {e}")

    # 检查配置对象
    print("\n⚙️ Config Object Status:")
    try:
        from backend.config.qwen_vl_config import qwen_vl_api_config
        print(f"   API URL: {qwen_vl_api_config.get_api_url()}")
        api_key = qwen_vl_api_config.get_api_key()
        if api_key:
            print(f"   API Key: {api_key[:8]}... (first 8 chars)")
        else:
            print(f"   API Key: Not set")
        print(f"   Model Name: {qwen_vl_api_config.get_model_name()}")
        print(f"   Timeout: {qwen_vl_api_config.get_timeout()}")
        print(f"   Is Configured: {qwen_vl_api_config.is_configured()}")
    except Exception as e:
        print(f"   ❌ Error accessing config object: {e}")

    # 检查主配置
    print("\n🔧 Main Configuration Check:")
    try:
        from backend.main import load_config
        cfg = load_config()
        print(f"   QWEN_VL_API_URL from main: {cfg.get('qwen_vl_api_url')}")
        print(f"   QWEN_VL_API_KEY from main: {cfg.get('qwen_vl_api_key')}")
        print(f"   QWEN_VL_MODEL_NAME from main: {cfg.get('qwen_vl_model_name')}")
    except Exception as e:
        print(f"   ❌ Error checking main config: {e}")

    # 检查当前工作目录和.env文件
    print("\n📂 File System Check:")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Project root: {project_root}")

    env_file = os.path.join(project_root, '.env')
    if os.path.exists(env_file):
        print(f"   ✅ .env file exists: {env_file}")
        # 读取.env文件内容（隐藏敏感信息）
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.read()
        # 隐藏API密钥
        import re
        masked_env = re.sub(r'QWEN_VL_API_KEY=([^\n]+)', r'QWEN_VL_API_KEY=****', env_content)
        print(f"   📄 .env content:\n{masked_env}")
    else:
        print(f"   ❌ .env file not found: {env_file}")

        # 检查其他可能的.env文件位置
        possible_locations = [
            os.path.join(project_root, '.env'),
            os.path.join(project_root, 'backend', '.env'),
            os.path.join(os.getcwd(), '.env')
        ]
        for location in possible_locations:
            if os.path.exists(location):
                print(f"   🔍 Found .env at: {location}")


if __name__ == "__main__":
    diagnose_qwen_vl_config()