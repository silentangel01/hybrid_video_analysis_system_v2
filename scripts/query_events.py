# scripts/query_events.py
"""
Query and display all stored events from MongoDB.
Fixed:
  - Module import path for 'backend'
  - MongoDBClient 初始化参数缺失问题
  - URL 显示断裂 → 独立一行 + 写入文件
"""

import sys
import os
from datetime import datetime
import logging

# 👇 修复导入路径：将项目根目录加入 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.models.event import EventModel
from storage.mongodb_client import MongoDBClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 输出 URL 到文件，避免终端换行截断
URL_OUTPUT_FILE = "event_urls.txt"


def main():
    # 清空或创建 URL 文件
    with open(URL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Event Image URLs (Copy & Paste in Browser) ===\n\n")

    # 👇 修复：传入 MongoDB 连接参数
    MONGO_URI = "mongodb://localhost:27017"
    DB_NAME = "video_analysis_db"

    mongo_client = MongoDBClient(mongo_uri=MONGO_URI, db_name=DB_NAME)
    if not mongo_client.client:
        logger.error("❌ Failed to connect to MongoDB")
        return

    # 查询数量为limit个事件
    events = mongo_client.get_all_events(limit=10)  # 可调整 limit

    if not events:
        print("🔍 No events found in database.")
        return

    print(f"\n🔍 Found {len(events)} event(s) in MongoDB:\n")

    for i, event in enumerate(events, 1):
        # 格式化时间
        dt = datetime.fromtimestamp(event.timestamp) if event.timestamp else "N/A"

        print(f"📌 Event #{i}")
        print(f"   📹 Source: {event.camera_id}")
        print(f"   ⏱️  Time: {dt} (Unix: {event.timestamp})")
        print(f"   🚨 Type: {event.event_type} (Confidence: {event.confidence:.2f})")
        print(f"   📦 BBox: {event.bbox}")

        # ✅ 修复 URL 显示断裂：独立一行 + 写入文件
        print(f"🖼️  Image URL:")
        print(f"{event.image_url}\n")  # 单独一行，避免 wrap

        # 同时写入文件便于复制
        with open(URL_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"Event #{i} | {event.camera_id} | {dt}\n")
            f.write(f"{event.image_url}\n")
            f.write("-" * 100 + "\n\n")

        print("   --------------------------------------------------")

    print(f"\n✅ All image URLs also saved to: {URL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
