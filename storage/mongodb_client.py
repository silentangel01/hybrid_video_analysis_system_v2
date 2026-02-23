"""MongoDB client wrapper for event metadata storage.""" 
# storage/mongodb_client.py

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from backend.models.event import EventModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    MongoDB client for storing event metadata.
    Does NOT store raw images — only metadata and MinIO URL.
    """

    def __init__(self, mongo_uri: str, db_name: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None
        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.db_name]
            self.collection = self.db["events"]
            self.client.server_info()
            logger.info(f"[MongoDB] Connected: {self.mongo_uri}, DB: {self.db_name}")

            # Create indexes for fast querying
            self.collection.create_index([("timestamp", -1)])
            self.collection.create_index([("camera_id", 1)])
            self.collection.create_index([("event_type", 1)])
            self.collection.create_index([("detection_stage", 1)])
            logger.info("[MongoDB] Indexes created")

        except Exception as e:
            logger.error(f"[MongoDB] Connection failed: {e}")
            self.client = None

    def save_event(self, event: EventModel) -> bool:
        """
        Insert event into MongoDB.
        ✅ 修复：使用 model_dump() 替代 dict() 以包含所有字段
        """
        if self.collection is None:
            logger.error("[MongoDB] Collection not initialized")
            return False

        try:
            # ✅ 修复：使用 model_dump() 而不是 dict()
            # dict() 方法不包含默认值，model_dump() 包含所有字段
            event_dict = event.model_dump(exclude_none=True)

            # 确保 created_at 字段存在
            if "created_at" not in event_dict:
                event_dict["created_at"] = datetime.utcnow()

            self.collection.insert_one(event_dict)
            logger.info(f"[MongoDB] Saved event: {event.event_type} from {event.camera_id}")
            logger.debug(f"[MongoDB] Event data: {event_dict.keys()}")
            return True

        except Exception as e:
            logger.error(f"[MongoDB] Save failed: {e}")
            logger.error(f"[MongoDB] Event data that failed: {event.model_dump() if hasattr(event, 'model_dump') else 'N/A'}")
            return False

    def save_events(self, events: List[EventModel]) -> bool:
        """批量插入多个事件 | Batch insert multiple events"""
        if self.collection is None:
            return False

        if not events:
            logger.warning("[MongoDB] No events to save")
            return True

        try:
            # 转换为字典列表
            event_dicts = []
            for event in events:
                event_dict = event.model_dump(exclude_none=True)
                if "created_at" not in event_dict:
                    event_dict["created_at"] = datetime.utcnow()
                event_dicts.append(event_dict)

            # 批量插入
            result = self.collection.insert_many(event_dicts)
            logger.info(f"[MongoDB] Saved {len(result.inserted_ids)} events")
            return True

        except Exception as e:
            logger.error(f"[MongoDB] Batch save failed: {e}")
            return False

    def find_events(
        self,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        detection_stage: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Query events with filters."""
        if self.collection is None:
            return []

        try:
            query = {}
            if camera_id:
                query["camera_id"] = camera_id
            if event_type:
                query["event_type"] = event_type
            if detection_stage:
                query["detection_stage"] = detection_stage
            if start_time or end_time:
                query["timestamp"] = {}
                if start_time:
                    query["timestamp"]["$gte"] = start_time
                if end_time:
                    query["timestamp"]["$lte"] = end_time

            cursor = self.collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
            return list(cursor)

        except Exception as e:
            logger.error(f"[MongoDB] Query failed: {e}")
            return []

    def get_all_events(self, limit: int = 100) -> List[EventModel]:
        """获取所有事件并转换为 EventModel 对象"""
        if self.collection is None:
            return []

        try:
            events = []
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)

            for doc in cursor:
                try:
                    # ✅ 修复：使用新的 EventModel 结构
                    # 移除 _id 字段，因为它不是 EventModel 的一部分
                    doc.pop('_id', None)

                    # 转换文档为 EventModel
                    event = EventModel(**doc)
                    events.append(event)

                except Exception as e:
                    logger.warning(f"[MongoDB] Failed to convert document to EventModel: {e}")
                    logger.debug(f"[MongoDB] Problematic document: {doc}")
                    continue

            logger.info(f"[MongoDB] Retrieved {len(events)} events")
            return events

        except Exception as e:
            logger.error(f"[MongoDB] Failed to fetch events: {e}")
            return []

    def get_events_by_type(self, event_type: str, limit: int = 50) -> List[EventModel]:
        """按事件类型获取事件"""
        return self.get_all_events(
            camera_id=None,
            event_type=event_type,
            limit=limit
        )

    def get_common_space_events(self, limit: int = 50) -> List[EventModel]:
        """获取公共空间分析事件"""
        events = self.find_events(
            event_type="common_space_utilization",
            limit=limit
        )

        # 转换为 EventModel 对象
        event_models = []
        for doc in events:
            try:
                doc.pop('_id', None)
                event_models.append(EventModel(**doc))
            except Exception as e:
                logger.warning(f"Failed to convert common space event: {e}")

        return event_models

    def get_event_count(self, camera_id: Optional[str] = None,
                       event_type: Optional[str] = None) -> int:
        """获取事件数量"""
        if self.collection is None:
            return 0

        try:
            query = {}
            if camera_id:
                query["camera_id"] = camera_id
            if event_type:
                query["event_type"] = event_type

            return self.collection.count_documents(query)

        except Exception as e:
            logger.error(f"[MongoDB] Count failed: {e}")
            return 0

    def delete_events(self, camera_id: Optional[str] = None,
                     before_timestamp: Optional[float] = None) -> int:
        """删除事件（谨慎使用）"""
        if self.collection is None:
            return 0

        try:
            query = {}
            if camera_id:
                query["camera_id"] = camera_id
            if before_timestamp:
                query["timestamp"] = {"$lt": before_timestamp}

            result = self.collection.delete_many(query)
            logger.info(f"[MongoDB] Deleted {result.deleted_count} events")
            return result.deleted_count

        except Exception as e:
            logger.error(f"[MongoDB] Delete failed: {e}")
            return 0

    def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计信息"""
        if self.collection is None:
            return {}

        try:
            # 总事件数
            total = self.collection.count_documents({})

            # 按事件类型统计
            pipeline = [
                {"$group": {
                    "_id": "$event_type",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"}
                }},
                {"$sort": {"count": -1}}
            ]

            type_stats = list(self.collection.aggregate(pipeline))

            # 按摄像头统计
            camera_pipeline = [
                {"$group": {
                    "_id": "$camera_id",
                    "count": {"$sum": 1},
                    "types": {"$addToSet": "$event_type"}
                }},
                {"$sort": {"count": -1}}
            ]

            camera_stats = list(self.collection.aggregate(camera_pipeline))

            return {
                "total_events": total,
                "by_type": type_stats,
                "by_camera": camera_stats,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[MongoDB] Statistics failed: {e}")
            return {}

    def close(self):
        """关闭 MongoDB 连接"""
        if self.client:
            self.client.close()
            logger.info("[MongoDB] Connection closed")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def health_check(self) -> bool:
        """健康检查"""
        try:
            if self.client is None:
                return False
            # 执行一个简单的命令来检查连接
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Health check failed: {e}")
            return False