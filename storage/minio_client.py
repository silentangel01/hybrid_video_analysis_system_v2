# storage/minio_client.py

"""
MinIO Client Wrapper for Secure and Scalable Image Storage.
Handles:
  - Connection to MinIO server
  - Bucket creation (if not exists)
  - Uploading event images with structured naming
  - Generating public URLs for frontend access
"""

import cv2
import io
from minio import Minio
from minio.error import S3Error
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinIOClient:
    """
    A wrapper class for MinIO object storage operations.
    Designed for storing video analysis evidence (e.g., violation snapshots).
    """

    def __init__(
            self,
            endpoint: str,
            access_key: str,
            secret_key: str,
            bucket_name: str,
            secure: bool = False
    ):
        """
        Initialize MinIO client.
        Args:
            endpoint: MinIO server address (e.g., 'localhost:9000')
            access_key: Username for authentication
            secret_key: Password for authentication
            bucket_name: Target bucket name (will be created if not exists)
            secure: Use HTTPS (set False for HTTP/local)
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name.lower()  # Bucket names must be lowercase
        self.secure = secure
        self.client: Optional[Minio] = None
        self._connect()

    def _connect(self):
        """Establish connection to MinIO server and ensure bucket exists."""
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )

            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"✅ Created MinIO bucket: {self.bucket_name}")
            else:
                logger.info(f"✅ Connected to existing bucket: {self.bucket_name}")

        except Exception as e:
            logger.error(f"[MinIO] Failed to connect or initialize: {str(e)}")
            self.client = None

    def upload_frame(
            self,
            image_data,
            camera_id: str,
            timestamp: float,
            event_type: str = "unknown"
    ) -> Optional[str]:
        """
        Upload a single frame to MinIO with structured path and return a public URL.
        The uploaded image can be accessed via the returned URL without authentication.
        Args:
            image_data: OpenCV image (numpy.ndarray) or bytes
            camera_id: Source identifier (e.g., 'camera_front_gate' or 'test_video.mp4')
            timestamp: Unix timestamp in seconds
            event_type: Type of event (e.g., 'parking_violation', 'smoke_detected')
        Returns:
            Public URL for accessing the image, or None on failure.
        """
        if self.client is None:
            logger.error("[MinIO] Client not connected.")
            return None

        try:
            # Convert numpy array to JPEG bytes if needed
            if hasattr(image_data, 'shape'):  # Likely a numpy array from OpenCV
                _, buffer = cv2.imencode('.jpg', image_data)
                img_bytes = buffer.tobytes()
            else:
                img_bytes = image_data  # Assume already in bytes format

            # 🔴 修复时间戳问题 - 添加时间戳验证和调试信息
            current_time = datetime.now()
            logger.info(f"🕒 MinIO Upload Debug:")
            logger.info(f"   - Received timestamp: {timestamp}")
            logger.info(f"   - Converted datetime: {datetime.fromtimestamp(timestamp)}")
            logger.info(f"   - Current system time: {current_time}")

            # 如果时间戳明显错误（比如是未来时间），使用当前时间
            if timestamp > 2000000000:  # 2033年之后的都认为是错误时间戳
                logger.warning(f"⚠️ Suspicious future timestamp detected: {timestamp}")
                logger.warning(f"⚠️ Using current system time instead")
                timestamp = current_time.timestamp()

            dt = datetime.fromtimestamp(timestamp)
            year, month, day, hour = dt.strftime("%Y %m %d %H").split()
            ms = int((timestamp * 1000) % 1000)

            # 使用更清晰的文件名格式
            filename = f"event_{event_type}_{dt.strftime('%H%M%S')}_{ms:03d}.jpg"
            object_name = f"{camera_id}/{year}/{month}/{day}/{hour}/{filename}"

            logger.info(f"📁 Object path: {object_name}")

            # Upload image
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(img_bytes),
                length=len(img_bytes),
                content_type="image/jpeg"
            )
            logger.info(f"✅ [MinIO] Image uploaded: {object_name}")

            # Generate public URL (no presigned needed since bucket is public)
            protocol = "https" if self.secure else "http"
            public_url = f"{protocol}://{self.endpoint}/{self.bucket_name}/{object_name}"

            logger.info(f"🔗 [MinIO] Public URL: {public_url}")
            return public_url

        except S3Error as e:
            logger.error(f"❌ [MinIO] S3 Error during upload: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ [MinIO] Unexpected error during upload: {str(e)}", exc_info=True)
            return None

    def upload_frame_with_datetime(
            self,
            image_data,
            camera_id: str,
            dt: datetime,
            event_type: str = "unknown"
    ) -> Optional[str]:
        """
        替代方法：直接使用 datetime 对象而不是时间戳
        Alternative method: use datetime object directly instead of timestamp.
        """
        try:
            # Convert numpy array to JPEG bytes if needed
            if hasattr(image_data, 'shape'):
                _, buffer = cv2.imencode('.jpg', image_data)
                img_bytes = buffer.tobytes()
            else:
                img_bytes = image_data

            logger.info(f"🕒 Using datetime directly: {dt}")

            year, month, day, hour = dt.strftime("%Y %m %d %H").split()
            filename = f"event_{event_type}_{dt.strftime('%H%M%S')}.jpg"
            object_name = f"{camera_id}/{year}/{month}/{day}/{hour}/{filename}"

            # Upload image
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(img_bytes),
                length=len(img_bytes),
                content_type="image/jpeg"
            )
            logger.info(f"✅ [MinIO] Image uploaded: {object_name}")

            # Generate public URL
            protocol = "https" if self.secure else "http"
            public_url = f"{protocol}://{self.endpoint}/{self.bucket_name}/{object_name}"

            logger.info(f"🔗 [MinIO] Public URL: {public_url}")
            return public_url

        except Exception as e:
            logger.error(f"❌ [MinIO] Error in upload_frame_with_datetime: {str(e)}")
            return None

    def get_public_url(self, object_name: str) -> str:
        """
        Generate public URL for an existing object.
        Args:
            object_name: Full object path in bucket
        Returns:
            Public URL string
        """
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.endpoint}/{self.bucket_name}/{object_name}"

    def list_objects(self, prefix: str = "") -> list:
        """
        List objects in bucket with optional prefix filter.
        """
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except Exception as e:
            logger.error(f"❌ [MinIO] Error listing objects: {str(e)}")
            return []

    def object_exists(self, object_name: str) -> bool:
        """
        Check if an object exists in the bucket.
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False
        except Exception:
            return False

    def close(self):
        """Close the MinIO client connection (optional)."""
        # MinIO Python SDK does not require explicit close
        pass

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


# 调试函数
def debug_timestamp_issue():
    """调试时间戳问题"""
    import time
    current_time = time.time()
    current_dt = datetime.now()

    print("=== 时间戳调试信息 ===")
    print(f"当前系统时间: {current_dt}")
    print(f"当前Unix时间戳: {current_time}")
    print(f"转换回日期: {datetime.fromtimestamp(current_time)}")

    # 测试问题时间戳
    problem_timestamp = 1761401166.298111
    print(f"\n问题时间戳: {problem_timestamp}")
    print(f"对应日期: {datetime.fromtimestamp(problem_timestamp)}")

    # 检查时间戳是否合理
    if problem_timestamp > current_time:
        print("❌ 时间戳是未来的时间！")
    else:
        print("✅ 时间戳是过去的时间")


if __name__ == "__main__":
    debug_timestamp_issue()