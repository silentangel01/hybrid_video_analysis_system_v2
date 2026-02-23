<!-- frontend/src/components/EventList.vue -->
<template>
  <div class="event-list-container">
    <div class="header">
      <h2>事件记录</h2>
      <button @click="refresh" :disabled="loading" class="refresh-btn">
        {{ loading ? '刷新中...' : '🔄 刷新' }}
      </button>
    </div>

    <!-- 错误提示 -->
    <div v-if="error" class="error-message">
      ❌ 加载失败：{{ error }}
      <button @click="fetchEvents" class="retry-btn">重试</button>
    </div>

    <!-- 加载状态 -->
    <div v-else-if="loading && events.length === 0" class="loading">
      正在加载事件记录...
    </div>

    <!-- 无数据 -->
    <div v-else-if="events.length === 0" class="no-data">
      📭 暂无事件记录
    </div>

    <!-- 事件列表 -->
    <div v-else class="events-grid">
      <div v-for="event in paginatedEvents" :key="event._id" class="event-card">
        <div class="event-header">
          <span class="event-type" :class="event.event_type">{{ event.event_type }}</span>
          <span class="confidence">{{ (event.confidence * 100).toFixed(1) }}%</span>
        </div>
        <p><strong>视频:</strong> {{ event.camera_id }}</p>
        <p><strong>时间:</strong> {{ formatTimestamp(event.timestamp) }}</p>

        <!-- 公共空间利用率 -->
        <div v-if="event.event_type === 'common_space_utilization' && event.analysis_summary?.space_occupancy">
          <p><strong>占用率:</strong> {{ event.analysis_summary.space_occupancy }}</p>
        </div>

        <!-- 描述文本 -->
        <div v-if="event.description">
          <p><strong>描述:</strong> {{ event.description }}</p>
        </div>

        <div v-if="event.event_type === 'car' && event.description?.includes('no-parking')">
          <span class="badge">🚫 违停</span>
        </div>

        <!-- 图片预览 -->
        <div v-if="event.image_url" class="image-preview">
          <img
              :src="event.image_url"
              :alt="event.event_type"
              @click="openImage(event.image_url)"
              @error="handleImageError"
              loading="lazy"
          />
        </div>
        <div v-else class="no-image">📷 无截图</div>
      </div>
    </div>

    <!-- 分页控件 -->
    <div v-if="events.length > itemsPerPage" class="pagination">
      <button
          @click="currentPage--"
          :disabled="currentPage <= 1"
          class="page-btn"
      >
        上一页
      </button>
      <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
      <button
          @click="currentPage++"
          :disabled="currentPage >= totalPages"
          class="page-btn"
      >
        下一页
      </button>
    </div>

    <!-- 图片放大查看器 -->
    <div v-if="fullImage" class="image-modal" @click="closeFullImage">
      <img :src="fullImage" alt="大图" @click.stop />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue';

const events = ref([]);
const loading = ref(false);
const error = ref(null);
const fullImage = ref(null);
const currentPage = ref(1);
const itemsPerPage = 10; // 每页显示10条

// 自动刷新定时器
let refreshInterval = null;

// 分页计算
const paginatedEvents = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage;
  return events.value.slice(start, start + itemsPerPage);
});

const totalPages = computed(() => Math.ceil(events.value.length / itemsPerPage));

async function fetchEvents() {
  loading.value = true;
  error.value = null;
  try {
    const response = await fetch('http://localhost:8080/api/events-all');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    if (data.success) {
      events.value = data.events || [];
      currentPage.value = 1; // 重置到第一页
    } else {
      throw new Error(data.error || '未知错误');
    }
  } catch (err) {
    console.error('获取事件失败:', err);
    error.value = err.message || '网络请求失败';
  } finally {
    loading.value = false;
  }
}

function refresh() {
  fetchEvents();
}

function formatTimestamp(timestamp) {
  const date = new Date(timestamp * 1000);
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}

function openImage(url) {
  fullImage.value = url;
}

function closeFullImage() {
  fullImage.value = null;
}

function handleImageError(e) {
  e.target.alt = '⚠️ 图片加载失败';
  e.target.style.opacity = '0.6';
}

// 初始化 & 自动刷新
onMounted(() => {
  fetchEvents();
  refreshInterval = setInterval(fetchEvents, 10000); // 每10秒自动刷新
});

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
});
</script>

<style scoped>
.event-list-container {
  margin-top: 30px;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.event-list-container h2 {
  color: black;
  font-weight: bold;
  padding-bottom: 4px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.refresh-btn {
  padding: 6px 12px;
  background: #4caf50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.refresh-btn:hover:not(:disabled) {
  background: #45a049;
}
.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-message {
  color: #d32f2f;
  padding: 12px;
  background: #ffebee;
  border-radius: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.retry-btn {
  padding: 4px 10px;
  background: #d32f2f;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.loading, .no-data {
  text-align: center;
  padding: 20px;
  color: #666;
}

.events-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 15px;
}

.event-card {
  background: gray;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  transition: transform 0.2s;
}
.event-card:hover {
  transform: translateY(-3px);
}

.event-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
}

.event-type {
  padding: 4px 8px;
  border-radius: 4px;
  color: white;
  font-size: 0.85rem;
  text-transform: uppercase;
}
.event-type.fire { background: #e53935; }
.event-type.smoke { background: #5d4037; }
.event-type.common_space_utilization { background: #09cd31; }

.confidence {
  font-weight: bold;
  color: #4caf50;
}

.image-preview img {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: zoom-in;
  transition: opacity 0.3s;
}

.no-image {
  color: #999;
  font-style: italic;
  text-align: center;
  padding: 10px;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 15px;
  margin-top: 20px;
  padding: 10px;
  color: #1f2e9f;
}

.page-btn {
  padding: 6px 12px;
  background: #2196f3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.image-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0,0,0,0.9);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2000;
}
.image-modal img {
  max-width: 90%;
  max-height: 90%;
  object-fit: contain;
}

.badge {
  background-color: #e74c3c;
  color: white;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

</style>