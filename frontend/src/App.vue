<template>
  <div id="app">
    <h1>Hybrid Video Analysis System</h1>
    <UploadVideo />

    <!-- 火情报警弹窗 -->
    <div v-if="fireAlert.show" class="alert-modal">
      <div class="alert-content">
        <div class="alert-icon">🔥</div>
        <h3>火情警报！</h3>
        <p>在以下视频中检测到火情：</p>
        <ul class="video-list">
          <li v-for="source in fireAlert.sources" :key="source">
            {{ source }}
          </li>
        </ul>
        <button @click="closeAlert">关闭</button>
      </div>
    </div>
    <!-- 新增：事件列表 -->
    <EventList />
  </div>
</template>

<script setup>
import {ref, onMounted, onUnmounted} from 'vue';
import UploadVideo from './components/UploadVideo.vue';
import EventList from './components/EventList.vue';

// 定义响应式报警状态
const fireAlert = ref({
  show: false,
  sources: []
});

let pollingInterval = null;

async function checkFireEvents() {
  try {
    console.log('🔍 开始检查火情事件...');
    const response = await fetch('http://localhost:8080/api/events');

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('📥 收到数据:', data);

    if (data.success && data.fireDetected && Array.isArray(data.sources) && data.sources.length > 0) {
      const uniqueSources = [...new Set(data.sources.filter(s => s))];
      console.log('🔥 检测到火情，来源:', uniqueSources);

      // 更新报警状态（ref.value 是响应式的）
      fireAlert.value = {
        show: true,
        sources: uniqueSources
      };
    } else {
      console.log('ℹ️ 未检测到火情或数据为空:', data);
    }
  } catch (error) {
    console.error('❌ 检查火情事件失败:', error);
  }
}

function closeAlert() {
  fireAlert.value.show = false;
  fireAlert.value.sources = [];
}

onMounted(() => {
  checkFireEvents(); // 立即检查一次
  pollingInterval = setInterval(checkFireEvents, 10000); // 每10秒轮询
});

onUnmounted(() => {
  if (pollingInterval) {
    clearInterval(pollingInterval);
  }
});



</script>

<style scoped>
#app {
  font-family: Arial, sans-serif;
  padding: 20px;
}

/* 弹窗样式 - 居中显示 */
.alert-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.alert-content {
  background: gray;
  padding: 25px;
  border-radius: 12px;
  max-width: 450px;
  width: 90%;
  text-align: center;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.alert-icon {
  font-size: 2.5rem;
  margin-bottom: 12px;
}

.video-list {
  list-style: none;
  padding: 0;
  margin: 15px 0;
}

.video-list li {
  background: #f8f9fa;
  padding: 8px 12px;
  margin: 6px 0;
  border-radius: 6px;
  font-family: monospace;
  color: #d32f2f;
  word-break: break-all;
}

button {
  padding: 10px 20px;
  background-color: #e53935;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  margin-top: 10px;
}

button:hover {
  background-color: #c62828;
}
</style>