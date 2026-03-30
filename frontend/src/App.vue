<template>
  <div id="app">
    <h1>Hybrid Video Analysis System</h1>

    <!-- Tab navigation -->
    <div class="tab-nav">
      <button
        :class="['tab-btn', { active: activeTab === 'analysis' }]"
        @click="activeTab = 'analysis'"
      >
        Video Analysis
      </button>
      <button
        :class="['tab-btn', { active: activeTab === 'rtsp' }]"
        @click="activeTab = 'rtsp'"
      >
        RTSP Management
      </button>
    </div>

    <!-- Tab 1: Video Analysis (existing) -->
    <div v-show="activeTab === 'analysis'">
      <UploadVideo />

      <!-- Fire alert modal -->
      <div v-if="fireAlert.show" class="alert-modal">
        <div class="alert-content">
          <div class="alert-icon">🔥</div>
          <h3>Fire Alert!</h3>
          <p>Fire detected in the following sources:</p>
          <ul class="video-list">
            <li v-for="source in fireAlert.sources" :key="source">
              {{ source }}
            </li>
          </ul>
          <button @click="closeAlert">Close</button>
        </div>
      </div>

      <EventList />
    </div>

    <!-- Tab 2: RTSP Management -->
    <div v-show="activeTab === 'rtsp'">
      <StreamManager />
    </div>
  </div>
</template>

<script setup>
import {ref, onMounted, onUnmounted} from 'vue';
import UploadVideo from './components/UploadVideo.vue';
import EventList from './components/EventList.vue';
import StreamManager from './components/StreamManager.vue';

const activeTab = ref('analysis');

const fireAlert = ref({
  show: false,
  sources: []
});

let pollingInterval = null;

async function checkFireEvents() {
  try {
    console.log('🔍 Checking fire events...');
    const response = await fetch('http://localhost:8080/api/events');

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('📥 Received data:', data);

    if (data.success && data.fireDetected && Array.isArray(data.sources) && data.sources.length > 0) {
      const uniqueSources = [...new Set(data.sources.filter(s => s))];
      console.log('🔥 Fire detected, sources:', uniqueSources);

      fireAlert.value = {
        show: true,
        sources: uniqueSources
      };
    } else {
      console.log('ℹ️ No fire detected:', data);
    }
  } catch (error) {
    console.error('❌ Fire event check failed:', error);
  }
}

function closeAlert() {
  fireAlert.value.show = false;
  fireAlert.value.sources = [];
}

onMounted(() => {
  checkFireEvents();
  pollingInterval = setInterval(checkFireEvents, 10000);
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

/* Tab navigation */
.tab-nav {
  display: flex;
  gap: 0;
  margin-bottom: 20px;
  border-bottom: 2px solid #e0e0e0;
}

.tab-btn {
  padding: 10px 24px;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  cursor: pointer;
  font-size: 15px;
  color: #666;
  transition: color 0.2s, border-color 0.2s;
}

.tab-btn:hover {
  color: #333;
}

.tab-btn.active {
  color: #1976d2;
  border-bottom-color: #1976d2;
  font-weight: 600;
}

/* Alert modal */
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
