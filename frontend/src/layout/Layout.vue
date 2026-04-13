<template>
  <div class="layout">
    <Sidebar />
    <div class="layout-main">
      <header class="layout-header">
        <h1 class="page-title">{{ pageTitle }}</h1>
        <div class="header-right">
          <span class="header-time">{{ currentTime }}</span>
          <button class="alert-bell" @click="showFireModal = true" :class="{ 'has-alert': fireAlert.active }">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 01-3.46 0"/>
            </svg>
            <span v-if="fireAlert.active" class="alert-dot"></span>
          </button>
        </div>
      </header>
      <main class="layout-content">
        <router-view />
      </main>
    </div>

    <!-- 火警弹窗 -->
    <div v-if="showFireModal && fireAlert.active" class="fire-modal-overlay" @click="showFireModal = false">
      <div class="fire-modal" @click.stop>
        <div class="fire-modal-header">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="var(--color-danger)" stroke="none">
            <path d="M12 2c-5.33 4.55-8 8.48-8 11.8 0 4.98 3.8 8.2 8 8.2s8-3.22 8-8.2c0-3.32-2.67-7.25-8-11.8zm0 18c-3.35 0-6-2.57-6-6.2 0-2.34 1.95-5.44 6-9.14 4.05 3.7 6 6.79 6 9.14 0 3.63-2.65 6.2-6 6.2z"/>
          </svg>
          <h3>Fire Alert</h3>
        </div>
        <p class="fire-modal-desc">Smoke/fire detected from the following sources:</p>
        <ul class="fire-source-list">
          <li v-for="source in fireAlert.sources" :key="source">{{ source }}</li>
        </ul>
        <button class="btn-primary" @click="showFireModal = false">Dismiss</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import Sidebar from './Sidebar.vue'
import { useFireAlert } from '../composables/useFireAlert'

const route = useRoute()
const { fireAlert } = useFireAlert()
const showFireModal = ref(false)
const currentTime = ref('')

const pageTitle = computed(() => route.meta?.title || 'Dashboard')

let timeTimer = null

function updateTime() {
  const now = new Date()
  currentTime.value = now.toLocaleString('en-US', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

onMounted(() => {
  updateTime()
  timeTimer = setInterval(updateTime, 1000)
})

onUnmounted(() => {
  if (timeTimer) clearInterval(timeTimer)
})
</script>

<style scoped>
.layout {
  display: flex;
  min-height: 100vh;
}

.layout-main {
  flex: 1;
  margin-left: var(--sidebar-width);
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.layout-header {
  height: var(--header-height);
  background: var(--color-bg-secondary);
  border-bottom: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  position: sticky;
  top: 0;
  z-index: 50;
}

.page-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-time {
  color: var(--color-text-secondary);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.alert-bell {
  position: relative;
  background: transparent;
  padding: 6px;
  border-radius: var(--radius-sm);
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
}

.alert-bell:hover {
  background: var(--color-bg-tertiary);
}

.alert-bell.has-alert {
  color: var(--color-danger);
}

.alert-dot {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 8px;
  height: 8px;
  background: var(--color-danger);
  border-radius: 50%;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.layout-content {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
}

/* 火警弹窗 */
.fire-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.fire-modal {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-danger);
  border-radius: var(--radius-lg);
  padding: 24px;
  max-width: 420px;
  width: 90%;
}

.fire-modal-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.fire-modal-header h3 {
  font-size: 18px;
  color: var(--color-danger);
}

.fire-modal-desc {
  color: var(--color-text-secondary);
  margin-bottom: 12px;
}

.fire-source-list {
  list-style: none;
  margin-bottom: 20px;
}

.fire-source-list li {
  background: var(--color-bg-tertiary);
  padding: 8px 12px;
  margin-bottom: 6px;
  border-radius: var(--radius-sm);
  font-family: monospace;
  font-size: 13px;
  color: var(--color-danger);
  border-left: 3px solid var(--color-danger);
}

.btn-primary {
  background: var(--color-accent);
  color: #fff;
  padding: 8px 24px;
  border-radius: var(--radius-sm);
  width: 100%;
}

.btn-primary:hover {
  background: var(--color-accent-hover);
}
</style>
