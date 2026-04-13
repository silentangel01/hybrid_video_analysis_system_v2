<template>
  <div class="dashboard">
    <!-- 统计卡片 -->
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-icon stat-icon--blue">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
          </svg>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ stats.activeStreams }}</span>
          <span class="stat-label">活跃视频流</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon stat-icon--green">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
          </svg>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ stats.todayEvents }}</span>
          <span class="stat-label">今日事件</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon stat-icon--red">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2c-5.33 4.55-8 8.48-8 11.8 0 4.98 3.8 8.2 8 8.2s8-3.22 8-8.2c0-3.32-2.67-7.25-8-11.8z"/>
          </svg>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ stats.fireAlerts }}</span>
          <span class="stat-label">烟火告警</span>
        </div>
      </div>
    </div>

    <!-- 双栏面板 -->
    <div class="panel-row">
      <!-- 最近事件 -->
      <div class="panel">
        <div class="panel-header">
          <h3>最近事件</h3>
          <router-link to="/events" class="panel-link">查看全部</router-link>
        </div>
        <div class="panel-body">
          <div v-if="recentEvents.length === 0" class="panel-empty">暂无事件</div>
          <div v-for="evt in recentEvents" :key="evt._id" class="event-row">
            <span class="event-type-dot" :class="'dot-' + evt.event_type"></span>
            <span class="event-type-text">{{ typeLabel(evt.event_type) }}</span>
            <span class="event-camera">{{ evt.camera_id }}</span>
            <span class="event-time">{{ formatTime(evt.timestamp) }}</span>
          </div>
        </div>
      </div>

      <!-- 视频流状态 -->
      <div class="panel">
        <div class="panel-header">
          <h3>视频流状态</h3>
          <router-link to="/streams" class="panel-link">管理</router-link>
        </div>
        <div class="panel-body">
          <div v-if="streams.length === 0" class="panel-empty">暂无视频流</div>
          <template v-for="s in streams" :key="s.stream_id">
            <div class="stream-row">
              <span :class="['status-dot', 'dot-status-' + s.status]"></span>
              <span class="stream-id">{{ s.stream_id }}</span>
              <span class="stream-tasks">
                <span v-for="t in s.tasks" :key="t" class="mini-tag">{{ taskLabel(t) }}</span>
              </span>
              <span :class="['stream-status', 'text-' + s.status]">{{ statusLabel(s.status) }}</span>
            </div>
            <div v-if="s.metrics" class="stream-metrics-row">
              <span class="sm-item">cap <b>{{ fmtF(s.metrics.capture?.capture_fps_10s) }}</b> fps</span>
              <span class="sm-item">emit <b>{{ fmtF(s.metrics.capture?.emit_fps_10s) }}</b> fps</span>
              <span class="sm-item">queue <b>{{ fmtN(s.metrics.executor?.queue_size) }}</b></span>
              <span class="sm-item">inflight <b>{{ fmtN(s.metrics.executor?.inflight_tasks) }}</b></span>
            </div>
            <div v-if="s.bottleneck_hints?.length" class="stream-bottleneck-row">
              <span v-for="(hint, i) in s.bottleneck_hints" :key="i" class="bottleneck-text">{{ hint }}</span>
            </div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const stats = ref({ activeStreams: 0, todayEvents: 0, fireAlerts: 0 })
const recentEvents = ref([])
const streams = ref([])

let refreshTimer = null

function typeLabel(t) {
  const map = { smoke_flame: '烟火告警', parking_violation: '违停检测', common_space_utilization: '空间分析' }
  return map[t] || t
}

function taskLabel(t) {
  const map = { parking_violation: '违停', smoke_flame: '烟火', common_space: '空间' }
  return map[t] || t
}

function statusLabel(s) {
  const map = { running: '运行中', connecting: '连接中', stopped: '已停止', error: '异常' }
  return map[s] || s
}

function formatTime(ts) {
  const d = new Date(ts * 1000)
  return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function fmtF(v) {
  if (v == null) return '0.0'
  return Number(v).toFixed(1)
}

function fmtN(v) {
  if (v == null) return '0'
  return String(Number(v))
}

async function fetchAll() {
  // 并行获取
  const [streamsRes, eventsRes] = await Promise.allSettled([
    fetch('http://localhost:5000/api/streams').then(r => r.ok ? r.json() : []),
    fetch('http://localhost:8080/api/events-all').then(r => r.ok ? r.json() : { success: false })
  ])

  // 视频流
  if (streamsRes.status === 'fulfilled') {
    const data = streamsRes.value
    streams.value = Array.isArray(data) ? data : []
    stats.value.activeStreams = streams.value.filter(s => s.status === 'running').length
  }

  // 事件
  if (eventsRes.status === 'fulfilled' && eventsRes.value.success) {
    const allEvents = eventsRes.value.events || []
    recentEvents.value = allEvents.slice(0, 5)

    const todayStart = new Date()
    todayStart.setHours(0, 0, 0, 0)
    const todayCutoff = todayStart.getTime() / 1000

    stats.value.todayEvents = allEvents.filter(e => e.timestamp >= todayCutoff).length
    stats.value.fireAlerts = allEvents.filter(e =>
      e.timestamp >= todayCutoff && e.event_type === 'smoke_flame'
    ).length
  }
}

onMounted(() => {
  fetchAll()
  refreshTimer = setInterval(fetchAll, 5000)
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
})
</script>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 统计卡片 */
.stat-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.stat-card {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.stat-icon--blue {
  background: rgba(59, 130, 246, 0.15);
  color: var(--color-accent);
}

.stat-icon--green {
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success);
}

.stat-icon--red {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger);
}

.stat-info {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  line-height: 1.2;
}

.stat-label {
  font-size: 13px;
  color: var(--color-text-secondary);
}

/* 双栏面板 */
.panel-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.panel {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 20px;
  border-bottom: 1px solid var(--color-border);
}

.panel-header h3 {
  font-size: 15px;
  font-weight: 600;
}

.panel-link {
  font-size: 13px;
  color: var(--color-accent);
}

.panel-link:hover {
  text-decoration: underline;
}

.panel-body {
  padding: 8px 0;
}

.panel-empty {
  text-align: center;
  padding: 30px;
  color: var(--color-text-muted);
  font-size: 13px;
}

/* 事件行 */
.event-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 20px;
  transition: background-color 0.15s;
}

.event-row:hover {
  background: var(--color-bg-tertiary);
}

.event-type-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dot-smoke_flame { background: var(--color-danger); }
.dot-parking_violation { background: var(--color-accent); }
.dot-common_space_utilization { background: var(--color-success); }

.event-type-text {
  font-size: 13px;
  font-weight: 500;
  min-width: 60px;
}

.event-camera {
  font-size: 12px;
  color: var(--color-text-muted);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.event-time {
  font-size: 12px;
  color: var(--color-text-muted);
  white-space: nowrap;
}

/* 视频流行 */
.stream-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 20px;
  transition: background-color 0.15s;
}

.stream-row:hover {
  background: var(--color-bg-tertiary);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dot-status-running { background: var(--color-success); }
.dot-status-connecting { background: var(--color-warning); }
.dot-status-stopped { background: var(--color-text-muted); }
.dot-status-error { background: var(--color-danger); }

.stream-id {
  font-size: 13px;
  font-weight: 500;
  min-width: 80px;
}

.stream-tasks {
  flex: 1;
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.mini-tag {
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 10px;
  background: rgba(59, 130, 246, 0.1);
  color: var(--color-accent);
}

.stream-status {
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.text-running { color: var(--color-success); }
.text-connecting { color: var(--color-warning); }
.text-stopped { color: var(--color-text-muted); }
.text-error { color: var(--color-danger); }

/* 流性能指标行 */
.stream-metrics-row {
  display: flex;
  gap: 12px;
  padding: 2px 20px 6px 38px;
  flex-wrap: wrap;
}

.sm-item {
  font-size: 11px;
  color: var(--color-text-muted);
  font-family: "Cascadia Code", "JetBrains Mono", monospace;
}

.sm-item b {
  color: var(--color-text-primary);
  font-weight: 600;
}

.stream-bottleneck-row {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 0 20px 6px 38px;
}

.bottleneck-text {
  font-size: 11px;
  color: var(--color-danger);
  font-weight: 500;
}
</style>
