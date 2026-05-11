<template>
  <div class="dashboard">
    <!-- Stat cards -->
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-icon stat-icon--blue">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
          </svg>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ formatCardNumber(stats.activeStreams) }}</span>
          <span class="stat-label">Active Streams</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon stat-icon--green">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
          </svg>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ formatCardNumber(stats.todayEvents) }}</span>
          <span class="stat-label">Today's Events</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon stat-icon--red">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2c-5.33 4.55-8 8.48-8 11.8 0 4.98 3.8 8.2 8 8.2s8-3.22 8-8.2c0-3.32-2.67-7.25-8-11.8z"/>
          </svg>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ formatCardNumber(stats.fireAlerts) }}</span>
          <span class="stat-label">Fire Alerts</span>
        </div>
      </div>
    </div>

    <div class="stat-row stat-row--secondary">
      <div class="stat-card">
        <div class="stat-icon stat-icon--amber">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 19h16" />
            <path d="M6 16V8" />
            <path d="M12 16V5" />
            <path d="M18 16v-3" />
          </svg>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ formatCardNumber(stats.totalEvents) }}</span>
          <span class="stat-label">Total Events</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon stat-icon--teal">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 21s7-4.35 7-10a7 7 0 10-14 0c0 5.65 7 10 7 10z" />
            <circle cx="12" cy="11" r="2.5" />
          </svg>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ formatCardNumber(stats.commonSpaceEvents) }}</span>
          <span class="stat-label">Space Events</span>
        </div>
      </div>
    </div>

    <!-- RTSP runtime overview -->
    <div class="panel runtime-panel">
      <div class="panel-header">
        <h3>RTSP Runtime</h3>
        <span class="panel-subtle">{{ streams.length }} streams monitored</span>
      </div>
      <div class="runtime-body">
        <div v-if="streams.length === 0" class="panel-empty">No RTSP streams</div>
        <div v-for="s in streams" :key="s.stream_id" class="runtime-row">
          <div class="runtime-main">
            <span :class="['status-dot', 'dot-status-' + s.status]"></span>
            <div class="runtime-identity">
              <span class="runtime-name">{{ s.camera_id || s.stream_id }}</span>
              <span class="runtime-url" :title="s.url">{{ s.url }}</span>
            </div>
            <span :class="['runtime-state', 'text-' + s.status]">{{ statusLabel(s.status) }}</span>
          </div>
          <div class="runtime-values">
            <div class="runtime-metric">
              <span class="runtime-label">Uptime</span>
              <b>{{ durationLabel(secondsSince(s.created_at)) }}</b>
            </div>
            <div class="runtime-metric">
              <span class="runtime-label">Connected</span>
              <b>{{ durationLabel(streamConnectedSeconds(s)) }}</b>
            </div>
            <div class="runtime-metric">
              <span class="runtime-label">Last Frame</span>
              <b :class="{ 'text-error': isFrameStale(s) }">{{ frameAgeLabel(s) }}</b>
            </div>
            <div class="runtime-metric">
              <span class="runtime-label">Capture</span>
              <b>{{ fmtF(s.metrics?.capture?.capture_fps_10s) }} fps</b>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Two-column panels -->
    <div class="panel-row">
      <!-- Recent Events -->
      <div class="panel">
        <div class="panel-header">
          <h3>Recent Events</h3>
          <router-link to="/events" class="panel-link">View All</router-link>
        </div>
        <div class="panel-body">
          <div v-if="recentEvents.length === 0" class="panel-empty">No events</div>
          <div v-for="evt in recentEvents" :key="evt._id" class="event-row">
            <span class="event-type-dot" :class="'dot-' + evt.event_type"></span>
            <span class="event-type-text">{{ typeLabel(evt.event_type) }}</span>
            <span class="event-camera">{{ evt.camera_id }}</span>
            <span v-if="evt.location" class="event-location">{{ evt.location }}</span>
            <span class="event-time">{{ formatTime(evt.timestamp) }}</span>
          </div>
        </div>
      </div>

      <!-- Stream Status -->
      <div class="panel">
        <div class="panel-header">
          <h3>Stream Status</h3>
          <router-link to="/streams" class="panel-link">Manage</router-link>
        </div>
        <div class="panel-body">
          <div v-if="streams.length === 0" class="panel-empty">No streams</div>
          <template v-for="s in streams" :key="s.stream_id">
            <div class="stream-row">
              <span :class="['status-dot', 'dot-status-' + s.status]"></span>
              <span class="stream-id">{{ s.stream_id }}</span>
              <span class="stream-tasks">
                <span v-for="t in s.tasks" :key="t" class="mini-tag">{{ taskLabel(t) }}</span>
              </span>
              <span class="stream-uptime">{{ durationLabel(secondsSince(s.created_at)) }}</span>
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

    <div class="panel">
      <div class="panel-header">
        <h3>Latest Space Reports</h3>
        <router-link to="/reports" class="panel-link">Open Reports</router-link>
      </div>
      <div class="panel-body">
        <div v-if="recentReports.length === 0" class="panel-empty">No saved reports</div>
        <div v-for="report in recentReports" :key="report._id" class="report-row">
          <div class="report-row-head">
            <span class="report-kind" :class="'report-kind-' + report.report_kind">{{ reportKindLabel(report.report_kind) }}</span>
            <span class="report-stream">{{ report.report_key?.stream_id || report.report_key?.camera_id || '-' }}</span>
            <span class="report-time">{{ formatReportTime(report.generated_at_ts) }}</span>
          </div>
          <div class="report-row-meta">
            <span>{{ report.summary_source || '-' }}</span>
            <span>{{ report.report_key?.location || report.report_key?.url || '-' }}</span>
            <span>{{ report.window?.event_count ?? 0 }} events</span>
            <span>{{ report.stats?.dominant_occupancy || 'unknown' }}</span>
          </div>
          <p class="report-row-summary">{{ report.display_summary || report.narrative || '-' }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const stats = ref({
  activeStreams: 0,
  todayEvents: 0,
  fireAlerts: 0,
  totalEvents: 0,
  commonSpaceEvents: 0
})
const recentEvents = ref([])
const streams = ref([])
const recentReports = ref([])
const nowSec = ref(Date.now() / 1000)

let refreshTimer = null
let runtimeTimer = null

function typeLabel(t) {
  const map = { smoke_flame: 'Smoke/Fire', parking_violation: 'Parking Violation', common_space_utilization: 'Space Analysis' }
  return map[t] || t
}

function taskLabel(t) {
  const map = { parking_violation: 'Parking', smoke_flame: 'Smoke', common_space: 'Space' }
  return map[t] || t
}

function statusLabel(s) {
  const map = { running: 'Running', connecting: 'Connecting', stopped: 'Stopped', error: 'Error' }
  return map[s] || s
}

function reportKindLabel(kind) {
  return kind === 'llm' ? 'LLM' : 'Rule'
}

function formatTime(ts) {
  const d = new Date(ts * 1000)
  return d.toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function formatReportTime(ts) {
  if (!ts) return '-'
  return formatTime(ts)
}

function fmtF(v) {
  if (v == null) return '0.0'
  return Number(v).toFixed(1)
}

function fmtN(v) {
  if (v == null) return '0'
  return String(Number(v))
}

function formatCardNumber(value) {
  return Number(value || 0).toLocaleString('en-US')
}

function toEpochSeconds(value) {
  const n = Number(value)
  if (!Number.isFinite(n) || n <= 0) return null
  return n > 1000000000000 ? n / 1000 : n
}

function secondsSince(value) {
  const ts = toEpochSeconds(value)
  if (ts == null) return null
  return Math.max(0, nowSec.value - ts)
}

function durationLabel(seconds) {
  if (seconds == null) return '-'
  const total = Math.floor(seconds)
  const days = Math.floor(total / 86400)
  const hours = Math.floor((total % 86400) / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const hh = String(hours).padStart(2, '0')
  const mm = String(minutes).padStart(2, '0')
  const ss = String(secs).padStart(2, '0')
  return days > 0 ? `${days}d ${hh}:${mm}:${ss}` : `${hh}:${mm}:${ss}`
}

function streamConnectedSeconds(stream) {
  if (stream.status !== 'running') return null
  return secondsSince(stream.last_connected_at)
}

function frameAgeLabel(stream) {
  const age = secondsSince(stream.last_frame_at)
  if (age == null) return '-'
  return `${durationLabel(age)} ago`
}

function isFrameStale(stream) {
  const age = secondsSince(stream.last_frame_at)
  return stream.status === 'running' && age != null && age > 10
}

function findTypeCount(items, targetType) {
  if (!Array.isArray(items)) return 0
  const match = items.find(item => item?._id === targetType)
  return Number(match?.count || 0)
}

async function fetchAll() {
  const [streamsRes, eventsRes, reportsRes, statsRes] = await Promise.allSettled([
    fetch('http://localhost:5000/api/streams').then(r => r.ok ? r.json() : []),
    fetch('http://localhost:5000/api/events-all').then(r => r.ok ? r.json() : { success: false }),
    fetch('http://localhost:5000/api/reports/common-space/history?limit=4').then(r => r.ok ? r.json() : { success: false }),
    fetch('http://localhost:5000/api/events/stats').then(r => r.ok ? r.json() : null)
  ])

  if (streamsRes.status === 'fulfilled') {
    const data = streamsRes.value
    streams.value = Array.isArray(data) ? data : []
    stats.value.activeStreams = streams.value.filter(s => s.status === 'running').length
  }

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

  if (reportsRes.status === 'fulfilled' && reportsRes.value.success) {
    recentReports.value = reportsRes.value.items || []
  }

  if (statsRes.status === 'fulfilled' && statsRes.value) {
    const eventStats = statsRes.value
    stats.value.totalEvents = Number(eventStats.total_events || 0)
    stats.value.commonSpaceEvents = findTypeCount(
      eventStats.by_type,
      'common_space_utilization'
    )
  }
}

onMounted(() => {
  nowSec.value = Date.now() / 1000
  fetchAll()
  refreshTimer = setInterval(fetchAll, 5000)
  runtimeTimer = setInterval(() => {
    nowSec.value = Date.now() / 1000
  }, 1000)
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
  if (runtimeTimer) clearInterval(runtimeTimer)
})
</script>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Stat cards */
.stat-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.stat-row--secondary {
  grid-template-columns: repeat(2, 1fr);
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

.stat-icon--amber {
  background: rgba(245, 158, 11, 0.15);
  color: var(--color-warning);
}

.stat-icon--teal {
  background: rgba(20, 184, 166, 0.15);
  color: #0f766e;
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

/* Two-column panels */
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

.panel-subtle {
  font-size: 12px;
  color: var(--color-text-muted);
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

/* Event row */
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

.event-location {
  font-size: 11px;
  color: var(--color-success);
  background: rgba(34, 197, 94, 0.1);
  padding: 1px 6px;
  border-radius: var(--radius-sm);
  white-space: nowrap;
}

.event-time {
  font-size: 12px;
  color: var(--color-text-muted);
  white-space: nowrap;
}

/* Stream row */
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

.stream-uptime {
  font-family: "Cascadia Code", "JetBrains Mono", monospace;
  font-size: 12px;
  color: var(--color-text-secondary);
  white-space: nowrap;
}

.text-running { color: var(--color-success); }
.text-connecting { color: var(--color-warning); }
.text-stopped { color: var(--color-text-muted); }
.text-error { color: var(--color-danger); }

.runtime-panel {
  overflow: hidden;
}

.runtime-body {
  display: flex;
  flex-direction: column;
}

.runtime-row {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) minmax(420px, 1.2fr);
  gap: 16px;
  align-items: center;
  padding: 12px 20px;
  border-bottom: 1px solid var(--color-border);
}

.runtime-row:last-child {
  border-bottom: 0;
}

.runtime-main {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.runtime-identity {
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.runtime-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.runtime-url {
  font-family: "Cascadia Code", "JetBrains Mono", monospace;
  font-size: 11px;
  color: var(--color-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.runtime-state {
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.runtime-values {
  display: grid;
  grid-template-columns: repeat(4, minmax(92px, 1fr));
  gap: 10px;
}

.runtime-metric {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 8px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
}

.runtime-label {
  font-size: 11px;
  color: var(--color-text-muted);
}

.runtime-metric b {
  font-family: "Cascadia Code", "JetBrains Mono", monospace;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Stream metrics row */
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

.report-row {
  padding: 12px 20px;
  border-bottom: 1px solid var(--color-border);
}

.report-row:last-child {
  border-bottom: 0;
}

.report-row-head,
.report-row-meta {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.report-row-head {
  margin-bottom: 6px;
}

.report-kind {
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.03em;
}

.report-kind-rule {
  background: rgba(59, 130, 246, 0.12);
  color: var(--color-accent);
}

.report-kind-llm {
  background: rgba(34, 197, 94, 0.12);
  color: var(--color-success);
}

.report-stream {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.report-time,
.report-row-meta {
  font-size: 12px;
  color: var(--color-text-muted);
}

.report-row-summary {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text-secondary);
}

@media (max-width: 1180px) {
  .runtime-row {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 760px) {
  .runtime-values {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .stream-row {
    flex-wrap: wrap;
  }
}
</style>
