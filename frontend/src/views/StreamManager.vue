<template>
  <div class="stream-manager">
    <!-- Add Stream Form -->
    <div class="form-card">
      <h3 class="form-title">Add Stream</h3>
      <div class="form-body">
        <div class="form-row">
          <label class="form-label">RTSP URL</label>
          <input
            v-model="newUrl"
            type="text"
            placeholder="e.g. rtsp://192.168.1.100:554/stream"
            class="form-input"
          />
        </div>
        <div class="form-row">
          <label class="form-label">Camera ID</label>
          <input
            v-model="newCameraId"
            type="text"
            placeholder="e.g. east_gate_01"
            class="form-input"
          />
          <span class="form-hint">Required when "Parking Violation" is selected, used to match no-parking zones.</span>
        </div>
        <div class="form-row">
          <label class="form-label">Latitude / Longitude</label>
          <input
            v-model="newLatLng"
            type="text"
            placeholder="e.g. 31.2304, 121.4737"
            class="form-input"
          />
        </div>
        <div class="form-row">
          <label class="form-label">Camera Location</label>
          <input
            v-model="newLocation"
            type="text"
            placeholder="e.g. East Gate Entrance"
            class="form-input"
          />
        </div>
        <div class="form-row">
          <label class="form-label">Area Code</label>
          <input
            v-model="newAreaCode"
            type="text"
            placeholder="e.g. east_district"
            class="form-input"
          />
          <span class="form-hint">Used by MUBS for dispatch rule matching.</span>
        </div>
        <div class="form-row">
          <label class="form-label">Group</label>
          <input
            v-model="newGroup"
            type="text"
            placeholder="e.g. fire_team"
            class="form-input"
          />
          <span class="form-hint">Used by MUBS for team assignment.</span>
        </div>
        <div class="form-row">
          <label class="form-label">Analysis Tasks</label>
          <div class="task-checkboxes">
            <label class="checkbox-item" v-for="t in allTasks" :key="t.value">
              <input type="checkbox" :value="t.value" v-model="newTasks" />
              <span>{{ t.label }}</span>
            </label>
          </div>
        </div>
        <div class="form-actions">
          <button class="btn-accent" @click="addStream" :disabled="!canAddStream">Add Stream</button>
          <span v-if="message" :class="['msg', messageType]">{{ message }}</span>
        </div>
      </div>
    </div>

    <!-- Stream List -->
    <div class="list-card">
      <h3 class="form-title">Stream List</h3>
      <div v-if="streams.length > 0" class="stream-list">
        <div v-for="s in streams" :key="s.stream_id" class="stream-item">
          <!-- Stream basic info row -->
          <div class="stream-header">
            <div class="stream-basic">
              <span class="stream-id">{{ s.stream_id }}</span>
              <code class="camera-code">{{ s.camera_id || '-' }}</code>
              <span v-if="s.location" class="location-tag" :title="s.lat_lng">{{ s.location }}</span>
              <span v-if="s.area_code" class="area-tag">{{ s.area_code }}</span>
              <span v-if="s.group" class="group-tag">{{ s.group }}</span>
              <span class="stream-url" :title="s.url">{{ s.url }}</span>
            </div>
            <div class="stream-right">
              <span :class="['status-tag', 'status-' + s.status]">{{ statusLabel(s.status) }}</span>
              <button class="btn-sm btn-ghost" @click="toggleMetrics(s.stream_id)" :title="expanded.has(s.stream_id) ? 'Collapse' : 'Expand'">
                <svg :class="{ 'chevron-open': expanded.has(s.stream_id) }" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
              </button>
            </div>
          </div>

          <!-- Task tags + actions -->
          <div class="stream-tags-row">
            <div class="stream-tags">
              <template v-if="editing === s.stream_id">
                <label v-for="t in allTasks" :key="t.value" class="edit-task-label">
                  <input type="checkbox" :value="t.value" v-model="editTasks" />
                  {{ t.label }}
                </label>
              </template>
              <template v-else>
                <span v-for="t in s.tasks" :key="t" class="task-tag">{{ taskLabel(t) }}</span>
              </template>
            </div>
            <div class="stream-actions">
              <template v-if="editing === s.stream_id">
                <button class="btn-sm btn-success" @click="saveTasks(s.stream_id)">Save</button>
                <button class="btn-sm btn-muted" @click="editing = null">Cancel</button>
              </template>
              <template v-else>
                <button class="btn-sm btn-preview" @click="togglePreview(s.stream_id)">
                  {{ previewing.has(s.stream_id) ? 'Hide' : 'Preview' }}
                </button>
                <button class="btn-sm btn-outline" @click="startEdit(s)">Edit</button>
                <button class="btn-sm btn-danger" @click="removeStream(s.stream_id)">Delete</button>
              </template>
            </div>
          </div>

          <!-- Video preview (MJPEG) -->
          <div v-if="previewing.has(s.stream_id)" class="preview-panel">
            <img
              :src="API + '/api/streams/' + s.stream_id + '/video_feed'"
              class="preview-img"
              alt="Live preview"
            />
          </div>

          <!-- Metrics panel (expandable) -->
          <div v-if="expanded.has(s.stream_id) && s.metrics" class="metrics-panel">
            <!-- Capture metrics -->
            <div v-if="s.metrics.capture" class="metrics-section">
              <h4 class="metrics-section-title">Capture</h4>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">capture</span>
                  <span class="metric-value">{{ f(s.metrics.capture.capture_fps_10s) }} fps</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">emit</span>
                  <span class="metric-value">{{ f(s.metrics.capture.emit_fps_10s) }} fps</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">batch</span>
                  <span class="metric-value">{{ f(s.metrics.capture.batch_rate_10s) }}/s</span>
                </div>
              </div>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">read</span>
                  <span class="metric-value">{{ n(s.metrics.capture.frames_read_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">emitted</span>
                  <span class="metric-value">{{ n(s.metrics.capture.frames_emitted_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">batches</span>
                  <span class="metric-value">{{ n(s.metrics.capture.batches_emitted_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">last batch</span>
                  <span class="metric-value">{{ n(s.metrics.capture.last_batch_size) }}</span>
                </div>
              </div>
            </div>

            <!-- Executor metrics -->
            <div v-if="s.metrics.executor" class="metrics-section">
              <h4 class="metrics-section-title">Executor</h4>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">framequeue</span>
                  <span class="metric-value">{{ n(s.metrics.executor.queue_size) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">inflight</span>
                  <span class="metric-value">{{ n(s.metrics.executor.inflight_tasks) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">submit</span>
                  <span class="metric-value">{{ f(s.metrics.executor.dispatch_fps_10s) }}/s</span>
                </div>
              </div>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">submitted</span>
                  <span class="metric-value">{{ n(s.metrics.executor.submitted_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">completed</span>
                  <span class="metric-value">{{ n(s.metrics.executor.completed_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">failed</span>
                  <span class="metric-value" :class="{ 'val-danger': s.metrics.executor.failed_total > 0 }">{{ n(s.metrics.executor.failed_total) }}</span>
                </div>
              </div>
            </div>

            <!-- Smoke/Fire Detection metrics -->
            <div v-if="s.metrics.tasks?.smoke_flame" class="metrics-section">
              <h4 class="metrics-section-title">Smoke/Fire Detection</h4>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">yolo</span>
                  <span class="metric-value">{{ f(s.metrics.tasks.smoke_flame.yolo_latency?.avg_ms) }} ms</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">qwen</span>
                  <span class="metric-value">{{ f(s.metrics.tasks.smoke_flame.qwen_latency?.avg_ms) }} ms</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">verifyQ</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.smoke_flame.verification_queue_size) }}</span>
                </div>
              </div>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">recv</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.smoke_flame.frames_received_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">submitted</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.smoke_flame.frames_submitted_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">qwen</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.smoke_flame.qwen_requests_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">saved</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.smoke_flame.events_saved_total) }}</span>
                </div>
              </div>
            </div>

            <!-- Parking Violation metrics -->
            <div v-if="s.metrics.tasks?.parking_violation" class="metrics-section">
              <h4 class="metrics-section-title">Parking Violation</h4>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">yolo</span>
                  <span class="metric-value">{{ f(s.metrics.tasks.parking_violation.yolo_latency?.avg_ms) }} ms</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">input fps</span>
                  <span class="metric-value">{{ f(s.metrics.tasks.parking_violation.input_fps_10s) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">detections</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.parking_violation.frames_with_detection_total) }}</span>
                </div>
              </div>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">frames</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.parking_violation.frames_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">violations</span>
                  <span class="metric-value" :class="{ 'val-danger': s.metrics.tasks.parking_violation.frames_with_violation_total > 0 }">{{ n(s.metrics.tasks.parking_violation.frames_with_violation_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">saved</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.parking_violation.events_saved_total) }}</span>
                </div>
              </div>
            </div>

            <!-- Common Space Analysis metrics -->
            <div v-if="s.metrics.tasks?.common_space" class="metrics-section">
              <h4 class="metrics-section-title">Common Space Analysis</h4>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">analysis</span>
                  <span class="metric-value">{{ f(s.metrics.tasks.common_space.analysis_latency?.avg_ms) }} ms</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">queue</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.common_space.analysis_queue_size) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">inflight</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.common_space.analysis_inflight) }}</span>
                </div>
              </div>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">recv</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.common_space.frames_received_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">sampled</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.common_space.frames_sampled_total) }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">saved</span>
                  <span class="metric-value">{{ n(s.metrics.tasks.common_space.events_saved_total) }}</span>
                </div>
              </div>
            </div>

            <!-- Bottleneck hints -->
            <div v-if="s.bottleneck_hints?.length" class="metrics-section">
              <h4 class="metrics-section-title warn-title">Bottleneck Hints</h4>
              <div class="bottleneck-list">
                <span v-for="(hint, i) in s.bottleneck_hints" :key="i" class="bottleneck-item">{{ hint }}</span>
              </div>
            </div>
          </div>

          <!-- No metrics data -->
          <div v-else-if="expanded.has(s.stream_id) && !s.metrics" class="metrics-panel">
            <div class="metrics-empty">No metrics data available</div>
          </div>
        </div>
      </div>
      <div v-else class="empty-state">No active streams. Add one above.</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const API = 'http://localhost:5000'

const streams = ref([])
const newUrl = ref('')
const newCameraId = ref('')
const newLatLng = ref('')
const newLocation = ref('')
const newAreaCode = ref('')
const newGroup = ref('')
const newTasks = ref([])
const message = ref('')
const messageType = ref('success')
const editing = ref(null)
const editTasks = ref([])
const expanded = ref(new Set())
const previewing = ref(new Set())

const allTasks = [
  { value: 'parking_violation', label: 'Parking Violation' },
  { value: 'smoke_flame', label: 'Smoke/Fire' },
  { value: 'common_space', label: 'Common Space' }
]

const requiresCameraId = computed(() => newTasks.value.includes('parking_violation'))
const canAddStream = computed(() => {
  if (!newUrl.value.trim()) return false
  if (newTasks.value.length === 0) return false
  if (requiresCameraId.value && !newCameraId.value.trim()) return false
  return true
})

let pollTimer = null

function f(v) {
  if (v == null) return '0.0'
  return Number(v).toFixed(1)
}

function n(v) {
  if (v == null) return '0'
  return Number(v).toLocaleString()
}

function taskLabel(t) {
  const map = {
    parking_violation: 'Parking Violation',
    smoke_flame: 'Smoke/Fire',
    common_space: 'Common Space'
  }
  return map[t] || t
}

function statusLabel(s) {
  const map = {
    running: 'Running',
    connecting: 'Connecting',
    stopped: 'Stopped',
    error: 'Error'
  }
  return map[s] || s
}

function toggleMetrics(id) {
  if (expanded.value.has(id)) {
    expanded.value.delete(id)
  } else {
    expanded.value.add(id)
  }
}

function togglePreview(id) {
  if (previewing.value.has(id)) {
    previewing.value.delete(id)
  } else {
    previewing.value.add(id)
  }
}

function showMsg(text, type = 'success') {
  message.value = text
  messageType.value = type
  setTimeout(() => { message.value = '' }, 3000)
}

async function fetchStreams() {
  try {
    const res = await fetch(`${API}/api/streams`)
    if (res.ok) streams.value = await res.json()
  } catch {
    // silent
  }
}

async function addStream() {
  const url = newUrl.value.trim()
  const cameraId = newCameraId.value.trim()
  if (!url || newTasks.value.length === 0) return
  if (requiresCameraId.value && !cameraId) {
    showMsg('Camera ID is required for Parking Violation', 'error')
    return
  }

  try {
    const res = await fetch(`${API}/api/streams`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        tasks: newTasks.value,
        camera_id: cameraId,
        lat_lng: newLatLng.value.trim(),
        location: newLocation.value.trim(),
        area_code: newAreaCode.value.trim(),
        group: newGroup.value.trim()
      })
    })
    const data = await res.json()
    if (res.ok) {
      showMsg(`Added ${data.stream_id}`)
      newUrl.value = ''
      newCameraId.value = ''
      newLatLng.value = ''
      newLocation.value = ''
      newAreaCode.value = ''
      newGroup.value = ''
      newTasks.value = []
      await fetchStreams()
    } else {
      showMsg(data.error || 'Failed to add', 'error')
    }
  } catch {
    showMsg('Network error', 'error')
  }
}

async function removeStream(id) {
  try {
    const res = await fetch(`${API}/api/streams/${id}`, { method: 'DELETE' })
    if (res.ok) {
      showMsg(`Removed ${id}`)
      await fetchStreams()
    } else {
      showMsg('Failed to remove', 'error')
    }
  } catch {
    showMsg('Network error', 'error')
  }
}

function startEdit(s) {
  editing.value = s.stream_id
  editTasks.value = [...s.tasks]
}

async function saveTasks(id) {
  if (editTasks.value.length === 0) {
    showMsg('Select at least one task', 'error')
    return
  }
  try {
    const res = await fetch(`${API}/api/streams/${id}/tasks`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tasks: editTasks.value })
    })
    if (res.ok) {
      showMsg(`Updated ${id}`)
      editing.value = null
      await fetchStreams()
    } else {
      showMsg('Failed to update', 'error')
    }
  } catch {
    showMsg('Network error', 'error')
  }
}

onMounted(() => {
  fetchStreams()
  pollTimer = setInterval(fetchStreams, 5000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<style scoped>
.stream-manager {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.form-card,
.list-card {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.form-title {
  font-size: 15px;
  font-weight: 600;
  padding: 14px 20px;
  border-bottom: 1px solid var(--color-border);
  color: var(--color-text-primary);
}

.form-body {
  padding: 20px;
}

.form-row {
  margin-bottom: 16px;
}

.form-label {
  display: block;
  font-size: 13px;
  color: var(--color-text-secondary);
  margin-bottom: 6px;
}

.form-input {
  width: 100%;
  padding: 8px 12px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text-primary);
  font-size: 14px;
}

.form-input:focus {
  border-color: var(--color-accent);
  outline: none;
}

.form-hint {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.task-checkboxes {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.checkbox-item {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  color: var(--color-text-primary);
  font-size: 14px;
}

.form-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-top: 4px;
}

.btn-accent {
  background: var(--color-accent);
  color: #fff;
  padding: 8px 20px;
  border-radius: var(--radius-sm);
}

.btn-accent:hover:not(:disabled) {
  background: var(--color-accent-hover);
}

.msg {
  font-size: 13px;
}

.msg.success {
  color: var(--color-success);
}

.msg.error {
  color: var(--color-danger);
}

/* Stream list */
.stream-list {
  display: flex;
  flex-direction: column;
}

.stream-item {
  border-bottom: 1px solid var(--color-border);
  padding: 14px 20px;
}

.stream-item:last-child {
  border-bottom: none;
}

.stream-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.stream-basic {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  flex: 1;
}

.stream-id {
  font-weight: 600;
  font-size: 13px;
  white-space: nowrap;
}

.camera-code {
  font-size: 12px;
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  white-space: nowrap;
}

.location-tag {
  font-size: 12px;
  color: var(--color-success);
  background: rgba(34, 197, 94, 0.1);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  white-space: nowrap;
}

.area-tag {
  font-size: 12px;
  color: var(--color-accent);
  background: rgba(59, 130, 246, 0.1);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  white-space: nowrap;
}

.group-tag {
  font-size: 12px;
  color: var(--color-warning);
  background: rgba(245, 158, 11, 0.1);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  white-space: nowrap;
}

.stream-url {
  font-family: monospace;
  font-size: 12px;
  color: var(--color-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}

.stream-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.status-tag {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.status-running {
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success);
}

.status-connecting {
  background: rgba(245, 158, 11, 0.15);
  color: var(--color-warning);
}

.status-stopped {
  background: rgba(100, 116, 139, 0.15);
  color: var(--color-text-muted);
}

.status-error {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger);
}

.btn-ghost {
  background: transparent;
  padding: 4px;
  display: flex;
  align-items: center;
  color: var(--color-text-muted);
  border-radius: var(--radius-sm);
}

.btn-ghost:hover {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

.btn-ghost svg {
  transition: transform 0.2s;
}

.chevron-open {
  transform: rotate(180deg);
}

/* Task tag row */
.stream-tags-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 8px;
  gap: 12px;
}

.stream-tags {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.task-tag {
  display: inline-block;
  background: rgba(59, 130, 246, 0.15);
  color: var(--color-accent);
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
}

.edit-task-label {
  font-size: 13px;
  margin-right: 12px;
  color: var(--color-text-primary);
  cursor: pointer;
}

.stream-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.btn-sm {
  padding: 4px 12px;
  font-size: 12px;
  border-radius: var(--radius-sm);
}

.btn-outline {
  background: transparent;
  border: 1px solid var(--color-accent);
  color: var(--color-accent);
}

.btn-outline:hover {
  background: rgba(59, 130, 246, 0.1);
}

.btn-danger {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger);
}

.btn-danger:hover {
  background: rgba(239, 68, 68, 0.25);
}

.btn-success {
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success);
}

.btn-success:hover {
  background: rgba(34, 197, 94, 0.25);
}

.btn-muted {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
}

/* Preview panel */
.preview-panel {
  margin-top: 10px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.preview-img {
  display: block;
  width: 100%;
  max-height: 400px;
  object-fit: contain;
  background: #000;
}

.btn-preview {
  background: rgba(139, 92, 246, 0.15);
  color: #a78bfa;
}

.btn-preview:hover {
  background: rgba(139, 92, 246, 0.25);
}

/* Metrics panel */
.metrics-panel {
  margin-top: 12px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 12px 16px;
}

.metrics-section {
  margin-bottom: 12px;
}

.metrics-section:last-child {
  margin-bottom: 0;
}

.metrics-section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--color-border);
}

.warn-title {
  color: var(--color-warning);
  border-bottom-color: rgba(245, 158, 11, 0.3);
}

.metrics-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  margin-bottom: 6px;
  font-family: "Cascadia Code", "JetBrains Mono", "Fira Code", monospace;
}

.metric-item {
  display: flex;
  align-items: baseline;
  gap: 6px;
  min-width: 140px;
}

.metric-label {
  font-size: 11px;
  color: var(--color-text-muted);
  white-space: nowrap;
}

.metric-value {
  font-size: 12px;
  color: var(--color-text-primary);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.val-danger {
  color: var(--color-danger);
}

.bottleneck-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.bottleneck-item {
  font-size: 12px;
  color: var(--color-warning);
  padding: 4px 8px;
  background: rgba(245, 158, 11, 0.08);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--color-warning);
}

.metrics-empty {
  text-align: center;
  padding: 16px;
  color: var(--color-text-muted);
  font-size: 13px;
}

.empty-state {
  text-align: center;
  padding: 40px;
  color: var(--color-text-muted);
}
</style>
