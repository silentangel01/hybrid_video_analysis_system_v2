<template>
  <div class="event-list">
    <!-- Filter bar -->
    <div class="filter-bar">
      <div class="filter-group">
        <label class="filter-label">Event Type</label>
        <select v-model="filterType" class="filter-select">
          <option value="">All Types</option>
          <option value="smoke_flame">Smoke/Fire Alert</option>
          <option value="parking_violation">Parking Violation</option>
          <option value="common_space_utilization">Space Analysis</option>
        </select>
      </div>
      <div class="filter-group">
        <label class="filter-label">Time Range</label>
        <div class="time-btns">
          <button
            v-for="opt in timeOptions"
            :key="opt.value"
            :class="['time-btn', { active: filterTime === opt.value }]"
            @click="filterTime = opt.value"
          >
            {{ opt.label }}
          </button>
        </div>
      </div>
      <button class="btn-refresh" @click="refresh" :disabled="loading">
        {{ loading ? 'Refreshing...' : 'Refresh' }}
      </button>
    </div>

    <!-- Error message -->
    <div v-if="error" class="error-bar">
      Failed to load: {{ error }}
      <button class="btn-retry" @click="fetchEvents">Retry</button>
    </div>

    <!-- Loading state -->
    <div v-else-if="loading && events.length === 0" class="state-msg">Loading events...</div>

    <!-- Empty state -->
    <div v-else-if="filteredEvents.length === 0" class="state-msg">No events found</div>

    <!-- Events grid -->
    <div v-else class="events-grid">
      <div
        v-for="event in paginatedEvents"
        :key="event._id"
        class="event-card"
        role="button"
        tabindex="0"
        @click="openEventDetail(event)"
        @keydown.enter="openEventDetail(event)"
        @keydown.space.prevent="openEventDetail(event)"
      >
        <div class="event-head">
          <span class="event-type-tag" :class="'type-' + event.event_type">{{ typeLabel(event.event_type) }}</span>
          <span class="confidence">{{ (event.confidence * 100).toFixed(1) }}%</span>
        </div>
        <div class="event-detail">
          <p><span class="detail-label">Source: </span>{{ event.camera_id }}</p>
          <p><span class="detail-label">Time: </span>{{ formatTimestamp(event.timestamp) }}</p>
          <p v-if="event.location"><span class="detail-label">Location: </span>{{ event.location }}</p>
          <p v-if="event.lat_lng"><span class="detail-label">Coordinates: </span>{{ event.lat_lng }}</p>
        </div>

        <div v-if="event.event_type === 'common_space_utilization' && hasCommonSpaceDetails(event)" class="event-detail">
          <p v-if="event.analysis_summary?.space_occupancy">
            <span class="detail-label">Occupancy: </span>{{ event.analysis_summary.space_occupancy }}
          </p>
          <p v-if="event.analysis_summary?.estimated_people_count != null">
            <span class="detail-label">People Count: </span>{{ event.analysis_summary.estimated_people_count }}
          </p>
          <p v-if="event.analysis_summary?.activity_types?.length">
            <span class="detail-label">Activities: </span>{{ formatActivityTypes(event.analysis_summary.activity_types) }}
          </p>
          <p v-if="event.analysis_summary?.safety_concerns !== undefined">
            <span class="detail-label">Safety: </span>{{ event.analysis_summary.safety_concerns ? 'Potential concern' : 'No visible concern' }}
          </p>
          <p v-if="event.analysis_summary?.scene_summary">
            <span class="detail-label">Summary: </span>{{ event.analysis_summary.scene_summary }}
          </p>
          <p v-if="event.analysis_summary?.occupancy_reason">
            <span class="detail-label">Reason: </span>{{ event.analysis_summary.occupancy_reason }}
          </p>
          <details v-if="event.analysis_result" class="analysis-details">
            <summary>Full Analysis</summary>
            <pre class="analysis-pre">{{ formatAnalysisResult(event.analysis_result) }}</pre>
          </details>
        </div>

        <div v-if="event.description" class="event-detail">
          <p><span class="detail-label">Description: </span>{{ event.description }}</p>
        </div>

        <div v-if="event.image_url" class="event-detail event-url-block">
          <p class="event-url-row">
            <span class="detail-label">Image URL: </span>
            <a
              :href="event.image_url"
              class="event-url-link"
              target="_blank"
              rel="noopener noreferrer"
              :title="event.image_url"
              @click.stop
            >
              {{ event.image_url }}
            </a>
          </p>
          <button class="copy-url-btn" @click.stop="copyText(event._id, event.image_url)">
            {{ copiedEventId === event._id ? 'Copied' : 'Copy URL' }}
          </button>
        </div>

        <div v-if="event.event_type === 'parking_violation'" class="violation-tag">Violation</div>

        <div v-if="event.image_url" class="image-preview">
          <img
            :src="event.image_url"
            :alt="typeLabel(event.event_type)"
            @click.stop="openImage(event.image_url)"
            @error="handleImageError"
            loading="lazy"
          />
        </div>
        <div v-else class="no-image">No screenshot</div>
      </div>
    </div>

    <!-- Pagination -->
    <div v-if="filteredEvents.length > itemsPerPage" class="pagination">
      <button class="page-btn" @click="currentPage--" :disabled="currentPage <= 1">Previous</button>
      <span class="page-info">Page {{ currentPage }} of {{ totalPages }}</span>
      <button class="page-btn" @click="currentPage++" :disabled="currentPage >= totalPages">Next</button>
    </div>

    <!-- Image modal -->
    <div v-if="fullImage" class="image-modal" @click="closeFullImage">
      <img :src="fullImage" alt="Full image" @click.stop />
    </div>

    <!-- Event detail modal -->
    <div v-if="selectedEvent" class="event-detail-modal" @click="closeEventDetail">
      <article class="event-detail-card" @click.stop>
        <header class="event-detail-header">
          <div>
            <span class="event-type-tag" :class="'type-' + selectedEvent.event_type">{{ typeLabel(selectedEvent.event_type) }}</span>
            <h3>{{ selectedEvent.description || 'Event Detail' }}</h3>
            <p class="event-id-line">Event ID: {{ eventId(selectedEvent) }}</p>
          </div>
          <button class="modal-close-btn" @click="closeEventDetail" aria-label="Close event detail">Close</button>
        </header>

        <div v-if="detailLoading" class="detail-loading">Loading detail...</div>
        <div v-if="detailError" class="detail-error">{{ detailError }}</div>

        <div v-if="selectedEvent.image_url" class="detail-image-wrap">
          <img
            :src="selectedEvent.image_url"
            :alt="typeLabel(selectedEvent.event_type)"
            @click="openImage(selectedEvent.image_url)"
            @error="handleImageError"
          />
        </div>

        <section class="detail-section">
          <div class="detail-section-head">
            <h4>All Attributes</h4>
            <button class="copy-url-btn" @click="copyText(eventId(selectedEvent), formatJsonValue(selectedEvent))">
              {{ copiedEventId === eventId(selectedEvent) ? 'Copied' : 'Copy JSON' }}
            </button>
          </div>
          <div class="attribute-list">
            <div v-for="[key, value] in eventDetailEntries" :key="key" class="attribute-row">
              <span class="attribute-key">{{ formatDetailKey(key) }}</span>
              <pre v-if="isComplexValue(value)" class="attribute-pre">{{ formatJsonValue(value) }}</pre>
              <span v-else class="attribute-value">{{ formatDetailValue(key, value) }}</span>
            </div>
          </div>
        </section>
      </article>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

const events = ref([])
const loading = ref(false)
const error = ref(null)
const fullImage = ref(null)
const copiedEventId = ref(null)
const selectedEvent = ref(null)
const detailLoading = ref(false)
const detailError = ref(null)
const currentPage = ref(1)
const itemsPerPage = 10

const filterType = ref('')
const filterTime = ref('')

const timeOptions = [
  { label: 'Last 1h', value: '1h' },
  { label: 'Last 24h', value: '24h' },
  { label: 'Last 7d', value: '7d' },
  { label: 'All', value: '' }
]

let refreshInterval = null

function typeLabel(t) {
  const map = {
    smoke_flame: 'Smoke/Fire Alert',
    parking_violation: 'Parking Violation',
    common_space_utilization: 'Space Analysis'
  }
  return map[t] || t
}

function hasCommonSpaceDetails(event) {
  return Boolean(
    event.analysis_summary?.space_occupancy ||
    event.analysis_summary?.estimated_people_count != null ||
    event.analysis_summary?.activity_types?.length ||
    event.analysis_summary?.scene_summary ||
    event.analysis_summary?.occupancy_reason ||
    event.analysis_result
  )
}

function formatActivityTypes(types) {
  if (!Array.isArray(types) || types.length === 0) return '-'
  return types
    .map(type => String(type).replace(/_/g, ' '))
    .join(', ')
}

function formatAnalysisResult(result) {
  if (!result) return ''
  if (typeof result === 'object') return formatAnalysisNarrative(result)

  let text = String(result).trim()
  if (!text) return ''

  if (text.startsWith('```')) {
    text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim()
  }

  if (text.startsWith('{') || text.startsWith('[')) {
    try {
      return formatAnalysisNarrative(JSON.parse(text))
    } catch {
      return text
    }
  }

  return text
}

function formatAnalysisNarrative(data) {
  if (!data || Array.isArray(data) || typeof data !== 'object') {
    return String(data || '')
  }

  const lines = []
  const people = data.estimated_people_count
  const occupancy = data.space_occupancy
  const activities = Array.isArray(data.activity_types) ? data.activity_types : []
  const safety = data.safety_concerns
  const summary = data.scene_summary
  const reason = data.occupancy_reason

  if (summary) lines.push(summary)
  if (people != null || occupancy) {
    const parts = []
    if (occupancy) parts.push(`${occupancy} occupancy`)
    if (people != null) parts.push(`about ${people} visible people`)
    if (parts.length) lines.push(`Assessment: ${parts.join(', ')}.`)
  }
  if (activities.length) {
    lines.push(`Activities: ${formatActivityTypes(activities)}.`)
  }
  if (typeof safety === 'boolean') {
    lines.push(safety ? 'Safety: potential concern detected.' : 'Safety: no visible concern detected.')
  }
  if (reason) {
    lines.push(`Reason: ${reason}`)
  }

  return lines.join('\n')
}

const filteredEvents = computed(() => {
  let list = events.value

  if (filterType.value) {
    list = list.filter(e => e.event_type === filterType.value)
  }

  if (filterTime.value) {
    const now = Date.now() / 1000
    const ranges = { '1h': 3600, '24h': 86400, '7d': 604800 }
    const cutoff = now - (ranges[filterTime.value] || 0)
    list = list.filter(e => e.timestamp >= cutoff)
  }

  return list
})

const totalPages = computed(() => Math.ceil(filteredEvents.value.length / itemsPerPage))

const paginatedEvents = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage
  return filteredEvents.value.slice(start, start + itemsPerPage)
})

const eventDetailEntries = computed(() => {
  if (!selectedEvent.value) return []
  return Object.entries(selectedEvent.value).sort(([a], [b]) => {
    const order = ['_id', 'event_id', 'event_type', 'camera_id', 'timestamp', 'frame_index', 'confidence', 'description', 'image_url']
    const ai = order.indexOf(a)
    const bi = order.indexOf(b)
    if (ai !== -1 || bi !== -1) return (ai === -1 ? order.length : ai) - (bi === -1 ? order.length : bi)
    return a.localeCompare(b)
  })
})

watch([filterType, filterTime], () => {
  currentPage.value = 1
})

async function fetchEvents() {
  loading.value = true
  error.value = null
  try {
    const response = await fetch('http://localhost:5000/api/events-all')
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    if (data.success) {
      events.value = data.events || []
    } else {
      throw new Error(data.error || 'Unknown error')
    }
  } catch (err) {
    error.value = err.message || 'Network request failed'
  } finally {
    loading.value = false
  }
}

function refresh() {
  currentPage.value = 1
  fetchEvents()
}

function formatTimestamp(timestamp) {
  const date = new Date(timestamp * 1000)
  return date.toLocaleString('en-US', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  })
}

function openImage(url) { fullImage.value = url }
function closeFullImage() { fullImage.value = null }

function eventId(event) {
  return event?._id || event?.event_id || '-'
}

async function openEventDetail(event) {
  selectedEvent.value = event
  detailError.value = null

  const id = eventId(event)
  if (!id || id === '-') return

  detailLoading.value = true
  try {
    const response = await fetch(`http://localhost:5000/api/events/${encodeURIComponent(id)}`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    selectedEvent.value = await response.json()
  } catch (err) {
    detailError.value = `Using list data; detail request failed: ${err.message || 'request failed'}`
  } finally {
    detailLoading.value = false
  }
}

function closeEventDetail() {
  selectedEvent.value = null
  detailLoading.value = false
  detailError.value = null
}

function formatDetailKey(key) {
  if (key === '_id') return 'eventID'
  return String(key).replace(/_/g, ' ')
}

function isComplexValue(value) {
  return value !== null && typeof value === 'object'
}

function formatJsonValue(value) {
  return JSON.stringify(value, null, 2)
}

function formatDetailValue(key, value) {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'number' && key === 'timestamp') {
    return `${formatTimestamp(value)} (${value})`
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

function handleImageError(e) {
  e.target.style.opacity = '0.3'
}

async function copyText(eventId, text) {
  if (!text) return

  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
    } else {
      const textarea = document.createElement('textarea')
      textarea.value = text
      textarea.setAttribute('readonly', '')
      textarea.style.position = 'absolute'
      textarea.style.left = '-9999px'
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
    }

    copiedEventId.value = eventId
    window.setTimeout(() => {
      if (copiedEventId.value === eventId) {
        copiedEventId.value = null
      }
    }, 1500)
  } catch {
    copiedEventId.value = null
  }
}

onMounted(() => {
  fetchEvents()
  refreshInterval = setInterval(fetchEvents, 10000)
})

onUnmounted(() => {
  if (refreshInterval) clearInterval(refreshInterval)
})
</script>

<style scoped>
.event-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* Filter bar */
.filter-bar {
  display: flex;
  align-items: flex-end;
  gap: 20px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 16px 20px;
  flex-wrap: wrap;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.filter-label {
  font-size: 12px;
  color: var(--color-text-muted);
}

.filter-select {
  min-width: 140px;
}

.time-btns {
  display: flex;
  gap: 0;
}

.time-btn {
  padding: 6px 14px;
  font-size: 13px;
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0;
}

.time-btn:first-child {
  border-radius: var(--radius-sm) 0 0 var(--radius-sm);
}

.time-btn:last-child {
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}

.time-btn + .time-btn {
  border-left: none;
}

.time-btn.active {
  background: var(--color-accent);
  color: #fff;
  border-color: var(--color-accent);
}

.btn-refresh {
  background: var(--color-accent);
  color: #fff;
  padding: 7px 16px;
  border-radius: var(--radius-sm);
  margin-left: auto;
}

.btn-refresh:hover:not(:disabled) {
  background: var(--color-accent-hover);
}

/* Error / state */
.error-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-md);
  color: var(--color-danger);
}

.btn-retry {
  background: var(--color-danger);
  color: #fff;
  padding: 4px 12px;
  border-radius: var(--radius-sm);
  font-size: 12px;
}

.state-msg {
  text-align: center;
  padding: 40px;
  color: var(--color-text-muted);
}

/* Events grid */
.events-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}

.event-card {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 16px;
  transition: border-color 0.2s;
  cursor: pointer;
}

.event-card:hover,
.event-card:focus-visible {
  border-color: var(--color-accent);
  outline: none;
}

.event-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.event-type-tag {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}

.type-smoke_flame {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger);
}

.type-parking_violation {
  background: rgba(59, 130, 246, 0.15);
  color: var(--color-accent);
}

.type-common_space_utilization {
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success);
}

.confidence {
  font-weight: 700;
  font-size: 13px;
  color: var(--color-success);
}

.event-detail {
  margin-bottom: 6px;
}

.event-detail p {
  font-size: 13px;
  color: var(--color-text-secondary);
  margin-bottom: 2px;
}

.event-url-block {
  padding-top: 4px;
}

.event-url-row {
  margin-bottom: 8px;
}

.event-url-link {
  color: var(--color-accent);
  word-break: break-all;
  user-select: text;
}

.event-url-link:hover {
  text-decoration: underline;
}

.copy-url-btn {
  padding: 5px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  font-size: 12px;
}

.copy-url-btn:hover {
  background: var(--color-bg-hover);
}

.analysis-details {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--color-border);
}

.analysis-details summary {
  cursor: pointer;
  font-size: 12px;
  color: var(--color-accent);
}

.analysis-pre {
  margin-top: 8px;
  padding: 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-x: auto;
}

.detail-label {
  color: var(--color-text-muted);
}

.violation-tag {
  display: inline-block;
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger);
  padding: 2px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  margin: 6px 0;
}

.image-preview img {
  width: 100%;
  height: 160px;
  object-fit: cover;
  border-radius: var(--radius-sm);
  cursor: zoom-in;
  margin-top: 10px;
  border: 1px solid var(--color-border);
}

.no-image {
  color: var(--color-text-muted);
  font-size: 12px;
  text-align: center;
  padding: 10px;
  margin-top: 8px;
}

/* Pagination */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
}

.page-btn {
  padding: 6px 16px;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
}

.page-btn:hover:not(:disabled) {
  background: var(--color-bg-hover);
}

.page-info {
  font-size: 13px;
  color: var(--color-text-secondary);
}

/* Image modal */
.image-modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3000;
  cursor: pointer;
}

.image-modal img {
  max-width: 90%;
  max-height: 90%;
  object-fit: contain;
  border-radius: var(--radius-md);
}

/* Event detail modal */
.event-detail-modal {
  position: fixed;
  inset: 0;
  z-index: 2100;
  background: rgba(0, 0, 0, 0.72);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.event-detail-card {
  width: min(980px, 100%);
  max-height: 90vh;
  overflow: auto;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
}

.event-detail-header {
  position: sticky;
  top: 0;
  z-index: 1;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 20px;
  background: var(--color-bg-secondary);
  border-bottom: 1px solid var(--color-border);
}

.event-detail-header h3 {
  margin-top: 10px;
  font-size: 18px;
  font-weight: 700;
  color: var(--color-text-primary);
}

.event-id-line {
  margin-top: 4px;
  font-family: "Cascadia Code", "JetBrains Mono", monospace;
  font-size: 12px;
  color: var(--color-text-muted);
  word-break: break-all;
}

.modal-close-btn {
  align-self: flex-start;
  padding: 6px 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

.modal-close-btn:hover {
  background: var(--color-bg-hover);
}

.detail-loading,
.detail-error {
  margin: 14px 20px 0;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  font-size: 13px;
}

.detail-loading {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
}

.detail-error {
  background: rgba(245, 158, 11, 0.12);
  color: var(--color-warning);
}

.detail-image-wrap {
  padding: 18px 20px 0;
}

.detail-image-wrap img {
  width: 100%;
  max-height: 420px;
  object-fit: contain;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-primary);
  cursor: zoom-in;
}

.detail-section {
  padding: 18px 20px 20px;
}

.detail-section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.detail-section h4 {
  font-size: 14px;
  font-weight: 700;
  color: var(--color-text-primary);
}

.attribute-list {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.attribute-row {
  display: grid;
  grid-template-columns: minmax(140px, 220px) minmax(0, 1fr);
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--color-border);
}

.attribute-row:last-child {
  border-bottom: 0;
}

.attribute-key {
  font-size: 12px;
  color: var(--color-text-muted);
  text-transform: capitalize;
}

.attribute-value,
.attribute-pre {
  min-width: 0;
  font-family: "Cascadia Code", "JetBrains Mono", monospace;
  font-size: 12px;
  color: var(--color-text-secondary);
  word-break: break-word;
}

.attribute-pre {
  margin: 0;
  white-space: pre-wrap;
}

@media (max-width: 720px) {
  .event-detail-modal {
    padding: 12px;
  }

  .event-detail-header {
    flex-direction: column;
  }

  .attribute-row {
    grid-template-columns: 1fr;
    gap: 4px;
  }
}
</style>
