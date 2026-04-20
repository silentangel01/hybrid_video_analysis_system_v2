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
      <div v-for="event in paginatedEvents" :key="event._id" class="event-card">
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

        <div v-if="event.event_type === 'common_space_utilization' && event.analysis_summary?.space_occupancy" class="event-detail">
          <p><span class="detail-label">Occupancy: </span>{{ event.analysis_summary.space_occupancy }}</p>
        </div>

        <div v-if="event.description" class="event-detail">
          <p><span class="detail-label">Description: </span>{{ event.description }}</p>
        </div>

        <div v-if="event.event_type === 'parking_violation'" class="violation-tag">Violation</div>

        <div v-if="event.image_url" class="image-preview">
          <img
            :src="event.image_url"
            :alt="typeLabel(event.event_type)"
            @click="openImage(event.image_url)"
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
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

const events = ref([])
const loading = ref(false)
const error = ref(null)
const fullImage = ref(null)
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

function handleImageError(e) {
  e.target.style.opacity = '0.3'
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
}

.event-card:hover {
  border-color: var(--color-accent);
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
  z-index: 2000;
  cursor: pointer;
}

.image-modal img {
  max-width: 90%;
  max-height: 90%;
  object-fit: contain;
  border-radius: var(--radius-md);
}
</style>
