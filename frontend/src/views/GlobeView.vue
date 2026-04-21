<template>
  <div class="globe-page">
    <div ref="globeContainer" class="globe-container" />

    <!-- Camera info panel -->
    <transition name="fade">
      <div v-if="selected" class="info-panel">
        <div class="info-header">
          <span class="info-title">{{ selected.cameraId }}</span>
          <button class="info-close" @click="selected = null">&times;</button>
        </div>
        <div class="info-body">
          <div class="info-row">
            <span class="info-label">Location</span>
            <span>{{ selected.location }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Event Type</span>
            <span class="event-tag" :class="'type-' + selected.eventType">{{ typeLabel(selected.eventType) }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Confidence</span>
            <span class="confidence">{{ (selected.confidence * 100).toFixed(1) }}%</span>
          </div>
          <div class="info-row">
            <span class="info-label">Events</span>
            <span>{{ selected.eventCount }}</span>
          </div>
        </div>
        <img
          v-if="selected.imageUrl"
          :src="selected.imageUrl"
          class="info-image"
          @error="e => e.target.style.display = 'none'"
        />
      </div>
    </transition>

    <!-- Loading overlay -->
    <div v-if="loading" class="globe-loading">Loading events...</div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import Globe from 'globe.gl'

const globeContainer = ref(null)
const selected = ref(null)
const loading = ref(true)

let globe = null
let resizeObserver = null
let autoRotateTimer = null

const EVENT_COLORS = {
  smoke_flame: '#ef4444',
  parking_violation: '#3b82f6',
  common_space_utilization: '#22c55e',
}

function typeLabel(t) {
  const map = {
    smoke_flame: 'Smoke/Fire Alert',
    parking_violation: 'Parking Violation',
    common_space_utilization: 'Space Analysis',
  }
  return map[t] || t
}

function parseLatLng(str) {
  if (!str) return null
  const parts = str.split(',').map(Number)
  if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) return parts
  return null
}

function buildPoints(events) {
  const grouped = new Map()
  for (const e of events) {
    if (!e.lat_lng || !e.camera_id) continue
    const arr = grouped.get(e.camera_id) || []
    arr.push(e)
    grouped.set(e.camera_id, arr)
  }

  const points = []
  for (const [cameraId, evts] of grouped) {
    evts.sort((a, b) => b.timestamp - a.timestamp)
    const latest = evts[0]
    const coords = parseLatLng(latest.lat_lng)
    if (!coords) continue
    points.push({
      lat: coords[0],
      lng: coords[1],
      cameraId,
      location: latest.location || 'Unknown',
      eventType: latest.event_type,
      confidence: latest.confidence || 0,
      eventCount: evts.length,
      imageUrl: latest.image_url,
      color: EVENT_COLORS[latest.event_type] || '#f59e0b',
    })
  }
  return points
}

function startAutoRotate() {
  if (!globe) return
  const controls = globe.controls()
  controls.autoRotate = true
  controls.autoRotateSpeed = 0.5
}

function pauseAutoRotate() {
  if (!globe) return
  globe.controls().autoRotate = false
  if (autoRotateTimer) clearTimeout(autoRotateTimer)
  autoRotateTimer = setTimeout(startAutoRotate, 3000)
}

async function init() {
  if (!globeContainer.value) return

  let events = []
  try {
    const res = await fetch('http://localhost:5000/api/events-all')
    const data = await res.json()
    if (data.success) events = data.events || []
  } catch { /* empty */ }

  const points = buildPoints(events)
  loading.value = false

  globe = Globe()(globeContainer.value)
    .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
    .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
    .pointsData(points)
    .pointLat('lat')
    .pointLng('lng')
    .pointColor('color')
    .pointAltitude(0.01)
    .pointRadius(0.4)
    .pointLabel(d => `<b>${d.cameraId}</b><br/>${d.location}`)
    .onPointClick(point => {
      selected.value = point
      pauseAutoRotate()
    })

  const { clientWidth, clientHeight } = globeContainer.value
  globe.width(clientWidth).height(clientHeight)

  startAutoRotate()
  globe.controls().addEventListener('start', pauseAutoRotate)

  resizeObserver = new ResizeObserver(([entry]) => {
    if (globe) globe.width(entry.contentRect.width).height(entry.contentRect.height)
  })
  resizeObserver.observe(globeContainer.value)
}

onMounted(init)

onUnmounted(() => {
  if (autoRotateTimer) clearTimeout(autoRotateTimer)
  if (resizeObserver) resizeObserver.disconnect()
  if (globe) {
    const el = globe.renderer().domElement
    el.parentNode?.removeChild(el)
    globe.scene().clear()
    globe.renderer().dispose()
    globe = null
  }
})
</script>

<style scoped>
.globe-page {
  position: relative;
  width: 100%;
  height: calc(100vh - var(--header-height) - 48px);
}

.globe-container {
  width: 100%;
  height: 100%;
  border-radius: var(--radius-md);
  overflow: hidden;
}

/* Info panel */
.info-panel {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 320px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  z-index: 10;
}

.info-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid var(--color-border);
}

.info-title {
  font-weight: 600;
  font-size: 15px;
  color: var(--color-text-primary);
}

.info-close {
  background: transparent;
  color: var(--color-text-muted);
  font-size: 20px;
  line-height: 1;
  padding: 0 4px;
}

.info-close:hover {
  color: var(--color-text-primary);
}

.info-body {
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  color: var(--color-text-secondary);
}

.info-label {
  color: var(--color-text-muted);
}

.event-tag {
  padding: 2px 8px;
  border-radius: 10px;
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
  color: var(--color-success);
}

.info-image {
  width: 100%;
  height: 160px;
  object-fit: cover;
}

/* Loading */
.globe-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-muted);
  font-size: 14px;
}

/* Transition */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.25s;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}
</style>
