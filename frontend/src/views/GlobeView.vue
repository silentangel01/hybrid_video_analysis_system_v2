<template>
  <div class="globe-page">
    <div class="globe-toolbar">
      <button class="refresh-btn" @click="refreshEvents" :disabled="loading">
        {{ loading ? 'Refreshing...' : 'Refresh Events' }}
      </button>
    </div>
    <div ref="globeContainer" class="globe-container" />

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
            <span class="info-label">Coordinates</span>
            <span class="mono">{{ selected.latLng }}</span>
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
            <span class="info-label">Stage</span>
            <span>{{ stageLabel(selected.detectionStage) }}</span>
          </div>
          <div v-if="selected.objectCount" class="info-row">
            <span class="info-label">Objects</span>
            <span>{{ selected.objectCount }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Latest Event</span>
            <span>{{ formatTime(selected.timestamp) }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Total Events</span>
            <span>{{ selected.eventCount }}</span>
          </div>
          <div v-if="selected.areaCode" class="info-row">
            <span class="info-label">Area Code</span>
            <span>{{ selected.areaCode }}</span>
          </div>
          <div v-if="selected.group" class="info-row">
            <span class="info-label">Group</span>
            <span>{{ selected.group }}</span>
          </div>
          <div v-if="selected.description" class="info-desc">
            {{ selected.description }}
          </div>
          <template v-if="selected.analysisSummary">
            <div v-if="selected.analysisSummary.estimated_people_count != null" class="info-row">
              <span class="info-label">People Count</span>
              <span>{{ selected.analysisSummary.estimated_people_count }}</span>
            </div>
            <div v-if="selected.analysisSummary.space_occupancy" class="info-row">
              <span class="info-label">Occupancy</span>
              <span class="event-tag" :class="occupancyClass(selected.analysisSummary.space_occupancy)">
                {{ selected.analysisSummary.space_occupancy }}
              </span>
            </div>
            <div v-if="selected.analysisSummary.activity_types?.length" class="info-row">
              <span class="info-label">Activities</span>
              <span>{{ formatActivityTypes(selected.analysisSummary.activity_types) }}</span>
            </div>
            <div v-if="selected.analysisSummary.safety_concerns !== undefined" class="info-row">
              <span class="info-label">Safety</span>
              <span>{{ selected.analysisSummary.safety_concerns ? 'Potential concern' : 'No visible concern' }}</span>
            </div>
            <div v-if="selected.analysisSummary.scene_summary" class="info-desc">
              {{ selected.analysisSummary.scene_summary }}
            </div>
            <div v-if="selected.analysisSummary.occupancy_reason" class="info-desc info-note">
              {{ selected.analysisSummary.occupancy_reason }}
            </div>
          </template>
        </div>
        <details v-if="selected.analysisResult" class="info-raw">
          <summary>Full Analysis</summary>
          <pre>{{ formatAnalysisResult(selected.analysisResult) }}</pre>
        </details>
        <img
          v-if="selected.imageUrl"
          :src="selected.imageUrl"
          class="info-image"
          @error="e => e.target.style.display = 'none'"
        />
      </div>
    </transition>

    <div v-if="loading" class="globe-loading">Loading events...</div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import Globe from 'globe.gl'

const globeContainer = ref(null)
const selected = ref(null)
const loading = ref(false)

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

function stageLabel(s) {
  const map = {
    yolo_initial: 'YOLO Initial',
    qwen_verified: 'Qwen Verified',
    qwen_vl_analysis: 'Qwen VL Analysis',
  }
  return map[s] || s || '-'
}

function formatTime(ts) {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleString('en-US', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function occupancyClass(level) {
  const map = { low: 'occ-low', medium: 'occ-med', high: 'occ-high' }
  return map[level] || ''
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
      latLng: latest.lat_lng,
      location: latest.location || 'Unknown',
      eventType: latest.event_type,
      confidence: latest.confidence || 0,
      detectionStage: latest.detection_stage,
      objectCount: latest.object_count,
      description: latest.description,
      timestamp: latest.timestamp,
      areaCode: latest.area_code,
      group: latest.group,
      analysisSummary: latest.analysis_summary,
      analysisResult: latest.analysis_result,
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

  globe = Globe()(globeContainer.value)
    .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
    .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
    .pointsData([])
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

async function refreshEvents() {
  loading.value = true

  let events = []
  try {
    const res = await fetch('http://localhost:5000/api/events-all')
    const data = await res.json()
    if (data.success) events = data.events || []
  } catch {
    // ignore fetch errors and keep the current globe state
  }

  const points = buildPoints(events)
  if (globe) {
    globe.pointsData(points)
    if (selected.value) {
      selected.value = points.find(point => point.cameraId === selected.value.cameraId) || null
    }
  }

  loading.value = false
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

.globe-toolbar {
  position: absolute;
  top: 16px;
  left: 16px;
  z-index: 10;
}

.refresh-btn {
  min-width: 132px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  background: rgba(15, 23, 42, 0.82);
  border: 1px solid rgba(148, 163, 184, 0.25);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  backdrop-filter: blur(8px);
}

.refresh-btn:hover:not(:disabled) {
  background: rgba(30, 41, 59, 0.9);
}

.refresh-btn:disabled {
  cursor: default;
  opacity: 0.75;
}

.globe-container {
  width: 100%;
  height: 100%;
  border-radius: var(--radius-md);
  overflow: hidden;
}

.info-panel {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 320px;
  max-height: calc(100% - 32px);
  overflow-y: auto;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
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

.info-desc {
  font-size: 12px;
  color: var(--color-text-muted);
  padding: 6px 0 2px;
  line-height: 1.5;
  border-top: 1px solid var(--color-border);
  margin-top: 4px;
}

.info-note {
  color: var(--color-text-secondary);
}

.mono {
  font-family: monospace;
  font-size: 12px;
}

.occ-low {
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success);
}

.occ-med {
  background: rgba(245, 158, 11, 0.15);
  color: var(--color-warning);
}

.occ-high {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger);
}

.info-raw {
  padding: 0 16px 16px;
  border-top: 1px solid var(--color-border);
}

.info-raw summary {
  cursor: pointer;
  padding-top: 12px;
  font-size: 12px;
  color: var(--color-accent);
}

.info-raw pre {
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

.globe-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-muted);
  font-size: 14px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.25s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
