<template>
  <div class="upload-page">
    <div class="upload-card">
      <h3 class="card-title">Upload Video</h3>
      <div class="card-body">
        <!-- 拖拽上传区域 -->
        <div
          class="drop-zone"
          :class="{ 'drop-active': isDragging, 'has-file': videoFile }"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="onDrop"
          @click="triggerFileInput"
        >
          <input
            ref="fileInput"
            type="file"
            accept=".mp4"
            class="file-input-hidden"
            @change="onFileChange"
          />
          <div class="drop-content">
            <svg v-if="!videoFile" viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="var(--color-accent)" stroke-width="1.5">
              <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M12 4v12M8 8l4-4 4 4"/>
            </svg>
            <svg v-else viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="var(--color-success)" stroke-width="1.5">
              <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            <p class="drop-main-text">
              {{ videoFile ? videoFile.name : 'Drag and drop video file here' }}
            </p>
            <p class="drop-sub-text">
              {{ videoFile ? formatFileSize(videoFile.size) : 'Or click to select a file (.mp4 only)' }}
            </p>
          </div>
        </div>

        <!-- 任务类型 -->
        <div class="form-row">
          <label class="form-label">Analysis Type</label>
          <select v-model="selectedType" class="type-select">
            <option value="parking">Parking Violation</option>
            <option value="smoke_flame">Smoke/Fire</option>
            <option value="common_space">Common Space</option>
          </select>
        </div>

        <!-- 上传按钮 -->
        <button class="btn-upload" @click="uploadVideo" :disabled="!videoFile || uploading">
          {{ uploading ? 'Uploading...' : 'Upload' }}
        </button>

        <!-- 提示信息 -->
        <div v-if="message" :class="['msg-box', msgType]">{{ message }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const videoFile = ref(null)
const selectedType = ref('parking')
const message = ref('')
const msgType = ref('success')
const uploading = ref(false)
const isDragging = ref(false)
const fileInput = ref(null)

function triggerFileInput() {
  fileInput.value?.click()
}

function onFileChange(event) {
  const file = event.target.files[0]
  validateAndSet(file)
}

function onDrop(event) {
  isDragging.value = false
  const file = event.dataTransfer.files[0]
  validateAndSet(file)
}

function validateAndSet(file) {
  if (!file) return
  if (file.type !== 'video/mp4' && !file.name.endsWith('.mp4')) {
    videoFile.value = null
    showMsg('Please select a valid .mp4 video file', 'error')
    return
  }
  videoFile.value = file
  message.value = ''
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

function showMsg(text, type = 'success') {
  message.value = text
  msgType.value = type
}

async function uploadVideo() {
  if (!videoFile.value) return
  uploading.value = true
  message.value = ''

  const formData = new FormData()
  formData.append('video', videoFile.value)
  formData.append('type', selectedType.value)

  try {
    const response = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) throw new Error(`HTTP ${response.status}`)

    const text = await response.text()
    let result
    try {
      result = JSON.parse(text)
    } catch {
      throw new Error('Invalid server response')
    }

    if (result.success) {
      showMsg('Video uploaded, backend will analyze automatically...', 'success')
      videoFile.value = null
      if (fileInput.value) fileInput.value.value = ''
    } else {
      showMsg('Upload failed: ' + (result.error || 'Unknown error'), 'error')
    }
  } catch {
    showMsg('Network error: cannot connect to upload service', 'error')
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.upload-page {
  max-width: 600px;
}

.upload-card {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  padding: 14px 20px;
  border-bottom: 1px solid var(--color-border);
}

.card-body {
  padding: 24px;
}

.drop-zone {
  border: 2px dashed var(--color-border);
  border-radius: var(--radius-md);
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background-color 0.2s;
  margin-bottom: 20px;
}

.drop-zone:hover,
.drop-zone.drop-active {
  border-color: var(--color-accent);
  background: rgba(59, 130, 246, 0.05);
}

.drop-zone.has-file {
  border-color: var(--color-success);
  background: rgba(34, 197, 94, 0.05);
}

.file-input-hidden {
  display: none;
}

.drop-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.drop-main-text {
  font-size: 15px;
  color: var(--color-text-primary);
  font-weight: 500;
}

.drop-sub-text {
  font-size: 13px;
  color: var(--color-text-muted);
}

.form-row {
  margin-bottom: 20px;
}

.form-label {
  display: block;
  font-size: 13px;
  color: var(--color-text-secondary);
  margin-bottom: 6px;
}

.type-select {
  width: 100%;
  padding: 8px 12px;
}

.btn-upload {
  width: 100%;
  padding: 10px;
  background: var(--color-accent);
  color: #fff;
  font-size: 15px;
  font-weight: 500;
  border-radius: var(--radius-sm);
}

.btn-upload:hover:not(:disabled) {
  background: var(--color-accent-hover);
}

.msg-box {
  margin-top: 16px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  font-size: 13px;
}

.msg-box.success {
  background: rgba(34, 197, 94, 0.1);
  color: var(--color-success);
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.msg-box.error {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
  border: 1px solid rgba(239, 68, 68, 0.3);
}
</style>
