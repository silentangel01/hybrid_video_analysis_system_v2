import { ref, onMounted, onUnmounted } from 'vue'

const fireAlert = ref({
  active: false,
  sources: []
})

let pollingInterval = null
let refCount = 0

async function checkFireEvents() {
  try {
    const response = await fetch('http://localhost:5000/api/events')
    if (!response.ok) return

    const data = await response.json()

    if (data.success && data.fireDetected && Array.isArray(data.sources) && data.sources.length > 0) {
      fireAlert.value = {
        active: true,
        sources: [...new Set(data.sources.filter(Boolean))]
      }
    } else {
      fireAlert.value = { active: false, sources: [] }
    }
  } catch {
    // 静默处理网络错误
  }
}

export function useFireAlert() {
  onMounted(() => {
    refCount++
    if (refCount === 1) {
      checkFireEvents()
      pollingInterval = setInterval(checkFireEvents, 10000)
    }
  })

  onUnmounted(() => {
    refCount--
    if (refCount === 0 && pollingInterval) {
      clearInterval(pollingInterval)
      pollingInterval = null
    }
  })

  return { fireAlert }
}
