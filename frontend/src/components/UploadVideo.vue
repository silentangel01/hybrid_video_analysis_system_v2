<!-- src/components/UploadVideo.vue -->
<template>
  <div style="margin: 20px; padding: 20px; border: 1px solid #ccc;">
    <h2>上传视频</h2>

    <!-- 文件选择 -->
    <input type="file" accept=".mp4" @change="onFileChange" />

    <!-- 类型选择 -->
    <select v-model="selectedType" style="margin-left: 10px;">
      <option value="parking">parking</option>
      <option value="smoke_flame">smoke_flame</option>
      <option value="common_space">common_space</option>
    </select>

    <!-- 上传按钮 -->
    <button @click="uploadVideo" :disabled="!videoFile" style="margin-left: 10px;">
      上传
    </button>

    <!-- 提示信息 -->
    <p v-if="message" :style="{ color: messageColor }">{{ message }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue';

// 响应式数据
const videoFile = ref(null);
const selectedType = ref('parking'); // 默认类型
const message = ref('');
const messageColor = ref('black');

// 处理文件选择
function onFileChange(event) {
  const file = event.target.files[0];
  if (file && file.type === 'video/mp4') {
    videoFile.value = file;
    message.value = '';
  } else {
    videoFile.value = null;
    message.value = '请选择有效的 .mp4 视频文件';
    messageColor.value = 'red';
  }
}

// 上传视频
async function uploadVideo() {
  if (!videoFile.value) return;

  const formData = new FormData();
  formData.append('video', videoFile.value);
  formData.append('type', selectedType.value);

  try {
    const response = await fetch('http://localhost:8080/upload', {
      method: 'POST',
      body: formData,
    });

    // 先检查状态码是否正常
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // 确保响应体不为空
    const text = await response.text(); // 先读取文本
    console.log('Response text:', text); // 调试用

    // 如果是JSON，解析它；否则抛错
    let result;
    try {
      result = JSON.parse(text);
    } catch (e) {
      throw new Error('Invalid JSON response');
    }

    if (result.success) {
      message.value = '✅ 视频已上传！后端将自动分析...';
      messageColor.value = 'green';
    } else {
      message.value = '❌ 上传失败：' + (result.error || '未知错误');
      messageColor.value = 'red';
    }
  } catch (error) {
    console.error('Upload error:', error);
    message.value = '❌ 网络错误：无法连接上传服务';
    messageColor.value = 'red';
  }
}
</script>