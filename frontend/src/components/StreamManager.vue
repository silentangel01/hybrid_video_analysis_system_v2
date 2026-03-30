<template>
  <div class="stream-manager">
    <h2>RTSP Stream Management</h2>

    <!-- Add stream form -->
    <div class="add-form">
      <div class="form-row">
        <input
          v-model="newUrl"
          type="text"
          placeholder="RTSP URL (e.g. rtsp://192.168.1.100:554/stream)"
          class="url-input"
        />
      </div>
      <div class="form-row task-checkboxes">
        <label>
          <input type="checkbox" value="parking_violation" v-model="newTasks" />
          Parking Violation
        </label>
        <label>
          <input type="checkbox" value="smoke_flame" v-model="newTasks" />
          Smoke/Flame Detection
        </label>
        <label>
          <input type="checkbox" value="common_space" v-model="newTasks" />
          Public Space Analysis
        </label>
      </div>
      <button @click="addStream" :disabled="!newUrl || newTasks.length === 0" class="btn-add">
        Add Stream
      </button>
      <span v-if="message" :class="['msg', messageType]">{{ message }}</span>
    </div>

    <!-- Stream list -->
    <table v-if="streams.length > 0" class="stream-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>URL</th>
          <th>Tasks</th>
          <th>Status</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="s in streams" :key="s.stream_id">
          <td>{{ s.stream_id }}</td>
          <td class="url-cell" :title="s.url">{{ s.url }}</td>
          <td>
            <!-- Edit mode -->
            <template v-if="editing === s.stream_id">
              <label v-for="t in allTasks" :key="t" class="edit-task-label">
                <input type="checkbox" :value="t" v-model="editTasks" />
                {{ taskLabel(t) }}
              </label>
            </template>
            <template v-else>
              <span v-for="t in s.tasks" :key="t" class="task-badge">{{ taskLabel(t) }}</span>
            </template>
          </td>
          <td>
            <span :class="['status-badge', 'status-' + s.status]">{{ s.status }}</span>
          </td>
          <td class="actions">
            <template v-if="editing === s.stream_id">
              <button @click="saveTasks(s.stream_id)" class="btn-save">Save</button>
              <button @click="editing = null" class="btn-cancel">Cancel</button>
            </template>
            <template v-else>
              <button @click="startEdit(s)" class="btn-edit">Edit</button>
              <button @click="removeStream(s.stream_id)" class="btn-remove">Remove</button>
            </template>
          </td>
        </tr>
      </tbody>
    </table>
    <p v-else class="no-streams">No active streams. Add one above.</p>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const API = 'http://localhost:5000';

const streams = ref([]);
const newUrl = ref('');
const newTasks = ref([]);
const message = ref('');
const messageType = ref('success');

const editing = ref(null);
const editTasks = ref([]);

const allTasks = ['parking_violation', 'smoke_flame', 'common_space'];

let pollTimer = null;

function taskLabel(t) {
  const map = {
    parking_violation: 'Parking',
    smoke_flame: 'Smoke/Flame',
    common_space: 'Public Space',
  };
  return map[t] || t;
}

function showMsg(text, type = 'success') {
  message.value = text;
  messageType.value = type;
  setTimeout(() => { message.value = ''; }, 3000);
}

async function fetchStreams() {
  try {
    const res = await fetch(`${API}/api/streams`);
    if (res.ok) {
      streams.value = await res.json();
    }
  } catch (e) {
    console.error('Failed to fetch streams:', e);
  }
}

async function addStream() {
  if (!newUrl.value || newTasks.value.length === 0) return;
  try {
    const res = await fetch(`${API}/api/streams`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: newUrl.value, tasks: newTasks.value }),
    });
    const data = await res.json();
    if (res.ok) {
      showMsg(`Added ${data.stream_id}`);
      newUrl.value = '';
      newTasks.value = [];
      await fetchStreams();
    } else {
      showMsg(data.error || 'Failed', 'error');
    }
  } catch (e) {
    showMsg('Network error', 'error');
  }
}

async function removeStream(id) {
  try {
    const res = await fetch(`${API}/api/streams/${id}`, { method: 'DELETE' });
    if (res.ok) {
      showMsg(`Removed ${id}`);
      await fetchStreams();
    } else {
      showMsg('Remove failed', 'error');
    }
  } catch (e) {
    showMsg('Network error', 'error');
  }
}

function startEdit(s) {
  editing.value = s.stream_id;
  editTasks.value = [...s.tasks];
}

async function saveTasks(id) {
  if (editTasks.value.length === 0) {
    showMsg('Select at least one task', 'error');
    return;
  }
  try {
    const res = await fetch(`${API}/api/streams/${id}/tasks`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tasks: editTasks.value }),
    });
    if (res.ok) {
      showMsg(`Updated ${id}`);
      editing.value = null;
      await fetchStreams();
    } else {
      showMsg('Update failed', 'error');
    }
  } catch (e) {
    showMsg('Network error', 'error');
  }
}

onMounted(() => {
  fetchStreams();
  pollTimer = setInterval(fetchStreams, 5000);
});

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer);
});
</script>

<style scoped>
.stream-manager {
  margin-top: 20px;
}

.add-form {
  background: #f5f5f5;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.form-row {
  margin-bottom: 10px;
}

.url-input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
  box-sizing: border-box;
}

.task-checkboxes label {
  margin-right: 16px;
  font-size: 14px;
  cursor: pointer;
}

.btn-add {
  padding: 8px 20px;
  background: #1976d2;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}
.btn-add:disabled {
  background: #aaa;
  cursor: not-allowed;
}
.btn-add:hover:not(:disabled) {
  background: #1565c0;
}

.msg {
  margin-left: 12px;
  font-size: 13px;
}
.msg.success { color: #2e7d32; }
.msg.error { color: #c62828; }

.stream-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.stream-table th,
.stream-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #e0e0e0;
  text-align: left;
}
.stream-table th {
  background: #fafafa;
  font-weight: 600;
}

.url-cell {
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: monospace;
  font-size: 13px;
}

.task-badge {
  display: inline-block;
  background: #e3f2fd;
  color: #1565c0;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  margin-right: 4px;
}

.edit-task-label {
  display: block;
  font-size: 13px;
  margin: 2px 0;
}

.status-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}
.status-running  { background: #c8e6c9; color: #2e7d32; }
.status-connecting { background: #fff9c4; color: #f57f17; }
.status-stopped  { background: #e0e0e0; color: #616161; }
.status-error    { background: #ffcdd2; color: #c62828; }

.actions button {
  padding: 4px 10px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  margin-right: 4px;
}
.btn-edit   { background: #e3f2fd; color: #1565c0; }
.btn-remove { background: #ffebee; color: #c62828; }
.btn-save   { background: #c8e6c9; color: #2e7d32; }
.btn-cancel { background: #e0e0e0; color: #616161; }

.no-streams {
  color: #999;
  text-align: center;
  padding: 30px;
}
</style>
