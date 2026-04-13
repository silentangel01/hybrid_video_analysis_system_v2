import { createRouter, createWebHistory } from 'vue-router'
import Layout from '../layout/Layout.vue'

const routes = [
  {
    path: '/',
    component: Layout,
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('../views/Dashboard.vue'),
        meta: { title: '数据概览' }
      },
      {
        path: 'streams',
        name: 'Streams',
        component: () => import('../views/StreamManager.vue'),
        meta: { title: '视频流管理' }
      },
      {
        path: 'events',
        name: 'Events',
        component: () => import('../views/EventList.vue'),
        meta: { title: '事件记录' }
      },
      {
        path: 'upload',
        name: 'Upload',
        component: () => import('../views/UploadVideo.vue'),
        meta: { title: '视频上传' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
