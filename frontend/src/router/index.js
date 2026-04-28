import { createRouter, createWebHistory } from 'vue-router'
import Layout from '../layout/Layout.vue'
import GlobeView from '../views/GlobeView.vue'

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
        meta: { title: 'Dashboard' }
      },
      {
        path: 'streams',
        name: 'Streams',
        component: () => import('../views/StreamManager.vue'),
        meta: { title: 'Streams' }
      },
      {
        path: 'events',
        name: 'Events',
        component: () => import('../views/EventList.vue'),
        meta: { title: 'Events' }
      },
      {
        path: 'upload',
        name: 'Upload',
        component: () => import('../views/UploadVideo.vue'),
        meta: { title: 'Upload' }
      },
      {
        path: 'globe',
        name: 'Globe',
        component: GlobeView,
        meta: { title: '3D Globe' }
      },
      {
        path: 'reports',
        name: 'Reports',
        component: () => import('../views/ReportsView.vue'),
        meta: { title: 'Reports' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
