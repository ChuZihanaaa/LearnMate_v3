import axios from 'axios'
import { useAppStore } from '../stores/app'

const http = axios.create({
  timeout: 20000,
})

http.interceptors.request.use((config) => {
  const app = useAppStore()
  config.baseURL = app.apiBase
  return config
})

http.interceptors.response.use(
  (resp) => resp,
  (error) => {
    const msg = error?.response?.data?.detail || error.message || 'Request error'
    return Promise.reject(new Error(msg))
  }
)

export default http

