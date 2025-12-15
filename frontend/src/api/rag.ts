import http from './http'

export type AskResponse = {
  answer: string
  sources: string[]
  response_time_ms: number
  cache_hit: boolean
}

export type QuizResponse = {
  question: string
  options: string[]
  answer: string
  explanation: string
}

export type UploadResult = {
  file: string
  chunks: number
}

export type UploadResponse = {
  results: UploadResult[]
  total_chunks: number
}

export type CacheEntry = {
  key: string
  preview: string
  length: number
}

export const api = {
  ask(question: string) {
    return http.post<AskResponse>('/api/v1/ask', { question }).then((r) => r.data)
  },
  quiz(topic: string) {
    return http.post<QuizResponse>('/api/v1/quiz', { topic }).then((r) => r.data)
  },
  upload(files: File[]) {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    return http.post<UploadResponse>('/api/v1/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },
  initVectorstore() {
    return http.post<{ vector_count: number }>('/api/v1/init').then((r) => r.data)
  },
  history(limit = 50) {
    return http.get<CacheEntry[]>('/api/v1/history', { params: { limit } }).then((r) => r.data)
  },
  clearCache() {
    return http.delete<{ remaining: number }>('/api/v1/cache').then((r) => r.data)
  },
  deleteCache(key: string) {
    return http.delete<{ remaining: number }>(`/api/v1/cache/${encodeURIComponent(key)}`).then((r) => r.data)
  },
}

