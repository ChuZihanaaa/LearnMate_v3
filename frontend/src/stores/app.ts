import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

type Theme = 'light' | 'dark'
type Locale = 'zh-CN' | 'en-US'

export const useAppStore = defineStore('app', () => {
  const apiBase = ref<string>(import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000')
  const theme = ref<Theme>('light')
  const locale = ref<Locale>('zh-CN')

  const setTheme = (value: Theme) => {
    theme.value = value
    const root = document.documentElement
    if (value === 'dark') {
      root.classList.add('dark')
      root.setAttribute('data-theme', 'dark')
    } else {
      root.classList.remove('dark')
      root.setAttribute('data-theme', 'light')
    }
  }

  const toggleTheme = () => setTheme(theme.value === 'dark' ? 'light' : 'dark')

  watch(theme, (val) => setTheme(val), { immediate: true })

  const setLocale = (value: Locale) => {
    locale.value = value
  }

  return {
    apiBase,
    theme,
    locale,
    setTheme,
    toggleTheme,
    setLocale,
  }
})

