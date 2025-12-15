<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useAppStore } from '../stores/app'
import { computed, ref, onMounted } from 'vue'
import UploadPanel from '../components/UploadPanel.vue'
import InitPanel from '../components/InitPanel.vue'
import QAPanel from '../components/QAPanel.vue'
import QuizPanel from '../components/QuizPanel.vue'
import CachePanel from '../components/CachePanel.vue'

const { t, locale } = useI18n()
const app = useAppStore()

const activeTab = ref('qa')

const isDark = computed(() => app.theme === 'dark')

const handleThemeToggle = (val: boolean) => {
  app.setTheme(val ? 'dark' : 'light')
}

const handleLocale = (val: string) => {
  locale.value = val
  app.setLocale(val as 'zh-CN' | 'en-US')
}

onMounted(() => {
  app.setTheme(app.theme)
  locale.value = app.locale
})
</script>

<template>
  <div class="page">
    <header class="topbar">
      <div class="title">{{ t('title') }}</div>
      <div class="actions">
        <el-select
          size="small"
          :model-value="locale"
          style="width: 140px"
          @change="handleLocale"
        >
          <el-option label="中文" value="zh-CN" />
          <el-option label="English" value="en-US" />
        </el-select>
        <el-switch
          :model-value="isDark"
          size="small"
          inline-prompt
          :active-text="t('dark')"
          @change="handleThemeToggle"
        />
      </div>
    </header>

    <el-card class="card">
      <el-tabs v-model="activeTab">
        <el-tab-pane :label="t('qa')" name="qa">
          <QAPanel />
        </el-tab-pane>
        <el-tab-pane :label="t('quiz')" name="quiz">
          <QuizPanel />
        </el-tab-pane>
        <el-tab-pane :label="t('upload')" name="upload">
          <UploadPanel />
        </el-tab-pane>
        <el-tab-pane :label="t('init')" name="init">
          <InitPanel />
        </el-tab-pane>
        <el-tab-pane :label="t('cache')" name="cache">
          <CachePanel />
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<style scoped>
.page {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.title {
  font-size: 20px;
  font-weight: 700;
}

.actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.card {
  min-height: 70vh;
}
</style>

