<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { api, type QuizResponse } from '../api/rag'
import { ElMessage } from 'element-plus'

const { t } = useI18n()
const topic = ref('')
const loading = ref(false)
const data = ref<QuizResponse | null>(null)

const handleSubmit = async () => {
  if (!topic.value.trim()) {
    ElMessage.warning('请输入知识点')
    return
  }
  loading.value = true
  try {
    data.value = await api.quiz(topic.value.trim())
  } catch (e: any) {
    ElMessage.error(e.message || '生成失败')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="wrap">
    <el-input
      v-model="topic"
      :placeholder="t('topicPlaceholder')"
      clearable
      @keyup.enter="handleSubmit"
    />
    <div class="actions">
      <el-button type="primary" :loading="loading" @click="handleSubmit">
        {{ t('submit') }}
      </el-button>
    </div>

    <el-card v-if="data">
      <div class="label">{{ data.question }}</div>
      <div class="options">
        <el-alert
          v-for="(opt, idx) in data.options"
          :key="idx"
          :title="String.fromCharCode(65 + idx) + '. ' + opt"
          :type="data.answer === String.fromCharCode(65 + idx) ? 'success' : 'info'"
          :closable="false"
          size="small"
        />
      </div>
      <div class="answer">
        ✅ {{ data.answer }}
      </div>
      <div class="exp">
        {{ t('explanation') }}: {{ data.explanation }}
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.wrap {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.actions {
  display: flex;
  justify-content: flex-end;
}
.options {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 12px 0;
}
.answer {
  font-weight: 700;
  margin-bottom: 8px;
}
.exp {
  color: #475569;
}
</style>

