<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { api, type AskResponse } from '../api/rag'
import { ElMessage } from 'element-plus'

const { t } = useI18n()
const question = ref('')
const loading = ref(false)
const answer = ref<AskResponse | null>(null)

const history = ref<AskResponse[]>([])

const handleAsk = async () => {
  if (!question.value.trim()) {
    ElMessage.warning('请输入问题')
    return
  }
  loading.value = true
  try {
    const res = await api.ask(question.value.trim())
    answer.value = res
    history.value.unshift(res)
  } catch (e: any) {
    ElMessage.error(e.message || '请求失败')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="wrap">
    <el-input
      v-model="question"
      type="textarea"
      :placeholder="t('questionPlaceholder')"
      :rows="3"
    />
    <div class="actions">
      <el-button type="primary" :loading="loading" @click="handleAsk">
        {{ t('submit') }}
      </el-button>
    </div>

    <div v-if="answer" class="answer">
      <el-card>
        <div class="label">{{ t('answer') }}</div>
        <div class="content">{{ answer?.answer }}</div>
        <div class="meta">
          <span>{{ t('duration') }}: {{ answer?.response_time_ms }} ms</span>
          <span>{{ t('cacheHit') }}: {{ answer?.cache_hit ? '✅' : '❌' }}</span>
        </div>
        <div class="sources" v-if="answer?.sources?.length">
          <span class="label">{{ t('sources') }}:</span>
          <el-tag v-for="s in answer?.sources" :key="s" size="small" style="margin-right: 8px">
            {{ s }}
          </el-tag>
        </div>
      </el-card>
    </div>

    <div v-if="history.length" class="history">
      <div class="label">History</div>
      <el-timeline>
        <el-timeline-item
          v-for="(item, idx) in history"
          :key="idx"
          :timestamp="`${item.response_time_ms} ms`"
          placement="top"
        >
          <div class="content">{{ item.answer }}</div>
        </el-timeline-item>
      </el-timeline>
    </div>
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

.label {
  font-weight: 600;
  margin-bottom: 6px;
}

.content {
  white-space: pre-wrap;
}

.meta {
  margin-top: 8px;
  display: flex;
  gap: 12px;
  color: #6b7280;
  font-size: 13px;
}

.sources {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.history {
  margin-top: 16px;
}
</style>

