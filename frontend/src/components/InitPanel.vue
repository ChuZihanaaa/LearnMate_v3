<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/rag'
import { ElMessage } from 'element-plus'

const { t } = useI18n()
const loading = ref(false)
const count = ref<number | null>(null)

const handleInit = async () => {
  loading.value = true
  try {
    const data = await api.initVectorstore()
    count.value = data.vector_count
    ElMessage.success('重建成功')
  } catch (e: any) {
    ElMessage.error(e.message || '重建失败')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="wrap">
    <el-button type="primary" :loading="loading" @click="handleInit">
      {{ t('rebuild') }}
    </el-button>
    <div v-if="count !== null" class="result">
      {{ t('rebuildResult') }}：{{ count }}
    </div>
  </div>
</template>

<style scoped>
.wrap {
  display: flex;
  align-items: center;
  gap: 12px;
}
.result {
  color: #475569;
}
</style>

