<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { api, type CacheEntry } from '../api/rag'
import { ElMessage, ElMessageBox } from 'element-plus'

const { t } = useI18n()
const loading = ref(false)
const data = ref<CacheEntry[]>([])

const fetchData = async () => {
  loading.value = true
  try {
    data.value = await api.history()
  } catch (e: any) {
    ElMessage.error(e.message || '加载失败')
  } finally {
    loading.value = false
  }
}

const clearAll = async () => {
  await ElMessageBox.confirm('确定清空缓存吗？', '提示')
  try {
    await api.clearCache()
    ElMessage.success('已清空')
    fetchData()
  } catch (e: any) {
    ElMessage.error(e.message || '操作失败')
  }
}

const removeOne = async (row: CacheEntry) => {
  try {
    await api.deleteCache(row.key)
    ElMessage.success('已删除')
    fetchData()
  } catch (e: any) {
    ElMessage.error(e.message || '操作失败')
  }
}

onMounted(fetchData)
</script>

<template>
  <div class="wrap">
    <div class="toolbar">
      <el-button size="small" @click="fetchData">{{ t('refresh') }}</el-button>
      <el-button size="small" type="danger" @click="clearAll">{{ t('clearAll') }}</el-button>
    </div>
    <el-table :data="data" v-loading="loading" size="small">
      <el-table-column prop="key" label="Key" min-width="260" show-overflow-tooltip />
      <el-table-column prop="preview" :label="t('answer')" min-width="240" show-overflow-tooltip />
      <el-table-column prop="length" label="Len" width="80" />
      <el-table-column width="120">
        <template #default="scope">
          <el-button size="small" text type="danger" @click="removeOne(scope.row)">{{ t('delete') }}</el-button>
        </template>
      </el-table-column>
    </el-table>
    <div v-if="!data.length" class="empty">{{ t('noData') }}</div>
  </div>
</template>

<style scoped>
.wrap {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.toolbar {
  display: flex;
  gap: 8px;
}
.empty {
  color: #94a3b8;
}
</style>

