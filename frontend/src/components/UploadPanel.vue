<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { api, type UploadResponse } from '../api/rag'
import { ElMessage } from 'element-plus'
import type { UploadUserFile, UploadRawFile } from 'element-plus'

const { t } = useI18n()
const loading = ref(false)
const fileList = ref<UploadUserFile[]>([])
const result = ref<UploadResponse | null>(null)

const beforeUpload = (rawFile: UploadRawFile) => {
  const ext = rawFile.name.toLowerCase()
  if (!ext.endsWith('.pdf') && !ext.endsWith('.srt')) {
    ElMessage.error('仅支持 PDF / SRT')
    return false
  }
  fileList.value.push({
    uid: rawFile.uid || Date.now(),
    name: rawFile.name,
    status: 'ready',
    raw: rawFile,
  })
  return false
}

const handleRemove = (file: any) => {
  fileList.value = fileList.value.filter((f) => f.name !== file.name)
}

const handleUpload = async () => {
  if (!fileList.value.length) {
    ElMessage.warning('请选择文件')
    return
  }
  loading.value = true
  try {
    const files = fileList.value.map((f) => f.raw as File)
    result.value = await api.upload(files)
    ElMessage.success('上传成功')
    fileList.value = []
  } catch (e: any) {
    ElMessage.error(e.message || '上传失败')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="wrap">
    <p class="tip">{{ t('uploadTip') }}</p>
    <el-upload
      drag
      multiple
      :auto-upload="false"
      :on-remove="handleRemove"
      :before-upload="beforeUpload"
      :file-list="fileList"
    >
      <i class="el-icon-upload" />
      <div class="el-upload__text">{{ t('selectFiles') }}</div>
    </el-upload>
    <el-button type="primary" :loading="loading" style="margin-top: 12px" @click="handleUpload">
      {{ loading ? t('uploading') : t('submit') }}
    </el-button>

    <div v-if="result" class="result">
      <el-alert type="success" :title="`Total chunks: ${result.total_chunks}`" />
      <el-table :data="result.results" size="small" style="margin-top: 12px">
        <el-table-column prop="file" label="File" />
        <el-table-column prop="chunks" label="Chunks" width="120" />
      </el-table>
    </div>
  </div>
</template>

<style scoped>
.wrap {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tip {
  margin: 0;
  color: #64748b;
}

.result {
  margin-top: 12px;
}
</style>

