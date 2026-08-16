<script setup>
import { computed, onMounted, ref, watch } from "vue"

const props = defineProps({
  patientId: { type: String, required: true },
  apiBase: { type: String, default: "/api/memory" },
})

const snapshot = ref(null)
const narrative = ref("")
const profile = ref({
  name: "",
  age: "",
  gender: "",
  education_years: "",
})
const factsText = ref("")
const preferencesText = ref("")
const loading = ref(false)
const saving = ref(false)
const error = ref("")
const saved = ref(false)
const mmseScore = ref("")
const weakDimensions = ref("")

const profileLabels = {
  name: "姓名",
  age: "年龄",
  gender: "性别",
  education_years: "受教育年限",
}

const emptyProfile = { ...profile.value }

const emotionRows = computed(() => {
  const latest = snapshot.value?.emotion_summary?.latest?.scores || {}
  return Object.entries(latest).map(([name, value]) => ({
    name,
    value: Number(value) || 0,
  }))
})

const mmseRows = computed(() => {
  return [...(snapshot.value?.mmse_history || [])]
    .map((item) => ({
      ...item,
      date: item.date || item.recorded_at || "未知日期",
      score: Number(item.score) || 0,
    }))
    .sort((left, right) => left.date.localeCompare(right.date))
})

const mmsePoints = computed(() => {
  const width = 320
  const height = 140
  const padding = { left: 24, right: 12, top: 12, bottom: 24 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const lastIndex = Math.max(mmseRows.value.length - 1, 1)
  return mmseRows.value.map((item, index) => ({
    x: padding.left + (mmseRows.value.length === 1 ? plotWidth / 2 : (index * plotWidth) / lastIndex),
    y: padding.top + ((35 - Math.max(0, Math.min(35, item.score))) / 35) * plotHeight,
    score: item.score,
    date: item.date,
  }))
})

const mmsePointString = computed(() =>
  mmsePoints.value.map((item) => `${item.x},${item.y}`).join(" "),
)

async function load() {
  if (!props.patientId) return
  loading.value = true
  error.value = ""
  try {
    const response = await fetch(`${props.apiBase}/${encodeURIComponent(props.patientId)}`)
    if (!response.ok) throw new Error("记忆读取失败")
    const payload = await response.json()
    snapshot.value = payload.data || payload
    profile.value = { ...emptyProfile, ...(snapshot.value.profile || {}) }
    narrative.value = snapshot.value.narrative || ""
    factsText.value = (snapshot.value.facts || []).join("\n")
    preferencesText.value = (snapshot.value.preferences || []).join("\n")
  } catch (cause) {
    error.value = cause.message
  } finally {
    loading.value = false
  }
}

async function save() {
  saving.value = true
  saved.value = false
  error.value = ""
  try {
    const profileUpdates = Object.fromEntries(
      Object.entries(profile.value).filter(([, value]) => String(value ?? "").trim()),
    )
    const response = await fetch(`${props.apiBase}/${encodeURIComponent(props.patientId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        updates: {
          profile: profileUpdates,
          facts: splitLines(factsText.value),
          preferences: splitLines(preferencesText.value),
          narrative: narrative.value,
        },
      }),
    })
    if (!response.ok) throw new Error("记忆保存失败")
    await response.json()
    await load()
    saved.value = true
  } catch (cause) {
    error.value = cause.message
  } finally {
    saving.value = false
  }
}

function splitLines(value) {
  return value
    .split(/\r?\n|[，,]/)
    .map((item) => item.trim())
    .filter(Boolean)
}

async function saveMmse() {
  const score = Number(mmseScore.value)
  if (!Number.isInteger(score) || score < 0 || score > 35) {
    error.value = "MMSE分数应为0到35之间的整数"
    return
  }
  const response = await fetch(`${props.apiBase}/${encodeURIComponent(props.patientId)}/mmse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      score,
      weak_dimensions: weakDimensions.value.split(/[，,]/).map((item) => item.trim()).filter(Boolean),
    }),
  })
  if (!response.ok) {
    error.value = "MMSE记录失败"
    return
  }
  mmseScore.value = ""
  weakDimensions.value = ""
  await load()
}

watch(() => props.patientId, load)
onMounted(load)
</script>

<template>
  <article class="memory-card" aria-live="polite">
    <header class="memory-card__header">
      <div>
        <p class="memory-card__eyebrow">长期记忆</p>
        <h2>患者记忆卡</h2>
      </div>
      <button type="button" class="memory-card__save" :disabled="saving || loading || !snapshot" @click="save">
        {{ saving ? "保存中" : saved ? "已保存" : "保存修改" }}
      </button>
    </header>

    <p v-if="loading" class="memory-card__status">正在读取</p>
    <p v-else-if="error" class="memory-card__status memory-card__status--error">{{ error }}</p>

    <template v-if="snapshot && !loading">
      <section class="memory-card__section">
        <h3>基本信息</h3>
        <div class="memory-card__profile">
          <label v-for="(value, key) in profile" :key="key">
            <span>{{ profileLabels[key] || key }}</span>
            <input v-model="profile[key]" :type="key === 'age' || key === 'education_years' ? 'number' : 'text'" />
          </label>
        </div>
      </section>

      <section class="memory-card__section">
        <h3>可编辑记忆</h3>
        <label class="memory-card__field">
          <span>长期事实</span>
          <textarea v-model="factsText" rows="3" aria-label="长期事实" />
        </label>
        <label class="memory-card__field">
          <span>兴趣爱好</span>
          <textarea v-model="preferencesText" rows="3" aria-label="兴趣爱好" />
        </label>
        <label class="memory-card__field">
          <span>记忆叙事</span>
          <textarea v-model="narrative" rows="4" aria-label="长期记忆叙事" />
        </label>
      </section>

      <section v-if="emotionRows.length" class="memory-card__section">
        <h3>近期情绪</h3>
        <div v-for="item in emotionRows" :key="item.name" class="memory-card__emotion">
          <span>{{ item.name }}</span>
          <span class="memory-card__track"><i :style="{ width: `${item.value * 100}%` }" /></span>
          <b>{{ item.value.toFixed(2) }}</b>
        </div>
      </section>

      <section v-if="mmseRows.length" class="memory-card__section">
        <h3>认知评估趋势</h3>
        <svg class="memory-card__chart" viewBox="0 0 320 140" role="img" aria-label="MMSE分数趋势">
          <text x="2" y="16">35</text>
          <text x="12" y="120">0</text>
          <line x1="24" y1="12" x2="24" y2="116" />
          <line x1="24" y1="116" x2="308" y2="116" />
          <polyline :points="mmsePointString" />
          <circle v-for="(point, index) in mmsePoints" :key="`${point.date}-${point.x}-${index}`" :cx="point.x" :cy="point.y" r="4" />
        </svg>
        <div class="memory-card__mmse-history">
          <span v-for="(item, index) in mmseRows" :key="`${item.date}-${item.score}-${index}`">{{ item.date }}：{{ item.score }}分</span>
        </div>
      </section>

      <section class="memory-card__section memory-card__mmse">
        <h3>记录MMSE</h3>
        <div class="memory-card__mmse-fields">
          <input v-model="mmseScore" inputmode="numeric" type="number" min="0" max="35" placeholder="总分" aria-label="MMSE总分" />
          <input v-model="weakDimensions" type="text" placeholder="薄弱维度" aria-label="薄弱维度" />
          <button type="button" @click="saveMmse">记录</button>
        </div>
      </section>
    </template>
  </article>
</template>

<style scoped>
.memory-card {
  background: #fff;
  border: 1px solid #d9e3e8;
  border-radius: 8px;
  color: #243142;
  padding: 18px;
  max-width: 680px;
}

.memory-card__header,
.memory-card__mmse-fields,
.memory-card__emotion {
  align-items: center;
  display: flex;
  gap: 12px;
}

.memory-card__header {
  justify-content: space-between;
}

.memory-card__eyebrow {
  color: #69798f;
  font-size: 0.8rem;
  font-weight: 700;
  margin: 0 0 4px;
}

h2,
h3 {
  margin: 0;
}

h2 {
  font-size: 1.1rem;
}

h3 {
  font-size: 0.92rem;
  margin-bottom: 10px;
}

.memory-card__save,
.memory-card__mmse-fields button {
  background: #438ea0;
  border: 0;
  border-radius: 8px;
  color: #fff;
  cursor: pointer;
  min-height: 44px;
  padding: 0 14px;
}

button:disabled {
  cursor: wait;
  opacity: 0.6;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible {
  outline: 3px solid rgba(67, 142, 160, 0.25);
  outline-offset: 2px;
}

.memory-card__status {
  color: #69798f;
  margin: 14px 0 0;
}

.memory-card__status--error {
  color: #b42318;
}

.memory-card__section {
  border-top: 1px solid #e2e8f0;
  margin-top: 18px;
  padding-top: 16px;
}

.memory-card__profile {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.memory-card__profile label,
.memory-card__field {
  display: grid;
  gap: 5px;
}

.memory-card__profile span,
.memory-card__field span {
  color: #69798f;
  font-size: 0.85rem;
}

.memory-card__field + .memory-card__field {
  margin-top: 10px;
}

textarea,
input {
  background: #fff;
  border: 1px solid #d9e3e8;
  border-radius: 8px;
  box-sizing: border-box;
  color: #111827;
  font: inherit;
  min-height: 44px;
  padding: 10px 12px;
  width: 100%;
}

textarea {
  display: block;
  resize: vertical;
}

.memory-card__emotion {
  margin: 9px 0;
}

.memory-card__emotion > span:first-child {
  flex: 0 0 72px;
  font-size: 0.85rem;
}

.memory-card__track {
  background: #edf3f8;
  border-radius: 4px;
  flex: 1;
  height: 8px;
  overflow: hidden;
}

.memory-card__track i {
  background: #438ea0;
  display: block;
  height: 100%;
}

.memory-card__emotion b {
  font-size: 0.8rem;
  min-width: 38px;
  text-align: right;
}

.memory-card__chart {
  display: block;
  height: auto;
  max-width: 100%;
  width: 320px;
}

.memory-card__chart line {
  stroke: #d9e3e8;
  stroke-width: 1;
}

.memory-card__chart text {
  fill: #69798f;
  font-size: 10px;
}

.memory-card__chart polyline {
  fill: none;
  stroke: #438ea0;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 3;
}

.memory-card__chart circle {
  fill: #fff;
  stroke: #438ea0;
  stroke-width: 2;
}

.memory-card__mmse-history {
  color: #69798f;
  display: flex;
  flex-wrap: wrap;
  font-size: 0.8rem;
  gap: 6px 12px;
}

.memory-card__mmse-fields input:first-child {
  flex: 0 0 92px;
}

.memory-card__mmse-fields input:nth-child(2) {
  flex: 1;
}

@media (max-width: 520px) {
  .memory-card {
    padding: 14px;
  }

  .memory-card__header {
    align-items: flex-start;
    flex-direction: column;
  }

  .memory-card__save {
    width: 100%;
  }

  .memory-card__profile {
    grid-template-columns: 1fr;
  }

  .memory-card__mmse-fields {
    align-items: stretch;
    flex-wrap: wrap;
  }

  .memory-card__mmse-fields input:first-child,
  .memory-card__mmse-fields input:nth-child(2) {
    flex: 1 1 100%;
  }

  .memory-card__mmse-fields button {
    width: 100%;
  }
}
</style>
