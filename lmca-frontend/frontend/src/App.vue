<script setup>
import { ref } from "vue"
import MemoryCard from "./components/MemoryCard.vue"

const patientId = ref("demo-patient")
const apiBase = ref(import.meta.env.VITE_MEMORY_API_BASE || "/api/memory")
const patientIdInput = ref(patientId.value)
const apiBaseInput = ref(apiBase.value)

function loadPatient() {
  const nextPatientId = patientIdInput.value.trim()
  const nextApiBase = apiBaseInput.value.trim().replace(/\/+$/, "")
  if (!nextPatientId) return
  patientId.value = nextPatientId
  apiBase.value = nextApiBase || "/api/memory"
}
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <div class="app-brand">
        <span class="app-brand__mark" aria-hidden="true">M</span>
        <div>
          <p class="app-brand__eyebrow">LMCA / MEMORY</p>
          <p class="app-brand__name">患者长期记忆</p>
        </div>
      </div>
      <span class="app-header__mode">SQLite-first</span>
    </header>

    <main class="app-main">
      <section class="app-heading" aria-labelledby="page-title">
        <div>
          <p class="app-heading__eyebrow">PATIENT MEMORY RECORD</p>
          <h1 id="page-title">记忆卡工作台</h1>
          <p class="app-heading__meta">维护患者资料、长期事实与认知评估记录</p>
        </div>
        <div class="app-heading__marker" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      </section>

      <form class="patient-toolbar" @submit.prevent="loadPatient">
        <label>
          <span>患者 ID</span>
          <input v-model="patientIdInput" name="patient-id" autocomplete="off" />
        </label>
        <label class="patient-toolbar__api">
          <span>记忆接口</span>
          <input v-model="apiBaseInput" name="api-base" autocomplete="off" />
        </label>
        <button type="submit">载入患者</button>
      </form>

      <div class="app-content">
        <MemoryCard
          :key="`${patientId}:${apiBase}`"
          :patient-id="patientId"
          :api-base="apiBase"
          login-path="/login?next=/"
        />

        <aside class="app-aside" aria-label="连接信息">
          <div class="app-aside__section">
            <p class="app-aside__label">当前患者</p>
            <p class="app-aside__value">{{ patientId }}</p>
          </div>
          <div class="app-aside__section">
            <p class="app-aside__label">本地事实源</p>
            <p class="app-aside__value">SQLite</p>
          </div>
          <div class="app-aside__section">
            <p class="app-aside__label">远程镜像</p>
            <p class="app-aside__note">由后端环境变量决定，未配置时不影响本地记忆读写。</p>
          </div>
        </aside>
      </div>
    </main>
  </div>
</template>

<style>
:root {
  color: #243142;
  background: #eef3f5;
  font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 320px;
}

button,
input {
  font: inherit;
}

button {
  border: 0;
}

.app-shell {
  min-height: 100vh;
}

.app-header {
  align-items: center;
  background: #102a43;
  color: #fff;
  display: flex;
  justify-content: space-between;
  min-height: 68px;
  padding: 12px clamp(20px, 5vw, 72px);
}

.app-brand {
  align-items: center;
  display: flex;
  gap: 12px;
}

.app-brand__mark {
  align-items: center;
  background: #e7b86b;
  border-radius: 6px;
  color: #102a43;
  display: inline-flex;
  font-size: 1.25rem;
  font-weight: 800;
  height: 36px;
  justify-content: center;
  width: 36px;
}

.app-brand__eyebrow,
.app-brand__name,
.app-heading__eyebrow,
.app-heading__meta,
.app-aside__label,
.app-aside__value,
.app-aside__note {
  margin: 0;
}

.app-brand__eyebrow,
.app-heading__eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.app-brand__eyebrow {
  color: #b9d2df;
  margin-bottom: 2px;
}

.app-brand__name {
  font-size: 1rem;
  font-weight: 700;
}

.app-header__mode {
  border: 1px solid rgba(255, 255, 255, 0.32);
  border-radius: 999px;
  color: #dcebf0;
  font-size: 0.75rem;
  padding: 5px 10px;
}

.app-main {
  margin: 0 auto;
  max-width: 1180px;
  padding: 42px clamp(20px, 5vw, 72px) 64px;
}

.app-heading {
  align-items: flex-end;
  border-bottom: 1px solid #c9d6dc;
  display: flex;
  justify-content: space-between;
  padding-bottom: 20px;
}

.app-heading__eyebrow {
  color: #438ea0;
  margin-bottom: 8px;
}

.app-heading h1 {
  color: #102a43;
  font-size: clamp(1.7rem, 3vw, 2.35rem);
  letter-spacing: 0;
  line-height: 1.15;
  margin: 0;
}

.app-heading__meta {
  color: #69798f;
  font-size: 0.92rem;
  margin-top: 10px;
}

.app-heading__marker {
  display: flex;
  gap: 6px;
  padding-bottom: 4px;
}

.app-heading__marker span {
  background: #e7b86b;
  display: block;
  height: 8px;
  width: 8px;
}

.app-heading__marker span:nth-child(2) {
  background: #438ea0;
}

.app-heading__marker span:nth-child(3) {
  background: #d76a5e;
}

.patient-toolbar {
  align-items: flex-end;
  display: grid;
  gap: 12px;
  grid-template-columns: minmax(160px, 0.7fr) minmax(220px, 1.4fr) auto;
  margin: 24px 0 28px;
}

.patient-toolbar label {
  display: grid;
  gap: 6px;
}

.patient-toolbar label span {
  color: #526579;
  font-size: 0.8rem;
  font-weight: 700;
}

.patient-toolbar input {
  background: #fff;
  border: 1px solid #c9d6dc;
  border-radius: 6px;
  color: #243142;
  min-height: 44px;
  padding: 0 12px;
  width: 100%;
}

.patient-toolbar input:focus-visible {
  outline: 3px solid rgba(67, 142, 160, 0.25);
  outline-offset: 2px;
}

.patient-toolbar button {
  background: #d76a5e;
  border-radius: 6px;
  color: #fff;
  cursor: pointer;
  min-height: 44px;
  padding: 0 18px;
}

.patient-toolbar button:hover {
  background: #ba5148;
}

.app-content {
  align-items: start;
  display: grid;
  gap: 32px;
  grid-template-columns: minmax(0, 680px) minmax(180px, 1fr);
}

.app-aside {
  border-left: 1px solid #c9d6dc;
  padding-left: 24px;
}

.app-aside__section + .app-aside__section {
  border-top: 1px solid #d7e1e5;
  margin-top: 22px;
  padding-top: 18px;
}

.app-aside__label {
  color: #69798f;
  font-size: 0.75rem;
  font-weight: 700;
  margin-bottom: 5px;
}

.app-aside__value {
  color: #102a43;
  font-size: 1rem;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.app-aside__note {
  color: #526579;
  font-size: 0.82rem;
  line-height: 1.6;
}

@media (max-width: 780px) {
  .app-main {
    padding-top: 28px;
  }

  .app-content {
    grid-template-columns: 1fr;
  }

  .app-aside {
    border-left: 0;
    border-top: 1px solid #c9d6dc;
    padding-left: 0;
    padding-top: 20px;
  }

  .app-aside__section + .app-aside__section {
    border-top: 0;
    margin-top: 12px;
    padding-top: 0;
  }
}

@media (max-width: 560px) {
  .app-header {
    padding-inline: 16px;
  }

  .app-header__mode {
    display: none;
  }

  .app-main {
    padding-inline: 14px;
  }

  .app-heading__marker {
    display: none;
  }

  .patient-toolbar {
    grid-template-columns: 1fr;
  }

  .patient-toolbar button {
    width: 100%;
  }
}
</style>
