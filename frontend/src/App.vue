<script setup lang="ts">
import DOMPurify from "dompurify";
import { marked } from "marked";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import {
  Activity,
  AlertCircle,
  BookOpenText,
  Bot,
  CheckCircle2,
  ClipboardList,
  Database,
  FileImage,
  Files,
  Flame,
  Image,
  LoaderCircle,
  Maximize2,
  Pill,
  Play,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Trash2,
  UploadCloud,
  UserRoundPlus,
  X,
} from "@lucide/vue";
import {
  ApiError,
  createPatientCase,
  diagnoseImage,
  getRuntimeInfo,
  interpretWithAgent,
  listKnowledgeEntries,
  listPatientCases,
  seedSamplePatientCases,
  streamInterpretWithAgent,
} from "./api";
import type {
  AgentInterpretationResponse,
  DiagnosisPayload,
  KnowledgeEntry,
  PatientCase,
  PatientCaseInput,
  RuntimeInfo,
} from "./types";

type ViewKey = "import" | "records" | "diagnosis" | "knowledge" | "agent";

const navigationItems: Array<{ key: ViewKey; label: string; description: string; icon: unknown }> = [
  { key: "import", label: "病人病例导入", description: "录入基础病史和检查信息", icon: UserRoundPlus },
  { key: "records", label: "病人病例查看", description: "检索和查看已导入病例", icon: Files },
  { key: "diagnosis", label: "乳腺超声肿瘤良恶性分类辅助诊断系统", description: "上传乳腺超声图像，生成模型辅助结果", icon: Stethoscope },
  { key: "knowledge", label: "乳腺肿瘤相关知识库", description: "BI-RADS 和风险评估知识", icon: BookOpenText },
  { key: "agent", label: "肿瘤 AI 智能体医学解读", description: "生成面向医生的结构化解释", icon: Bot },
];

marked.setOptions({
  breaks: true,
  gfm: true,
});

const agentEmptyPrompt =
  "选择病例、完成图像模型分析或输入医生问题后，点击“生成辅助建议”。系统会调用 DeepSeek V4 Pro 生成结构化医学解读；未配置 API key 时会返回本地兜底草稿。";

const runtime = ref<RuntimeInfo | null>(null);
const selectedFile = ref<File | null>(null);
const previewUrl = ref<string | null>(null);
const diagnosis = ref<DiagnosisPayload | null>(null);
const threshold = ref(0.51);
const needSegmentation = ref(true);
const needExplanation = ref(true);
const loading = ref(false);
const runtimeLoading = ref(true);
const errorMessage = ref<string | null>(null);
const dragActive = ref(false);
const fileInput = ref<HTMLInputElement | null>(null);
const enlargedImage = ref<{ src: string; title: string; detail: string } | null>(null);
const activeView = ref<ViewKey>("diagnosis");

const patientCases = ref<PatientCase[]>([]);
const patientLoading = ref(false);
const patientError = ref<string | null>(null);
const patientSearch = ref("");
const selectedCaseId = ref<number | null>(null);
const importSaving = ref(false);
const sampleImporting = ref(false);
const importMessage = ref<string | null>(null);
const patientForm = ref<PatientCaseInput>({
  patient_code: "",
  name: "",
  sex: "女",
  age: null,
  contact: "",
  visit_date: new Date().toISOString().slice(0, 10),
  department: "乳腺外科",
  primary_complaint: "",
  ultrasound_description: "",
  birads_category: "",
  pathology_status: "待完善",
  risk_level: "待评估",
  notes: "",
});

const agentQuestion = ref("");
const agentLoading = ref(false);
const agentError = ref<string | null>(null);
const agentResponse = ref<AgentInterpretationResponse | null>(null);
const agentStreamingAnswer = ref("");
const knowledgeEntries = ref<KnowledgeEntry[]>([]);
const knowledgeLoading = ref(false);
const knowledgeError = ref<string | null>(null);
const knowledgeSearch = ref("");
const activeKnowledgeCategory = ref("全部");
const selectedKnowledgeEntry = ref<KnowledgeEntry | null>(null);

const malignantPercent = computed(() =>
  diagnosis.value?.result ? diagnosis.value.result.malignant_probability * 100 : 0,
);
const benignPercent = computed(() =>
  diagnosis.value?.result ? diagnosis.value.result.benign_probability * 100 : 0,
);
const hasResult = computed(() => Boolean(diagnosis.value?.result));
const verdictClass = computed(() =>
  diagnosis.value?.result?.final_label === "malignant" ? "danger" : "safe",
);
const primaryImage = computed(() => diagnosis.value?.images.original ?? previewUrl.value);
const selectedPatientCase = computed(
  () => patientCases.value.find((patientCase) => patientCase.id === selectedCaseId.value) ?? null,
);
const highRiskCaseCount = computed(
  () => patientCases.value.filter((patientCase) => patientCase.risk_level.includes("高")).length,
);
const activeNavigationItem = computed(
  () => navigationItems.find((item) => item.key === activeView.value) ?? navigationItems[0],
);
const apiOnline = computed(() => Boolean(runtime.value) && !runtimeLoading.value);
const knowledgeCategories = computed(() => [
  "全部",
  ...Array.from(new Set(knowledgeEntries.value.map((entry) => entry.category))),
]);
const visibleKnowledgeEntries = computed(() => {
  if (activeKnowledgeCategory.value === "全部") {
    return knowledgeEntries.value;
  }
  return knowledgeEntries.value.filter((entry) => entry.category === activeKnowledgeCategory.value);
});
const selectedCaseSummary = computed(() => {
  if (!selectedPatientCase.value) {
    return "未选择病例";
  }
  return `${selectedPatientCase.value.name} · ${selectedPatientCase.value.patient_code} · ${selectedPatientCase.value.risk_level}`;
});
const agentStatusText = computed(() => {
  if (!agentResponse.value) {
    return "等待生成";
  }
  return agentResponse.value.fallback_used ? "本地兜底草稿" : `DeepSeek ${agentResponse.value.model}`;
});
const agentOutputMarkdown = computed(() => {
  const source = agentResponse.value?.answer?.trim() || agentEmptyPrompt;
  return DOMPurify.sanitize(marked.parse(source, { async: false }));
});

function openImageViewer(src: string | null | undefined, title: string, detail: string | null | undefined): void {
  if (!src) {
    return;
  }
  enlargedImage.value = {
    src,
    title,
    detail: detail || "图像预览",
  };
}

function closeImageViewer(): void {
  enlargedImage.value = null;
}

function openKnowledgeDetail(entry: KnowledgeEntry): void {
  selectedKnowledgeEntry.value = entry;
}

function closeKnowledgeDetail(): void {
  selectedKnowledgeEntry.value = null;
}

function revokePreview(): void {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
    previewUrl.value = null;
  }
}

function setSelectedFile(file: File): void {
  revokePreview();
  selectedFile.value = file;
  previewUrl.value = URL.createObjectURL(file);
  diagnosis.value = null;
  errorMessage.value = null;
}

function handleFiles(files: FileList | null): void {
  const file = files?.item(0);
  if (!file) {
    return;
  }
  setSelectedFile(file);
}

function clearWorkspace(): void {
  revokePreview();
  selectedFile.value = null;
  diagnosis.value = null;
  errorMessage.value = null;
  dragActive.value = false;
  enlargedImage.value = null;
  if (fileInput.value) {
    fileInput.value.value = "";
  }
}

async function runDiagnosis(): Promise<void> {
  if (!selectedFile.value) {
    errorMessage.value = "请先上传一张乳腺超声图像。";
    return;
  }
  loading.value = true;
  errorMessage.value = null;
  try {
    diagnosis.value = await diagnoseImage({
      file: selectedFile.value,
      threshold: threshold.value,
      needSegmentation: needSegmentation.value,
      needExplanation: needExplanation.value,
    });
  } catch (error) {
    errorMessage.value =
      error instanceof ApiError || error instanceof Error
        ? error.message
        : "诊断请求失败，请检查后端服务。";
  } finally {
    loading.value = false;
  }
}

function onDrop(event: DragEvent): void {
  dragActive.value = false;
  handleFiles(event.dataTransfer?.files ?? null);
}

function emptyPatientForm(): PatientCaseInput {
  return {
    patient_code: "",
    name: "",
    sex: "女",
    age: null,
    contact: "",
    visit_date: new Date().toISOString().slice(0, 10),
    department: "乳腺外科",
    primary_complaint: "",
    ultrasound_description: "",
    birads_category: "",
    pathology_status: "待完善",
    risk_level: "待评估",
    notes: "",
  };
}

async function refreshPatientCases(): Promise<void> {
  patientLoading.value = true;
  patientError.value = null;
  try {
    patientCases.value = await listPatientCases(patientSearch.value);
    if (!selectedCaseId.value && patientCases.value.length) {
      selectedCaseId.value = patientCases.value[0].id;
    }
  } catch (error) {
    patientError.value =
      error instanceof ApiError || error instanceof Error ? error.message : "病例列表读取失败。";
  } finally {
    patientLoading.value = false;
  }
}

async function ensureSampleCasesVisible(): Promise<void> {
  await refreshPatientCases();
  if (patientCases.value.length) {
    return;
  }

  const samples = await seedSamplePatientCases();
  await refreshPatientCases();
  if (!selectedCaseId.value && samples.length) {
    selectedCaseId.value = samples[0].id;
  }
}

async function refreshKnowledgeEntries(): Promise<void> {
  knowledgeLoading.value = true;
  knowledgeError.value = null;
  try {
    knowledgeEntries.value = await listKnowledgeEntries({
      query: knowledgeSearch.value,
      category: activeKnowledgeCategory.value === "全部" ? "" : activeKnowledgeCategory.value,
    });
    if (
      activeKnowledgeCategory.value !== "全部" &&
      !knowledgeEntries.value.some((entry) => entry.category === activeKnowledgeCategory.value)
    ) {
      activeKnowledgeCategory.value = "全部";
    }
  } catch (error) {
    knowledgeError.value =
      error instanceof ApiError || error instanceof Error ? error.message : "知识库读取失败。";
  } finally {
    knowledgeLoading.value = false;
  }
}

async function submitPatientCase(): Promise<void> {
  importSaving.value = true;
  importMessage.value = null;
  patientError.value = null;
  try {
    const created = await createPatientCase(patientForm.value);
    importMessage.value = `病例 ${created.patient_code} 已导入。`;
    patientForm.value = emptyPatientForm();
    await refreshPatientCases();
    selectedCaseId.value = created.id;
    activeView.value = "records";
  } catch (error) {
    patientError.value =
      error instanceof ApiError || error instanceof Error ? error.message : "病例导入失败。";
  } finally {
    importSaving.value = false;
  }
}

async function loadSampleCases(): Promise<void> {
  sampleImporting.value = true;
  importMessage.value = null;
  patientError.value = null;
  try {
    const samples = await seedSamplePatientCases();
    await refreshPatientCases();
    if (samples.length) {
      selectedCaseId.value = samples[0].id;
    }
    importMessage.value = `已载入 ${samples.length} 份示例病例。`;
    activeView.value = "records";
  } catch (error) {
    patientError.value =
      error instanceof ApiError || error instanceof Error ? error.message : "示例病例载入失败。";
  } finally {
    sampleImporting.value = false;
  }
}

function selectCase(patientCase: PatientCase): void {
  selectedCaseId.value = patientCase.id;
}

async function runAgentInterpretation(): Promise<void> {
  agentLoading.value = true;
  agentError.value = null;
  agentStreamingAnswer.value = "";
  agentResponse.value = {
    answer: "",
    model: "deepseek-v4-pro",
    provider: "deepseek",
    fallback_used: false,
    retrieved_knowledge: [],
    warnings: [],
  };
  try {
    const payload = {
      question: agentQuestion.value,
      patient_case: selectedPatientCase.value,
      diagnosis: diagnosis.value,
    };
    const streamed = await streamInterpretWithAgent(payload, {
      onMetadata(metadata) {
        agentResponse.value = {
          answer: agentStreamingAnswer.value,
          ...metadata,
        };
      },
      onDelta(text) {
        agentStreamingAnswer.value += text;
        if (agentResponse.value) {
          agentResponse.value.answer = agentStreamingAnswer.value;
        }
      },
      onComplete(response) {
        agentResponse.value = response;
        agentStreamingAnswer.value = response.answer;
      },
      onFallback(response) {
        agentResponse.value = response;
        agentStreamingAnswer.value = response.answer;
      },
    });
    if (!streamed && !agentResponse.value?.answer) {
      agentResponse.value = await interpretWithAgent(payload);
      agentStreamingAnswer.value = agentResponse.value.answer;
    }
  } catch (error) {
    agentError.value =
      error instanceof ApiError || error instanceof Error ? error.message : "智能体解读请求失败。";
  } finally {
    agentLoading.value = false;
  }
}

onMounted(async () => {
  runtimeLoading.value = true;
  try {
    runtime.value = await getRuntimeInfo();
    threshold.value = runtime.value.threshold.default;
    needSegmentation.value = runtime.value.segmentation_enabled;
    needExplanation.value = runtime.value.gradcam_enabled;
  } catch (error) {
    errorMessage.value =
      error instanceof ApiError || error instanceof Error
        ? error.message
        : "无法连接后端服务。";
  } finally {
    runtimeLoading.value = false;
  }
  try {
    await ensureSampleCasesVisible();
  } catch (error) {
    patientError.value =
      error instanceof ApiError || error instanceof Error ? error.message : "示例病例初始化失败。";
  }
  await refreshKnowledgeEntries();
});

onBeforeUnmount(revokePreview);
</script>

<template>
  <main class="platform-shell">
    <aside class="platform-sidebar" aria-label="平台导航">
      <div class="platform-brand">
        <div class="brand-mark" aria-hidden="true">
          <Activity :size="26" />
        </div>
        <div>
          <strong>乳腺肿瘤风险辅助评估平台</strong>
          <span>Hospital BUCAD Platform</span>
        </div>
      </div>

      <section class="platform-kpi-strip" aria-label="平台状态概览">
        <div>
          <span>病例总数</span>
          <strong>{{ patientCases.length }}</strong>
        </div>
        <div>
          <span>高风险</span>
          <strong>{{ highRiskCaseCount }}</strong>
        </div>
        <div>
          <span>知识条目</span>
          <strong>{{ knowledgeEntries.length }}</strong>
        </div>
        <div>
          <span>智能体</span>
          <strong>{{ agentStatusText }}</strong>
        </div>
      </section>

      <nav class="module-list">
        <button
          v-for="item in navigationItems"
          :key="item.key"
          class="module-button"
          :class="{ active: activeView === item.key }"
          type="button"
          @click="activeView = item.key"
        >
          <component :is="item.icon" :size="20" />
          <span>
            <strong>{{ item.label }}</strong>
            <small>{{ item.description }}</small>
          </span>
        </button>
      </nav>

      <div class="sidebar-status">
        <span :class="{ pending: runtimeLoading, offline: !runtimeLoading && !apiOnline }"></span>
        <div>
          <strong>{{ runtimeLoading ? "服务连接中" : apiOnline ? "本地服务就绪" : "本地服务未连接" }}</strong>
          <small>{{ patientCases.length }} 份病例 · {{ highRiskCaseCount }} 份高风险</small>
        </div>
      </div>
    </aside>

    <section class="platform-main">
      <header class="platform-header">
        <div class="title-group">
          <h1>{{ activeNavigationItem.label }}</h1>
        </div>
        <div class="runtime-pill" :class="{ pending: runtimeLoading, offline: !runtimeLoading && !apiOnline }">
          <LoaderCircle v-if="runtimeLoading" :size="16" class="spin" />
          <CheckCircle2 v-else-if="apiOnline" :size="16" />
          <AlertCircle v-else :size="16" />
          <span>{{ runtimeLoading ? "连接中" : apiOnline ? "API 已连接" : "API 未连接" }}</span>
        </div>
      </header>

      <section v-if="activeView === 'import'" class="module-surface">
        <div class="module-toolbar">
          <div>
            <h2>导入病人病例</h2>
            <p>录入患者基础信息、超声描述、BI-RADS 分类和病理状态。</p>
          </div>
          <button class="secondary-action" type="button" :disabled="sampleImporting" @click="loadSampleCases">
            <LoaderCircle v-if="sampleImporting" :size="18" class="spin" />
            <Database v-else :size="18" />
            <span>{{ sampleImporting ? "载入中" : "载入示例病例" }}</span>
          </button>
        </div>

        <form class="case-form" @submit.prevent="submitPatientCase">
          <label>
            <span>病例编号</span>
            <input v-model.trim="patientForm.patient_code" required placeholder="例如 BUCAD-2026-001" />
          </label>
          <label>
            <span>姓名</span>
            <input v-model.trim="patientForm.name" required placeholder="患者姓名" />
          </label>
          <label>
            <span>性别</span>
            <select v-model="patientForm.sex">
              <option>女</option>
              <option>男</option>
              <option>未填写</option>
            </select>
          </label>
          <label>
            <span>年龄</span>
            <input v-model.number="patientForm.age" min="0" max="130" type="number" placeholder="年龄" />
          </label>
          <label>
            <span>联系方式</span>
            <input v-model.trim="patientForm.contact" placeholder="手机号或院内联系方式" />
          </label>
          <label>
            <span>就诊日期</span>
            <input v-model="patientForm.visit_date" type="date" />
          </label>
          <label>
            <span>科室</span>
            <input v-model.trim="patientForm.department" />
          </label>
          <label>
            <span>BI-RADS</span>
            <input v-model.trim="patientForm.birads_category" placeholder="例如 3 / 4A / 4B / 5" />
          </label>
          <label>
            <span>病理状态</span>
            <select v-model="patientForm.pathology_status">
              <option>待完善</option>
              <option>良性</option>
              <option>恶性</option>
              <option>随访中</option>
            </select>
          </label>
          <label>
            <span>风险等级</span>
            <select v-model="patientForm.risk_level">
              <option>待评估</option>
              <option>低风险</option>
              <option>中风险</option>
              <option>高风险</option>
            </select>
          </label>
          <label class="span-2">
            <span>主诉</span>
            <textarea v-model.trim="patientForm.primary_complaint" rows="3" placeholder="例如 右乳触及肿块 2 周。"></textarea>
          </label>
          <label class="span-2">
            <span>超声描述</span>
            <textarea
              v-model.trim="patientForm.ultrasound_description"
              rows="4"
              placeholder="记录肿块位置、大小、边界、形态、回声、血流等。"
            ></textarea>
          </label>
          <label class="span-2">
            <span>备注</span>
            <textarea v-model.trim="patientForm.notes" rows="3" placeholder="补充家族史、既往史、随访计划等。"></textarea>
          </label>

          <div class="form-actions span-2">
            <button class="primary-action" type="submit" :disabled="importSaving">
              <LoaderCircle v-if="importSaving" :size="18" class="spin" />
              <ClipboardList v-else :size="18" />
              <span>{{ importSaving ? "导入中" : "导入病例" }}</span>
            </button>
            <button class="secondary-action" type="button" @click="patientForm = emptyPatientForm()">清空表单</button>
          </div>
          <p v-if="importMessage" class="success-text span-2">{{ importMessage }}</p>
          <p v-if="patientError" class="error-text span-2">{{ patientError }}</p>
        </form>
      </section>

      <section v-else-if="activeView === 'records'" class="module-surface records-view">
        <div class="module-toolbar">
          <div>
            <h2>病人病例查看</h2>
            <p>检索本地 SQLite 病例库，查看患者病史、影像描述和风险等级。</p>
          </div>
          <label class="search-box">
            <Search :size="18" />
            <input v-model="patientSearch" placeholder="搜索编号、姓名、BI-RADS 或风险等级" @keyup.enter="refreshPatientCases" />
          </label>
          <button class="secondary-action" type="button" @click="refreshPatientCases">查询</button>
        </div>

        <div class="records-layout">
          <div class="record-list">
            <button
              v-for="patientCase in patientCases"
              :key="patientCase.id"
              class="record-item"
              :class="{ active: selectedCaseId === patientCase.id }"
              type="button"
              @click="selectCase(patientCase)"
            >
              <strong>{{ patientCase.name }} · {{ patientCase.patient_code }}</strong>
              <span>{{ patientCase.visit_date || "未填写日期" }} · {{ patientCase.risk_level }}</span>
            </button>
            <div v-if="patientLoading" class="empty-state">病例读取中。</div>
            <div v-else-if="!patientCases.length" class="empty-state">暂无病例，请先导入病人病例。</div>
          </div>

          <article class="record-detail">
            <template v-if="selectedPatientCase">
              <header>
                <div>
                  <h3>{{ selectedPatientCase.name }}</h3>
                  <p>{{ selectedPatientCase.patient_code }} · {{ selectedPatientCase.sex }} · {{ selectedPatientCase.age ?? "年龄未填" }}岁</p>
                </div>
                <span class="risk-badge">{{ selectedPatientCase.risk_level }}</span>
              </header>
              <dl class="case-detail-grid">
                <div><dt>就诊日期</dt><dd>{{ selectedPatientCase.visit_date || "未填写" }}</dd></div>
                <div><dt>科室</dt><dd>{{ selectedPatientCase.department || "未填写" }}</dd></div>
                <div><dt>BI-RADS</dt><dd>{{ selectedPatientCase.birads_category || "未填写" }}</dd></div>
                <div><dt>病理状态</dt><dd>{{ selectedPatientCase.pathology_status }}</dd></div>
              </dl>
              <section>
                <h4>主诉</h4>
                <p>{{ selectedPatientCase.primary_complaint || "未填写" }}</p>
              </section>
              <section>
                <h4>超声描述</h4>
                <p>{{ selectedPatientCase.ultrasound_description || "未填写" }}</p>
              </section>
              <section>
                <h4>备注</h4>
                <p>{{ selectedPatientCase.notes || "未填写" }}</p>
              </section>
            </template>
            <div v-else class="empty-state">请选择左侧病例。</div>
          </article>
        </div>
      </section>

      <section v-else-if="activeView === 'diagnosis'" class="diagnosis-module" aria-label="辅助诊断工作区">
        <aside class="control-panel">
          <section class="panel-section">
            <div class="section-heading">
              <UploadCloud :size="18" />
              <span>图像输入与辅助诊断设置</span>
            </div>

            <label
              class="drop-zone"
              :class="{ active: dragActive, filled: selectedFile }"
              @dragenter.prevent="dragActive = true"
              @dragover.prevent="dragActive = true"
              @dragleave.prevent="dragActive = false"
              @drop.prevent="onDrop"
            >
              <input
                ref="fileInput"
                type="file"
                accept=".png,.jpg,.jpeg,.bmp,image/png,image/jpeg,image/bmp"
                @change="handleFiles(($event.target as HTMLInputElement).files)"
              />
              <FileImage :size="34" />
              <span class="upload-title">
                {{ selectedFile ? selectedFile.name : "点击或拖拽上传乳腺超声图像" }}
              </span>
              <span class="upload-meta">支持 JPG / PNG / BMP，DICOM 需先转换为普通图像。</span>
            </label>

            <div class="field-block">
              <div class="field-row">
                <label for="threshold">恶性判定阈值</label>
                <strong>{{ threshold.toFixed(3) }}</strong>
              </div>
              <input
                id="threshold"
                v-model.number="threshold"
                type="range"
                :min="runtime?.threshold.min ?? 0.1"
                :max="runtime?.threshold.max ?? 0.9"
                :step="runtime?.threshold.step ?? 0.001"
              />
              <p class="field-hint">
                当前主线推荐 {{ (runtime?.threshold.default ?? 0.51).toFixed(3) }}；ROI 使用 0.40 mask 阈值、最大连通域与面积质量门控。
              </p>
            </div>

            <div class="toggle-row">
              <label class="toggle-item">
                <input v-model="needSegmentation" type="checkbox" />
                <span>生成病灶定位图</span>
              </label>
              <label class="toggle-item">
                <input v-model="needExplanation" type="checkbox" />
                <span>生成 Grad-CAM 热力图</span>
              </label>
            </div>

            <div class="action-row">
              <button class="primary-action" type="button" :disabled="loading" @click="runDiagnosis">
                <LoaderCircle v-if="loading" :size="18" class="spin" />
                <Play v-else :size="18" />
                <span>{{ loading ? "分析中" : "开始辅助诊断" }}</span>
              </button>
              <button class="secondary-action" type="button" @click="clearWorkspace">
                <Trash2 :size="18" />
                <span>清空</span>
              </button>
            </div>
          </section>

          <section class="result-card" :class="verdictClass">
            <div class="section-heading">
              <ShieldCheck :size="18" />
              <span>辅助诊断结果</span>
            </div>

            <div v-if="hasResult" class="metric-grid">
              <div class="metric-cell danger-soft">
                <span>恶性概率</span>
                <strong>{{ malignantPercent.toFixed(1) }}%</strong>
                <div class="meter danger-meter">
                  <i :style="{ width: `${malignantPercent}%` }"></i>
                </div>
              </div>
              <div class="metric-cell safe-soft">
                <span>良性概率</span>
                <strong>{{ benignPercent.toFixed(1) }}%</strong>
                <div class="meter safe-meter">
                  <i :style="{ width: `${benignPercent}%` }"></i>
                </div>
              </div>
              <div class="metric-cell verdict-cell">
                <span>模型判定</span>
                <strong>{{ diagnosis?.result?.final_label_text }}</strong>
                <em>{{ diagnosis?.result?.confidence_band_text }}置信度</em>
              </div>
            </div>

            <div v-else class="empty-state">上传图像并点击“开始辅助诊断”后显示模型结果。</div>

            <div v-if="hasResult" class="result-strip">
              <span>阈值 <b>{{ threshold.toFixed(3) }}</b></span>
              <span>状态 <b>{{ diagnosis?.status_text }}</b></span>
              <span class="model-name">模型 <b>{{ diagnosis?.result?.model_display_name }}</b></span>
            </div>

            <p v-if="hasResult" class="recommendation">
              {{ diagnosis?.result?.recommendation_text }}
            </p>
          </section>

          <section class="status-card">
            <div class="section-heading">
              <AlertCircle :size="18" />
              <span>提示信息</span>
            </div>
            <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>
            <ul v-else-if="diagnosis?.warnings.length" class="warning-list">
              <li v-for="warning in diagnosis.warnings" :key="warning">{{ warning }}</li>
            </ul>
            <p v-else class="hint-text">
              {{ runtime?.disclaimer ?? "本系统仅用于辅助分析和原型演示，不能替代医生诊断。" }}
            </p>
          </section>
        </aside>

        <section class="viewer-panel">
          <div class="viewer-header">
            <div>
              <div class="section-heading">
                <Image :size="18" />
                <span>图像分析结果</span>
              </div>
              <p>展示原图、病灶定位叠加图与 Grad-CAM 热力图，辅助医生复核模型关注区域。</p>
            </div>
            <button class="ghost-action" type="button" @click="clearWorkspace">
              <RotateCcw :size="17" />
              <span>重置视图</span>
            </button>
          </div>

          <div class="image-grid">
            <article class="image-card primary-image">
              <div class="image-title">
                <span>原始超声图像</span>
                <b>{{ selectedFile?.name ?? "未上传" }}</b>
              </div>
              <div class="image-frame">
                <button
                  class="image-zoom-button"
                  type="button"
                  :disabled="!primaryImage"
                  aria-label="放大原始超声图像"
                  title="放大"
                  @click="openImageViewer(primaryImage, '原始超声图像', selectedFile?.name ?? '未上传')"
                >
                  <Maximize2 :size="17" />
                </button>
                <img v-if="primaryImage" :src="primaryImage" alt="原始超声图像" />
                <div v-else class="empty-visual">
                  <FileImage :size="46" />
                  <span>等待图像</span>
                </div>
              </div>
            </article>

            <div class="mini-image-row">
              <article class="image-card">
                <div class="image-title">
                  <span>病灶定位叠加图</span>
                  <b>{{ diagnosis?.lesion_visualization_missing_reason ?? "Mask Overlay" }}</b>
                </div>
                <div class="image-frame">
                  <button
                    class="image-zoom-button"
                    type="button"
                    :disabled="!diagnosis?.images.lesion"
                    aria-label="放大病灶定位叠加图"
                    title="放大"
                    @click="
                      openImageViewer(
                        diagnosis?.images.lesion,
                        '病灶定位叠加图',
                        diagnosis?.lesion_visualization_missing_reason ?? 'Mask Overlay',
                      )
                    "
                  >
                    <Maximize2 :size="17" />
                  </button>
                  <img v-if="diagnosis?.images.lesion" :src="diagnosis.images.lesion" alt="病灶定位叠加图" />
                  <div v-else class="empty-visual small">
                    <ShieldCheck :size="36" />
                    <span>等待输出</span>
                  </div>
                </div>
              </article>

              <article class="image-card">
                <div class="image-title">
                  <span>模型关注区域（Grad-CAM）</span>
                  <b>{{ diagnosis?.explanation_missing_reason ?? "Grad-CAM" }}</b>
                </div>
                <div class="image-frame">
                  <button
                    class="image-zoom-button"
                    type="button"
                    :disabled="!diagnosis?.images.explanation"
                    aria-label="放大模型关注区域 Grad-CAM"
                    title="放大"
                    @click="
                      openImageViewer(
                        diagnosis?.images.explanation,
                        '模型关注区域（Grad-CAM）',
                        diagnosis?.explanation_missing_reason ?? 'Grad-CAM',
                      )
                    "
                  >
                    <Maximize2 :size="17" />
                  </button>
                  <img
                    v-if="diagnosis?.images.explanation"
                    :src="diagnosis.images.explanation"
                    alt="模型关注区域 Grad-CAM"
                  />
                  <div v-else class="empty-visual small">
                    <Flame :size="36" />
                    <span>等待输出</span>
                  </div>
                </div>
              </article>
            </div>
          </div>

          <footer class="viewer-note">
            病灶定位图和热力图用于辅助理解模型输出，不代表临床标注、病理结论或最终诊断。
          </footer>
        </section>
      </section>

      <section v-else-if="activeView === 'knowledge'" class="module-surface knowledge-view">
        <div class="module-toolbar">
          <div>
            <h2>乳腺肿瘤相关知识库</h2>
            <p>内置 BI-RADS、病理、分子标志物、药物治疗和影像病理一致性知识，用于智能体检索引用。</p>
          </div>
          <label class="search-box">
            <Search :size="18" />
            <input v-model="knowledgeSearch" placeholder="搜索 HER2、三阴性、穿刺、BI-RADS 等" @keyup.enter="refreshKnowledgeEntries" />
          </label>
          <button class="secondary-action" type="button" @click="refreshKnowledgeEntries">检索</button>
        </div>

        <div class="knowledge-categories">
          <button
            v-for="category in knowledgeCategories"
            :key="category"
            type="button"
            :class="{ active: activeKnowledgeCategory === category }"
            @click="activeKnowledgeCategory = category"
          >
            {{ category }}
          </button>
        </div>

        <div v-if="knowledgeError" class="error-text">{{ knowledgeError }}</div>
        <div v-else-if="knowledgeLoading" class="empty-state">知识库读取中。</div>
        <div v-else class="knowledge-list">
          <article
            v-for="entry in visibleKnowledgeEntries"
            :key="entry.id"
            class="knowledge-card"
            role="button"
            tabindex="0"
            :aria-label="`查看知识点详情：${entry.title}`"
            @click="openKnowledgeDetail(entry)"
            @keydown.enter.prevent="openKnowledgeDetail(entry)"
            @keydown.space.prevent="openKnowledgeDetail(entry)"
          >
            <header>
              <span>{{ entry.category }}</span>
              <h2>{{ entry.title }}</h2>
            </header>
            <p>{{ entry.summary }}</p>
            <div class="clinical-use">
              <strong>临床用途</strong>
              <span>{{ entry.clinical_use }}</span>
            </div>
            <div class="keyword-row">
              <small v-for="keyword in entry.keywords" :key="keyword">{{ keyword }}</small>
            </div>
            <div class="knowledge-card-action">点击查看详细内容</div>
          </article>
        </div>
      </section>

      <section v-else class="module-surface agent-view">
        <div class="module-toolbar">
          <div>
            <h2>肿瘤 AI 智能体医学解读</h2>
            <p>调用 DeepSeek V4 Pro 智能体流程，生成规范化辅助诊断建议。</p>
          </div>
          <Sparkles :size="30" />
        </div>

        <div class="agent-workbench">
          <section class="agent-context">
            <div class="section-heading">
              <ClipboardList :size="18" />
              <span>当前上下文</span>
            </div>
            <dl>
              <div>
                <dt>选中病例</dt>
                <dd>{{ selectedCaseSummary }}</dd>
              </div>
              <div>
                <dt>图像模型结果</dt>
                <dd>
                  <template v-if="diagnosis?.result">
                    {{ diagnosis.result.final_label_text }} · 恶性概率
                    {{ (diagnosis.result.malignant_probability * 100).toFixed(1) }}%
                  </template>
                  <template v-else>未完成图像模型分析</template>
                </dd>
              </div>
            </dl>
            <label class="agent-input">
              <span>医生关注问题</span>
              <textarea v-model="agentQuestion" rows="6" placeholder="例如：请解释该病例为什么需要进一步穿刺活检，并列出还缺哪些病理信息。"></textarea>
            </label>
            <button class="primary-action" type="button" :disabled="agentLoading" @click="runAgentInterpretation">
              <LoaderCircle v-if="agentLoading" :size="18" class="spin" />
              <Bot v-else :size="18" />
              <span>{{ agentLoading ? "生成中" : "生成辅助建议" }}</span>
            </button>
            <p v-if="agentError" class="error-text">{{ agentError }}</p>
          </section>

          <section class="agent-output-panel">
            <div class="agent-output-header">
              <div class="section-heading">
                <Bot :size="18" />
                <span>结构化辅助建议</span>
              </div>
              <span :class="{ fallback: agentResponse?.fallback_used }">
                {{ agentStatusText }}
              </span>
            </div>
            <div class="agent-output markdown-body" v-html="agentOutputMarkdown"></div>
          </section>
        </div>
      </section>
    </section>

    <Teleport to="body">
      <div
        v-if="selectedKnowledgeEntry"
        class="knowledge-detail-overlay"
        role="dialog"
        aria-modal="true"
        :aria-label="`${selectedKnowledgeEntry.title}知识点详情`"
        @click.self="closeKnowledgeDetail"
      >
        <section class="knowledge-detail-dialog">
          <header class="knowledge-detail-header">
            <div>
              <span>{{ selectedKnowledgeEntry.category }}</span>
              <h2>{{ selectedKnowledgeEntry.title }}</h2>
              <p>{{ selectedKnowledgeEntry.summary }}</p>
            </div>
            <button class="image-viewer-close" type="button" aria-label="关闭知识点详情" @click="closeKnowledgeDetail">
              <X :size="20" />
            </button>
          </header>

          <div class="knowledge-detail-body">
            <section>
              <h3>核心要点</h3>
              <ul>
                <li v-for="detail in selectedKnowledgeEntry.details" :key="detail">{{ detail }}</li>
              </ul>
            </section>
            <section>
              <h3>临床用途</h3>
              <p>{{ selectedKnowledgeEntry.clinical_use }}</p>
            </section>
            <section>
              <h3>关键词</h3>
              <div class="keyword-row">
                <small v-for="keyword in selectedKnowledgeEntry.keywords" :key="keyword">{{ keyword }}</small>
              </div>
            </section>
            <section v-if="selectedKnowledgeEntry.references.length">
              <h3>参考来源</h3>
              <ul>
                <li v-for="reference in selectedKnowledgeEntry.references" :key="reference">{{ reference }}</li>
              </ul>
            </section>
          </div>
        </section>
      </div>

      <div
        v-if="enlargedImage"
        class="image-viewer-overlay"
        role="dialog"
        aria-modal="true"
        :aria-label="`${enlargedImage.title}放大预览`"
        @click.self="closeImageViewer"
      >
        <section class="image-viewer-dialog">
          <header class="image-viewer-header">
            <div>
              <strong>{{ enlargedImage.title }}</strong>
              <span>{{ enlargedImage.detail }}</span>
            </div>
            <button class="image-viewer-close" type="button" aria-label="关闭放大预览" @click="closeImageViewer">
              <X :size="20" />
            </button>
          </header>
          <div class="image-viewer-frame">
            <img :src="enlargedImage.src" :alt="`${enlargedImage.title}放大预览`" />
          </div>
        </section>
      </div>
    </Teleport>
  </main>
</template>
