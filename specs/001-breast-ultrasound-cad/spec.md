# Feature Specification: 乳腺超声辅助诊断系统

**Feature Branch**: `[001-breast-ultrasound-cad]`  
**Created**: 2026-04-23  
**Status**: Draft  
**Input**: User description: "构建一个面向乳腺超声图像的智能辅助诊断系统，整合良恶性判断、病灶区域可视化和辅助解释信息，形成可直接演示和交付的软件原型。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 单图诊断输出 (Priority: P1)

作为使用者，我希望上传一张乳腺超声图像后，系统能给出良性概率、恶性概率和最终判断结果，
从而快速获得一个稳定、可重复的辅助诊断参考。

**Why this priority**: 这是系统最核心的输出，也是“是否值得继续关注该病例”的第一判断依据。
即使只完成这一条，系统也已经具备基础的辅助诊断价值。

**Independent Test**: 上传一张有效的乳腺超声图像，系统能够返回良性概率、恶性概率和明确的最终
判断，且结果页面对非开发人员可直接理解。

**Acceptance Scenarios**:

1. **Given** 使用者上传一张有效的乳腺超声图像，**When** 系统完成分析，
   **Then** 系统显示恶性概率、良性概率和最终判定结果。
2. **Given** 图像可被读取但风险判断接近边界，**When** 系统生成结果，
   **Then** 系统明确标记该结果为低置信度或建议人工复核，而不是给出模糊结论。

---

### User Story 2 - 病灶与解释可视化 (Priority: P2)

作为使用者，我希望系统除了给出结论外，还能把可疑病灶区域和模型主要关注区域直观展示出来，
从而理解“问题大概在哪里”以及“系统为什么倾向于这样判断”。

**Why this priority**: 用户对系统的信任不仅来自分类结果，还来自是否能看到与结论对应的图像证据。
这一能力直接影响系统的可解释性、说服力和答辩展示效果。

**Independent Test**: 上传一张有效图像后，结果页面能够同时展示原图、病灶区域可视化结果和解释性
可视化结果，并且这些内容与本次分析图像一一对应。

**Acceptance Scenarios**:

1. **Given** 系统完成图像分析，**When** 使用者查看结果页面，
   **Then** 页面显示病灶位置标注或分割叠加效果，帮助定位可疑区域。
2. **Given** 系统输出最终判断，**When** 使用者查看辅助解释信息，
   **Then** 页面展示热力图或等效的关注区域说明，帮助理解结论依据。

---

### User Story 3 - 可演示的软件化流程 (Priority: P3)

作为比赛评审、指导教师或项目演示人员，我希望系统以可操作的软件原型形式运行，
这样可以通过“上传图像 -> 自动分析 -> 查看结果”的完整流程展示项目价值，而不是只展示模型分数。

**Why this priority**: 项目目标不仅是训练一个模型，还要交付一个可展示、可讲解、可操作的辅助诊断
成果。这条用户故事决定了项目是否具备完整交付形态。

**Independent Test**: 从打开系统到完成一次图像分析，不依赖开发环境说明即可完成上传、分析和查看
结果的全流程操作。

**Acceptance Scenarios**:

1. **Given** 使用者进入系统主界面，**When** 上传图像并触发分析，
   **Then** 系统在同一流程中返回诊断结论、病灶可视化和解释信息。
2. **Given** 某些附加输出暂时不可用，**When** 使用者执行分析，
   **Then** 系统仍提供基础诊断结果，并清楚说明缺失项和建议处理方式。

### Edge Cases

- 上传的图像不是乳腺超声图像、图像损坏或质量过低，系统必须阻止误导性输出并给出清晰提示。
- 图像中存在多个可疑区域、病灶边界不清或无明确高风险区域时，系统必须以可理解方式展示不确定性。
- 分类概率接近决策边界时，系统必须突出“建议人工复核”而不是把边缘案例伪装成高确定性结论。
- 病灶可视化或解释性图层暂时无法生成时，系统必须保留基础诊断结果并明确说明输出缺失。

## Data & Evaluation Constraints *(mandatory for data/model features)*

- **Training Data Boundary**: 系统训练和内部验证仅使用 BUSBRA 训练数据，并对训练集采用病例级划分，
  防止同一病例信息同时出现在不同阶段。
- **Evaluation Data Boundary**: BUSI 外部测试数据仅用于独立评测、演示验证和结果汇报，不得回流到训练、
  阈值调优或模型选择流程。
- **Leakage Safeguards**: 所有涉及数据划分、评测和结果记录的流程都必须保留防泄漏检查，
  至少覆盖病例隔离、样本数量核对、标签一致性核对和训练/测试边界核对。
- **Required Evidence**: 功能验收必须同时提供分类结果指标、关键可视化示例、完整演示截图或录屏，
  以及一条可复现的单图分析验证流程。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow a user to submit a single breast ultrasound image for analysis.
- **FR-002**: System MUST determine whether the image contains a suspicious lesion candidate or clearly state that no reliable suspicious region was identified.
- **FR-003**: System MUST present malignant probability, benign probability, and a final risk-oriented judgment for the analyzed image.
- **FR-004**: System MUST present the suspected lesion location on the image through a visible region overlay, segmentation result, bounding indication, or equivalent spatial cue.
- **FR-005**: System MUST provide an explanation view tied to the analyzed image, such as a model-attention heatmap, highlighted risk region, or equivalent visual evidence.
- **FR-006**: System MUST present the original image, diagnostic conclusion, lesion visualization, and explanation output together in one reviewable result flow.
- **FR-007**: System MUST explain that the output is an auxiliary reference and not the sole basis for final clinical diagnosis.
- **FR-008**: System MUST provide understandable feedback when the uploaded image is invalid, unsupported, corrupted, or too poor in quality for reliable analysis.
- **FR-009**: System MUST distinguish low-confidence or borderline cases from routine cases and recommend manual review for those results.
- **FR-010**: System MUST make each analysis result reviewable after generation, including the case identifier or filename, analysis time, probabilities, final judgment, and generated visual evidence.

### Key Entities *(include if feature involves data)*

- **Diagnostic Case**: 一次待分析的乳腺超声图像任务，包含输入图像、病例标识信息和本次分析上下文。
- **Diagnostic Result**: 一次分析后生成的结构化结果，包含良恶性概率、最终判断、置信提示和结果说明。
- **Lesion Finding**: 与图像中可疑病灶相关的空间性发现，包含病灶位置、范围或叠加可视化结果。
- **Explanation Artifact**: 用于解释系统判断依据的附加输出，包含关注区域热力图、风险说明和可视化证据。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 在有效输入图像上，90% 以上的单次分析可在 60 秒内完成，并返回完整结果页面。
- **SC-002**: 在约定的独立评测数据上，系统达到不低于 0.75 的 AUC，并对恶性病例保持不低于 0.85 的敏感性阈值目标。
- **SC-003**: 100% 的成功分析结果都同时展示分类结论与至少一种空间可视化证据，或明确说明某项证据为何缺失。
- **SC-004**: 在演示场景下，90% 以上的首次使用者能在 2 分钟内独立完成“上传图像 -> 查看诊断结果 -> 理解重点区域”的完整流程。

## Assumptions

- 目标使用者包括临床辅助阅片者、比赛评审、指导教师和项目演示人员，他们需要的是可理解的辅助判断，而非自动替代医生。
- 本阶段范围聚焦于单张乳腺超声图像的风险判断、病灶可视化和解释展示，不包含治疗建议、病历管理或正式临床决策闭环。
- 输入以常见静态乳腺超声图像为主，每次分析默认面向一个病例图像，不要求一次性处理批量正式诊断任务。
- 对于低质量图像、边界病例或低置信度结果，系统默认以“提示风险并建议人工复核”为标准处理方式。
- BUSBRA 默认作为训练与内部验证来源，BUSI 默认作为外部独立评测来源，二者在验收与迭代中保持边界隔离。
- 项目验收默认同时关注分类性能、可解释展示效果和软件演示完整性，而不是只看单一模型分数。
