"""Curated breast tumor medical knowledge base and lightweight retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from re import findall

from pydantic import BaseModel, Field


class KnowledgeEntry(BaseModel):
    """Public knowledge card returned to the frontend and agent harness."""

    id: str
    category: str
    title: str
    summary: str
    details: list[str]
    clinical_use: str
    keywords: list[str]
    references: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeSearchResult:
    """Internal search result with score."""

    entry: KnowledgeEntry
    score: int


KNOWLEDGE_ENTRIES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        id="birads-risk",
        category="影像风险分层",
        title="BI-RADS 乳腺超声风险分层",
        summary="BI-RADS 将影像发现转化为标准化风险等级，用于决定随访、活检或进一步检查。",
        details=[
            "BI-RADS 3 多为可能良性，常建议短期影像随访，不能单凭 AI 概率升级为恶性诊断。",
            "BI-RADS 4 代表可疑恶性，常细分为 4A、4B、4C，恶性概率从低到高递增，通常需要组织学取材。",
            "BI-RADS 5 高度提示恶性，应优先推动病理确诊和分期评估。",
            "最终处理需结合年龄、体征、既往影像变化、家族史和患者意愿。",
        ],
        clinical_use="将模型恶性概率和 BI-RADS 共同呈现，帮助医生区分随访、穿刺活检和进一步影像检查路径。",
        keywords=["BI-RADS", "4A", "4B", "4C", "5", "风险分层", "随访", "活检"],
        references=[
            "ACR BI-RADS Atlas: breast imaging reporting and data system.",
        ],
    ),
    KnowledgeEntry(
        id="ultrasound-malignant-features",
        category="超声征象",
        title="提示恶性的常见乳腺超声征象",
        summary="形态、边界、纵横比、后方回声和血流征象是乳腺超声风险评估的核心线索。",
        details=[
            "形态不规则、边界不清或毛刺样边缘可提高恶性风险。",
            "纵横比大于 1、非平行生长常提示病灶突破正常组织层面。",
            "后方声影、微钙化、结构扭曲和异常血流可作为风险加权证据。",
            "复杂囊实性结节需关注实性成分、乳头状突起和血流分布。",
        ],
        clinical_use="当 AI 热力图关注区域与可疑超声征象重合时，可提高解释可信度；若关注黑边或标尺区域，应降低模型解释权重。",
        keywords=["超声", "边界", "形态", "纵横比", "声影", "钙化", "血流", "Grad-CAM"],
        references=[
            "ACR BI-RADS Ultrasound lexicon.",
        ],
    ),
    KnowledgeEntry(
        id="pathology-types",
        category="病理知识",
        title="乳腺癌常见病理类型",
        summary="浸润性导管癌和浸润性小叶癌是常见浸润性乳腺癌类型，病理类型会影响影像表现和治疗策略。",
        details=[
            "浸润性导管癌最常见，超声上可表现为不规则低回声肿块、边界不清、后方声影。",
            "浸润性小叶癌可能呈弥漫浸润或影像边界不典型，影像和临床范围评估需更谨慎。",
            "导管原位癌可与钙化、导管改变或非肿块样表现相关，单纯超声可能低估范围。",
            "病理报告需关注组织学类型、分级、脉管侵犯和切缘信息。",
        ],
        clinical_use="智能体在建议中应提示影像诊断不能替代病理类型判定，疑似恶性病灶需组织学确认。",
        keywords=["病理", "浸润性导管癌", "浸润性小叶癌", "导管原位癌", "分级", "切缘"],
        references=[
            "NCI PDQ Breast Cancer Treatment.",
        ],
    ),
    KnowledgeEntry(
        id="biomarkers",
        category="分子标志物",
        title="ER、PR、HER2、Ki-67 与分子分型",
        summary="ER、PR、HER2 和 Ki-67 是乳腺癌治疗决策中的关键标志物。",
        details=[
            "ER 或 PR 阳性提示肿瘤可能从内分泌治疗中获益。",
            "HER2 阳性患者可考虑抗 HER2 靶向治疗方案，需结合 IHC 和 ISH/FISH 等检测结果。",
            "三阴性乳腺癌指 ER、PR、HER2 均阴性，治疗常更依赖化疗、免疫治疗和临床分期。",
            "Ki-67 反映增殖活性，不同机构阈值可能存在差异，应结合病理科标准解释。",
        ],
        clinical_use="用于解释为什么影像风险后仍需穿刺活检和免疫组化，避免把图像模型结果误认为治疗分型。",
        keywords=["ER", "PR", "HER2", "Ki-67", "三阴性", "免疫组化", "分子分型"],
        references=[
            "NCI PDQ Breast Cancer Treatment: hormone receptor and HER2 status.",
        ],
    ),
    KnowledgeEntry(
        id="endocrine-therapy",
        category="药物治疗",
        title="内分泌治疗相关药物",
        summary="激素受体阳性乳腺癌常使用内分泌治疗，具体选择取决于绝经状态、分期和复发风险。",
        details=[
            "他莫昔芬可用于部分绝经前和绝经后激素受体阳性患者。",
            "芳香化酶抑制剂如来曲唑、阿那曲唑、依西美坦常用于绝经后患者或卵巢功能抑制联合方案。",
            "氟维司群可用于部分晚期或转移性激素受体阳性乳腺癌场景。",
            "CDK4/6 抑制剂可与内分泌治疗联用，用于部分 HR 阳性、HER2 阴性晚期乳腺癌。",
        ],
        clinical_use="智能体可提示药物方向，但必须说明治疗方案需肿瘤专科医生基于分期、病理和指南决定。",
        keywords=["内分泌治疗", "他莫昔芬", "来曲唑", "阿那曲唑", "依西美坦", "氟维司群", "CDK4/6"],
        references=[
            "NCI PDQ Breast Cancer Treatment: hormone therapy.",
            "American Cancer Society: hormone therapy for breast cancer.",
        ],
    ),
    KnowledgeEntry(
        id="anti-her2-therapy",
        category="药物治疗",
        title="HER2 靶向治疗相关药物",
        summary="HER2 阳性乳腺癌可从抗 HER2 靶向治疗中获益，常需心脏功能和治疗阶段评估。",
        details=[
            "曲妥珠单抗和帕妥珠单抗是常见抗 HER2 单克隆抗体。",
            "T-DM1、T-DXd 等抗体偶联药物用于特定辅助、晚期或转移性治疗场景。",
            "拉帕替尼、奈拉替尼、图卡替尼等小分子药物可用于部分 HER2 阳性疾病阶段。",
            "HER2 治疗前后通常需要结合心功能、既往治疗和转移部位综合判断。",
        ],
        clinical_use="当病理提示 HER2 阳性时，智能体应建议转入 HER2 靶向治疗路径评估，而不是根据超声图像直接推荐具体方案。",
        keywords=["HER2", "曲妥珠单抗", "帕妥珠单抗", "T-DM1", "T-DXd", "拉帕替尼", "图卡替尼"],
        references=[
            "NCI PDQ Breast Cancer Treatment: HER2-targeted therapy.",
        ],
    ),
    KnowledgeEntry(
        id="triple-negative-and-immunotherapy",
        category="药物治疗",
        title="三阴性乳腺癌、免疫治疗与 PARP 抑制剂",
        summary="三阴性乳腺癌治疗更依赖分期、PD-L1、BRCA 等信息，部分患者可考虑免疫治疗或 PARP 抑制剂。",
        details=[
            "三阴性乳腺癌通常缺乏 ER、PR、HER2 靶点，早期治疗常以化疗为核心。",
            "PD-L1 阳性或高危早期/转移性场景中，免疫检查点抑制剂可能进入治疗评估。",
            "携带胚系 BRCA1/2 致病变异的部分患者可评估 PARP 抑制剂。",
            "治疗前需明确分期、复发风险、转移部位、基因检测和患者基础状况。",
        ],
        clinical_use="帮助智能体解释为什么需要免疫组化和遗传风险评估，避免仅基于影像给出治疗建议。",
        keywords=["三阴性", "PD-L1", "免疫治疗", "帕博利珠单抗", "PARP", "奥拉帕利", "BRCA"],
        references=[
            "NCI PDQ Breast Cancer Treatment: immunotherapy and PARP inhibitors.",
        ],
    ),
    KnowledgeEntry(
        id="biopsy-and-concordance",
        category="诊疗流程",
        title="穿刺活检与影像病理一致性",
        summary="可疑乳腺病灶通常需要组织学证据，且病理结论需与影像风险相互校验。",
        details=[
            "粗针穿刺活检可获得组织学类型、分级和免疫组化信息，是治疗分型的关键来源。",
            "影像高度可疑但病理良性时，应评估取材代表性和影像病理是否一致。",
            "若影像病理不一致，可能需要重复穿刺、真空辅助活检或手术活检。",
            "AI 结果可作为复核线索，但不能替代影像科和病理科的一致性判断。",
        ],
        clinical_use="智能体输出中应优先建议影像病理一致性评估，而不是直接把单次模型判定作为诊断终点。",
        keywords=["穿刺活检", "粗针", "病理", "影像病理一致性", "取材", "复核"],
        references=[
            "NCI PDQ Breast Cancer Treatment: diagnostic evaluation.",
        ],
    ),
    KnowledgeEntry(
        id="risk-factors-and-screening",
        category="筛查与风险因素",
        title="乳腺癌风险因素与筛查线索",
        summary="年龄、家族史、遗传易感、乳腺致密度、既往病史和激素暴露是风险评估中常用的背景信息。",
        details=[
            "年龄增长是乳腺癌风险评估中的重要背景因素，不能仅依靠单次影像概率判断整体风险。",
            "一级亲属乳腺癌或卵巢癌病史、BRCA1/2 等遗传易感信息可影响筛查强度和遗传咨询需求。",
            "乳腺致密度较高时，钼靶敏感性可能受限，超声或 MRI 可在特定场景中作为补充检查。",
            "既往胸部放疗、既往乳腺癌、良性增生性病变和绝经后肥胖等信息应记录在病例摘要中。",
        ],
        clinical_use="用于病例导入和智能体解读时提示医生补全风险背景，帮助判断是否需要强化筛查或遗传咨询。",
        keywords=["风险因素", "筛查", "家族史", "BRCA", "乳腺致密度", "遗传咨询", "年龄"],
        references=[
            "NCI PDQ Breast Cancer Treatment: risk factors.",
            "NCI PDQ Breast Cancer Screening.",
        ],
    ),
    KnowledgeEntry(
        id="imaging-workup-modalities",
        category="影像检查选择",
        title="乳腺影像检查方式的互补作用",
        summary="超声、钼靶、MRI、弹性成像和造影检查提供不同维度的信息，应按临床问题选择。",
        details=[
            "超声适合评估肿块囊实性、边界、后方回声、血流和腋窝淋巴结，常用于致密乳腺或触诊异常的补充评估。",
            "钼靶对钙化、结构扭曲和双侧对比有重要价值，特别是在导管原位癌或微钙化评估中不可忽视。",
            "乳腺 MRI 对多灶、多中心、术前范围评估和高危人群筛查有价值，但阳性发现仍需病理确认。",
            "弹性成像、超声造影等技术可作为风险评估的补充证据，不能脱离常规灰阶和 BI-RADS 体系单独下结论。",
        ],
        clinical_use="帮助智能体根据病例缺口建议补充合适检查，而不是默认所有患者都追加同一种影像检查。",
        keywords=["超声", "钼靶", "MRI", "弹性成像", "超声造影", "致密乳腺", "钙化"],
        references=[
            "NCI PDQ Breast Cancer Treatment: diagnosis.",
            "ACR BI-RADS Atlas.",
        ],
    ),
    KnowledgeEntry(
        id="birads-0-to-3",
        category="影像风险分层",
        title="BI-RADS 0、1、2、3 的处理含义",
        summary="BI-RADS 0 表示评估不完整，1/2 多为阴性或良性，3 通常代表可能良性并倾向短期随访。",
        details=[
            "BI-RADS 0 不是风险终点，表示需要补充既往影像、加做检查或获取更多信息后再分类。",
            "BI-RADS 1 表示未见明确异常，BI-RADS 2 表示良性发现，仍需结合症状和临床体征处理。",
            "BI-RADS 3 通常用于可能良性发现，常见策略是短期影像随访，随访间隔由医生按规范和个体情况决定。",
            "若 BI-RADS 3 病灶出现增大、形态改变或临床症状进展，应重新评估分类和取材必要性。",
        ],
        clinical_use="在低风险或随访病例中提醒智能体区分“暂不活检”和“无需管理”，避免低估动态变化。",
        keywords=["BI-RADS 0", "BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "随访", "良性", "评估不完整"],
        references=[
            "ACR BI-RADS Atlas: assessment categories.",
        ],
    ),
    KnowledgeEntry(
        id="birads-4-to-6",
        category="影像风险分层",
        title="BI-RADS 4、5、6 的处理含义",
        summary="BI-RADS 4 为可疑恶性，5 为高度提示恶性，6 为已病理证实恶性，处理路径通常更强调组织学和治疗分期。",
        details=[
            "BI-RADS 4 常细分为 4A、4B、4C，风险递增；通常需要组织学取材以明确性质。",
            "BI-RADS 5 表示影像高度提示恶性，应尽快推动病理确诊和治疗前分期评估。",
            "BI-RADS 6 指已病理证实恶性，影像任务转向范围评估、疗效监测、术前规划或复发评估。",
            "AI 输出可作为风险复核信息，但不能把概率分数直接等同于 BI-RADS 分类或病理诊断。",
        ],
        clinical_use="用于高风险病例的智能体建议，强调穿刺活检、影像病理一致性和治疗前资料补全。",
        keywords=["BI-RADS 4", "BI-RADS 5", "BI-RADS 6", "4A", "4B", "4C", "高度可疑", "病理证实"],
        references=[
            "ACR BI-RADS Atlas: assessment categories.",
            "NCI PDQ Breast Cancer Treatment: diagnosis.",
        ],
    ),
    KnowledgeEntry(
        id="benign-ultrasound-patterns",
        category="超声征象",
        title="倾向良性的常见超声表现",
        summary="边界清楚、形态规则、平行生长、后方增强和典型囊性结构常支持良性倾向，但仍需结合变化和临床背景。",
        details=[
            "单纯囊肿常表现为无回声、边界清楚、后方增强，通常属于良性影像特征。",
            "纤维腺瘤常见椭圆形、边界清楚、平行生长和均匀低回声，但非典型表现仍需复核。",
            "炎症、脂肪坏死、术后改变等良性病变可能模拟恶性征象，应结合病史和动态变化判断。",
            "良性倾向不等于不随访；若症状持续、快速增大或影像不典型，应重新评估。",
        ],
        clinical_use="帮助医生解释低风险输出时仍需结合病史和随访，降低模型过度简化良恶性的风险。",
        keywords=["良性", "囊肿", "纤维腺瘤", "平行生长", "后方增强", "边界清楚"],
        references=[
            "ACR BI-RADS Ultrasound lexicon.",
        ],
    ),
    KnowledgeEntry(
        id="lymph-node-assessment",
        category="超声征象",
        title="腋窝淋巴结超声评估要点",
        summary="乳腺肿瘤风险评估中应关注腋窝淋巴结皮质、门结构、形态和血流改变。",
        details=[
            "皮质增厚、门结构消失、形态趋圆、边界异常或异常血流分布可提示淋巴结受累风险升高。",
            "炎症和反应性增生也可导致淋巴结改变，应结合乳腺病灶和临床背景判断。",
            "疑似恶性乳腺病灶合并可疑腋窝淋巴结时，可考虑淋巴结穿刺或治疗前分期评估。",
            "AI 图像分类结果通常只反映上传图像，若未上传腋窝图像，不能替代腋窝评估。",
        ],
        clinical_use="在病例摘要和智能体输出中提示不要忽略腋窝状态，尤其是 BI-RADS 4C/5 或高风险病例。",
        keywords=["腋窝", "淋巴结", "皮质增厚", "门结构", "分期", "转移"],
        references=[
            "NCI PDQ Breast Cancer Treatment: stage information.",
        ],
    ),
    KnowledgeEntry(
        id="core-needle-vab-fna",
        category="诊疗流程",
        title="粗针、真空辅助和细针取材的差异",
        summary="不同取材方式提供的信息量不同，乳腺可疑实性病灶通常更重视组织学和免疫组化资料。",
        details=[
            "粗针穿刺活检可获得组织结构，便于判断病理类型、分级和免疫组化，是可疑实性病灶常用取材方式。",
            "真空辅助活检可获得更多组织，常用于部分微小病灶、钙化相关病变或需要更多样本的场景。",
            "细针穿刺主要提供细胞学信息，部分情况下难以完整评估组织结构和免疫组化。",
            "取材方式需结合病灶位置、大小、影像可见性、出血风险和机构流程决定。",
        ],
        clinical_use="当智能体建议“取材”时，可更规范地区分组织学确认、免疫组化和取材代表性的临床意义。",
        keywords=["粗针穿刺", "真空辅助活检", "细针穿刺", "组织学", "免疫组化", "取材"],
        references=[
            "NCI PDQ Breast Cancer Treatment: diagnosis.",
        ],
    ),
    KnowledgeEntry(
        id="surgery-principles",
        category="局部治疗",
        title="乳腺癌手术治疗的基本路径",
        summary="手术方式通常依据肿瘤范围、乳房大小比例、分期、病理类型、患者意愿和综合治疗计划确定。",
        details=[
            "保乳手术需关注切缘、肿瘤乳房比例、病灶范围和术后放疗条件。",
            "全乳切除可用于多灶、多中心、范围较广、无法保乳或患者选择等场景。",
            "前哨淋巴结活检和腋窝处理取决于临床腋窝状态、术前治疗和病理结果。",
            "影像 AI 平台只能辅助风险识别，不能替代外科手术适应证判断。",
        ],
        clinical_use="用于解释为什么疑似恶性病例需要进入多学科路径，而不是在影像界面直接确定手术方案。",
        keywords=["手术", "保乳", "全乳切除", "前哨淋巴结", "腋窝", "切缘"],
        references=[
            "NCI PDQ Breast Cancer Treatment: surgical treatment.",
        ],
    ),
    KnowledgeEntry(
        id="radiation-therapy-principles",
        category="局部治疗",
        title="乳腺癌放疗的常见决策场景",
        summary="放疗常与手术方式、淋巴结状态、肿瘤大小、切缘和复发风险相关。",
        details=[
            "保乳术后通常需要评估全乳放疗及必要时的瘤床加量。",
            "全乳切除后是否放疗取决于肿瘤大小、淋巴结受累、切缘和其他高危因素。",
            "区域淋巴结照射需结合腋窝、锁骨上和内乳区风险综合评估。",
            "放疗计划需要放疗科基于病理、影像范围和患者基础情况制定。",
        ],
        clinical_use="智能体涉及治疗路径时可提示放疗评估场景，但不生成照射剂量或靶区处方。",
        keywords=["放疗", "保乳术后", "全乳切除后", "区域淋巴结", "切缘", "复发风险"],
        references=[
            "NCI PDQ Breast Cancer Treatment: radiation therapy.",
        ],
    ),
    KnowledgeEntry(
        id="chemotherapy-neoadjuvant",
        category="系统治疗",
        title="化疗与新辅助治疗评估",
        summary="化疗和新辅助治疗决策依赖分期、分子分型、肿瘤负荷、淋巴结状态和患者基础情况。",
        details=[
            "新辅助治疗可用于部分局部进展、HER2 阳性、三阴性或希望降期保乳的病例评估。",
            "治疗前需尽量取得组织学和免疫组化信息，并建立基线影像用于疗效比较。",
            "新辅助过程中超声可用于病灶大小和腋窝变化监测，但疗效判定需结合多模态信息。",
            "化疗方案选择需肿瘤专科基于指南、分期、合并症和患者意愿决定。",
        ],
        clinical_use="帮助智能体在高风险病例中提示“补全分期和病理后进入系统治疗评估”，避免直接推荐方案。",
        keywords=["化疗", "新辅助治疗", "降期", "疗效评估", "HER2", "三阴性", "分期"],
        references=[
            "NCI PDQ Breast Cancer Treatment: systemic therapy.",
        ],
    ),
    KnowledgeEntry(
        id="fibroadenoma",
        category="病理知识",
        title="乳腺纤维腺瘤",
        summary="纤维腺瘤是常见良性乳腺肿瘤，典型影像多为边界清楚、形态规则的实性结节。",
        details=[
            "年轻女性常见，超声可表现为椭圆形、平行生长、边界清楚、内部回声相对均匀。",
            "快速增大、体积较大、形态不典型或年龄较大的新发实性结节需要谨慎复核。",
            "与叶状肿瘤等病变鉴别时，单次超声可能不足，应结合生长速度、组织学和临床判断。",
            "处理策略包括随访、穿刺确认或切除，需根据风险、症状和患者意愿决定。",
        ],
        clinical_use="用于解释低风险实性结节的可能良性路径，同时保留对快速增大或非典型表现的警惕。",
        keywords=["纤维腺瘤", "良性肿瘤", "实性结节", "椭圆形", "随访", "叶状肿瘤"],
        references=[
            "NCI Breast Cancer Treatment PDQ: histopathological classification.",
        ],
    ),
    KnowledgeEntry(
        id="intraductal-papilloma",
        category="病理知识",
        title="导管内乳头状病变",
        summary="导管内乳头状病变可表现为导管扩张、囊内或导管内实性成分，需关注不典型增生和恶性风险。",
        details=[
            "临床可出现乳头溢液，影像可能见导管内结节或囊实性结构。",
            "超声应记录病灶与导管关系、血流、乳头状突起和是否合并可疑钙化。",
            "穿刺结果若提示乳头状病变，需结合是否伴不典型增生和影像病理一致性决定后续处理。",
            "仅凭 AI 良恶性概率难以判断导管内病变的升级风险。",
        ],
        clinical_use="帮助智能体在乳头溢液或导管内病变场景中提示专门的病理和影像复核重点。",
        keywords=["导管内乳头状瘤", "乳头溢液", "导管扩张", "囊实性", "不典型增生"],
        references=[
            "NCI PDQ Breast Cancer Treatment: diagnosis.",
        ],
    ),
    KnowledgeEntry(
        id="phyllodes-tumor",
        category="病理知识",
        title="乳腺叶状肿瘤",
        summary="叶状肿瘤可与纤维腺瘤表现相近，但更强调快速生长、体积较大和组织学诊断。",
        details=[
            "叶状肿瘤可为良性、交界性或恶性，影像表现可能与纤维腺瘤重叠。",
            "短期快速增大、较大实性肿块或内部裂隙样改变时，应提高鉴别意识。",
            "治疗多强调完整切除和切缘评估，具体策略需外科和病理共同判断。",
            "AI 图像分类模型若训练集中叶状肿瘤样本不足，输出可靠性可能受限。",
        ],
        clinical_use="用于提醒智能体在大体积、快速增长的良性倾向结节中保留叶状肿瘤鉴别。",
        keywords=["叶状肿瘤", "纤维腺瘤", "快速增大", "切缘", "交界性"],
        references=[
            "NCI PDQ Breast Cancer Treatment: histopathological classification.",
        ],
    ),
    KnowledgeEntry(
        id="inflammatory-breast-cancer",
        category="特殊类型",
        title="炎性乳腺癌识别提示",
        summary="炎性乳腺癌可表现为乳房红肿、皮肤增厚、橘皮样改变和进展迅速，可能没有典型孤立肿块。",
        details=[
            "临床进展快、弥漫性皮肤改变和乳房肿胀是重要警示信号。",
            "超声可见皮肤增厚、乳腺实质水肿、弥漫异常和腋窝淋巴结改变。",
            "应与乳腺炎鉴别；治疗延误风险较高，疑似时需尽快组织学确认和分期。",
            "单张局部超声图像模型可能低估弥漫性病变，需结合临床照片、查体和多模态影像。",
        ],
        clinical_use="在病例主诉出现红肿热痛、皮肤改变或快速进展时，提示医生不要只依赖单图分类输出。",
        keywords=["炎性乳腺癌", "皮肤增厚", "橘皮样", "乳腺炎", "弥漫性", "快速进展"],
        references=[
            "NCI PDQ Breast Cancer Treatment: special presentations.",
        ],
    ),
    KnowledgeEntry(
        id="dcis-and-calcification",
        category="病理知识",
        title="导管原位癌与钙化评估",
        summary="导管原位癌常与钙化和导管改变相关，单纯超声可能低估病变范围。",
        details=[
            "钼靶对细小钙化和节段性分布识别具有重要价值，超声阴性不能排除钙化相关病变。",
            "导管原位癌可表现为导管改变、非肿块样低回声、囊实性结构或伴实性结节。",
            "病理需明确是否存在浸润成分、核级别、坏死和切缘情况。",
            "术前范围评估应结合钼靶、超声和必要时 MRI。",
        ],
        clinical_use="帮助智能体在报告提示钙化或导管改变时建议补充钼靶/病理信息，而不是只看超声概率。",
        keywords=["导管原位癌", "DCIS", "钙化", "钼靶", "导管改变", "浸润成分"],
        references=[
            "NCI PDQ Breast Cancer Treatment: ductal carcinoma in situ.",
        ],
    ),
    KnowledgeEntry(
        id="follow-up-and-surveillance",
        category="随访管理",
        title="随访与疗后监测",
        summary="随访管理应记录影像变化、治疗史、病理状态和复发风险，不能只比较单次 AI 分数。",
        details=[
            "随访重点包括病灶大小、形态、边界、血流、钙化和腋窝状态的动态变化。",
            "治疗后乳腺可能出现瘢痕、脂肪坏死、皮肤增厚或结构改变，需与复发鉴别。",
            "同一患者不同时间点的影像应尽量在相同检查方式和可比切面下比较。",
            "AI 分数变化可作为提示，但不能替代影像科正式比较和临床随访计划。",
        ],
        clinical_use="用于病例查看和智能体解读中提示医生保留既往影像和治疗信息，提升纵向判断质量。",
        keywords=["随访", "疗后监测", "复发", "影像变化", "脂肪坏死", "瘢痕"],
        references=[
            "NCI PDQ Breast Cancer Treatment: posttherapy surveillance.",
        ],
    ),
    KnowledgeEntry(
        id="red-flags-and-escalation",
        category="诊疗流程",
        title="需要升级处理的临床红旗信号",
        summary="快速增大、皮肤改变、乳头血性溢液、固定肿块、腋窝异常和影像病理不一致均应触发复核。",
        details=[
            "短期快速增大或固定质硬肿块，即使单次影像概率不高，也应结合临床检查升级评估。",
            "乳头血性溢液、乳头回缩、皮肤凹陷或橘皮样改变需要与影像发现共同判断。",
            "可疑腋窝淋巴结或全身症状提示需要进一步分期和专科评估。",
            "影像高度可疑但病理良性时，应优先复核取材是否代表病灶和影像病理是否一致。",
        ],
        clinical_use="作为智能体兜底安全规则，避免系统在关键信息提示高危时给出过低风险表达。",
        keywords=["红旗信号", "快速增大", "血性溢液", "乳头回缩", "腋窝", "影像病理不一致"],
        references=[
            "NCI PDQ Breast Cancer Treatment: diagnosis.",
        ],
    ),
    KnowledgeEntry(
        id="breast-density-screening",
        category="筛查与风险因素",
        title="乳腺致密度与补充筛查",
        summary="乳腺致密度会影响钼靶检出能力，也与风险评估有关，补充筛查需结合个人风险和机构流程。",
        details=[
            "致密乳腺可能降低钼靶对部分病灶的显示能力，报告中应记录乳腺组成和检查限制。",
            "超声可作为致密乳腺或触诊异常的补充工具，但筛查获益、假阳性和过度诊断风险需共同评估。",
            "MRI 常用于遗传高危、既往乳腺癌或术前范围评估等特定场景，不宜作为所有患者的默认追加检查。",
            "智能体在建议补充检查时应说明原因，例如致密乳腺、钙化线索、范围评估或影像病理不一致。",
        ],
        clinical_use="帮助平台在病例缺少乳腺致密度或筛查背景时提示补录，避免机械地推荐单一检查方式。",
        keywords=["乳腺致密度", "补充筛查", "钼靶", "超声", "MRI", "假阳性", "高危"],
        references=[
            "NCI Breast Cancer Screening PDQ.",
            "USPSTF Breast Cancer Screening recommendation.",
        ],
    ),
    KnowledgeEntry(
        id="genetic-risk-brca",
        category="筛查与风险因素",
        title="遗传风险、BRCA 与遗传咨询",
        summary="早发乳腺癌、双侧或多原发肿瘤、男性乳腺癌以及乳腺/卵巢癌家族史可提示遗传风险评估需求。",
        details=[
            "BRCA1/2 等胚系致病变异会影响筛查强度、手术讨论、系统治疗和家属风险管理。",
            "病史采集应记录一级亲属乳腺癌、卵巢癌、胰腺癌、前列腺癌等相关肿瘤谱。",
            "三阴性乳腺癌、年轻发病或多发原发肿瘤病例，应更主动提示遗传咨询或基因检测评估。",
            "遗传检测解读需由具备资质的临床团队完成，AI 输出只能提示是否存在评估线索。",
        ],
        clinical_use="用于智能体在病例导入缺少家族史时提出补充问题，并在高危病例中提示遗传咨询路径。",
        keywords=["BRCA", "遗传咨询", "家族史", "三阴性", "早发", "胚系", "卵巢癌"],
        references=[
            "NCI PDQ Breast Cancer Treatment: risk factors.",
            "NCI Breast Cancer Screening PDQ.",
        ],
    ),
    KnowledgeEntry(
        id="pathology-report-checklist",
        category="病理知识",
        title="乳腺穿刺与术后病理报告要素",
        summary="规范病理报告应支持诊断、分型、分级、分期和治疗决策，影像平台需提示关键缺失项。",
        details=[
            "穿刺病理至少应关注良恶性、组织学类型、核级别或组织学分级，以及是否足以解释影像表现。",
            "恶性病例常需 ER、PR、HER2、Ki-67 等免疫组化或分子检测信息，用于后续治疗分层。",
            "术后病理还需关注肿瘤大小、切缘、淋巴结状态、脉管侵犯和原位癌范围等信息。",
            "若病理结果与 BI-RADS 4C/5 或可疑超声征象不一致，应提示影像病理一致性复核。",
        ],
        clinical_use="让智能体输出“还缺哪些病理信息”时更具体，避免只笼统写需要病理检查。",
        keywords=["病理报告", "组织学类型", "分级", "切缘", "淋巴结", "脉管侵犯", "免疫组化"],
        references=[
            "NCI PDQ Breast Cancer Treatment: diagnostic evaluation.",
        ],
    ),
    KnowledgeEntry(
        id="her2-low-and-adc",
        category="药物治疗",
        title="HER2-low 与抗体偶联药物概念",
        summary="HER2 表达状态不再只用于阳性/阴性二分，部分 HER2-low 晚期病例可能进入 ADC 药物评估。",
        details=[
            "HER2 状态需依据规范 IHC 和 ISH/FISH 结果判读，不能由影像或 AI 概率推断。",
            "HER2-low 通常指 HER2 IHC 1+ 或 IHC 2+ 且 ISH 阴性等场景，具体定义应以病理标准为准。",
            "T-DXd 等抗体偶联药物在特定晚期或转移性场景中具有治疗意义，需要结合既往治疗线数和适应证。",
            "平台只能提示补全 HER2 检测和肿瘤专科评估，不能生成个体化用药方案。",
        ],
        clinical_use="当病例提到 HER2 低表达、复发转移或 ADC 时，为智能体提供规范化解释边界。",
        keywords=["HER2-low", "HER2 低表达", "ADC", "T-DXd", "抗体偶联药物", "IHC", "ISH"],
        references=[
            "NCI PDQ Breast Cancer Treatment: HER2 status.",
            "American Cancer Society: targeted drug therapy for breast cancer.",
        ],
    ),
    KnowledgeEntry(
        id="pi3k-mtor-akt-pathway",
        category="药物治疗",
        title="PI3K/AKT/mTOR 通路相关治疗信息",
        summary="部分 HR 阳性晚期乳腺癌治疗会结合 PIK3CA、AKT 通路或 mTOR 通路状态进行评估。",
        details=[
            "PIK3CA 突变、AKT 通路异常或内分泌治疗耐药背景可能影响后续靶向治疗讨论。",
            "阿培利司、依维莫司等药物属于特定分子或临床场景下的系统治疗选项，需严格依据适应证评估。",
            "这类治疗与血糖、口腔黏膜炎、皮疹等不良反应风险相关，需要肿瘤专科监测。",
            "影像系统可提示检测和专科评估需求，不能替代分子检测或药物安全性评估。",
        ],
        clinical_use="扩展药物知识库，使智能体能解释为什么晚期或复发病例可能需要分子检测信息。",
        keywords=["PIK3CA", "PI3K", "AKT", "mTOR", "阿培利司", "依维莫司", "内分泌耐药"],
        references=[
            "NCI PDQ Breast Cancer Treatment: targeted therapy.",
        ],
    ),
    KnowledgeEntry(
        id="neoadjuvant-response-imaging",
        category="影像检查选择",
        title="新辅助治疗疗效影像评估",
        summary="新辅助治疗前后需建立可比较的影像基线，超声、MRI 和临床查体各有价值和限制。",
        details=[
            "治疗前应记录病灶最大径、多灶范围、腋窝状态和可比切面，便于后续疗效判断。",
            "超声可动态观察肿块大小、形态和腋窝变化，但纤维化、坏死和残余肿瘤可能混杂。",
            "MRI 对范围和残余强化评估有价值，仍需结合病理完全缓解等终点评价。",
            "AI 单图分数不宜作为疗效评价指标，应作为影像复核的辅助线索。",
        ],
        clinical_use="当病例为治疗中或术前降期场景时，引导智能体强调基线和多模态比较。",
        keywords=["新辅助治疗", "疗效评估", "基线影像", "MRI", "残余病灶", "病理完全缓解"],
        references=[
            "NCI PDQ Breast Cancer Treatment: neoadjuvant therapy.",
        ],
    ),
    KnowledgeEntry(
        id="male-breast-cancer",
        category="特殊类型",
        title="男性乳腺癌评估提示",
        summary="男性乳腺癌少见但不能忽视，乳头后肿块、乳头溢液或回缩时需要规范影像和病理评估。",
        details=[
            "男性乳腺肿块可由乳腺发育、炎症或肿瘤等多种原因引起，单凭触诊和 AI 图像概率不足以下结论。",
            "可疑男性乳腺病灶同样需要关注腋窝淋巴结、家族史和遗传风险线索。",
            "病理和 ER、PR、HER2 状态仍是治疗决策的重要依据。",
            "病例字段中如果性别为男，智能体应避免默认女性筛查表述，并提示少见但需排除恶性。",
        ],
        clinical_use="提高平台对男性病例的表达质量，避免模板化输出造成临床语境错误。",
        keywords=["男性乳腺癌", "男性乳腺肿块", "乳腺发育", "遗传风险", "ER", "PR", "HER2"],
        references=[
            "NCI PDQ Male Breast Cancer Treatment.",
        ],
    ),
    KnowledgeEntry(
        id="pregnancy-lactation-breast-lesions",
        category="特殊类型",
        title="妊娠和哺乳期乳腺病变",
        summary="妊娠或哺乳期乳腺组织变化明显，炎症和良性病变常见，但持续肿块仍需及时评估。",
        details=[
            "哺乳期乳腺炎、脓肿、乳汁潴留和腺瘤样改变可能造成疼痛、红肿或局部肿块。",
            "妊娠相关乳腺癌可能因生理变化被延迟发现，持续或进展性肿块不应简单归因于哺乳。",
            "超声通常是妊娠和哺乳期局部肿块评估的重要工具，必要时仍需组织学确认。",
            "建议中应提示放射性检查和药物处理需要由临床医生结合孕周、哺乳状态和风险决定。",
        ],
        clinical_use="让智能体在年轻女性、哺乳或炎症样主诉中保留恶性鉴别，同时避免过度诊断。",
        keywords=["妊娠", "哺乳期", "乳腺炎", "脓肿", "持续肿块", "妊娠相关乳腺癌"],
        references=[
            "NCI PDQ Breast Cancer Treatment: special populations.",
        ],
    ),
    KnowledgeEntry(
        id="model-limitations-quality-control",
        category="AI质控",
        title="AI 模型输出的质量控制与适用边界",
        summary="图像质量、训练数据分布、ROI 定位和热力图一致性会影响 AI 辅助诊断可靠性。",
        details=[
            "上传图像若含大量标尺、文字、黑边、压缩伪影或非乳腺区域，模型关注可能偏离病灶。",
            "训练集未充分覆盖的特殊类型、术后改变、妊娠哺乳期改变或男性病例，输出不确定性更高。",
            "热力图只反映模型判别关注区域，不等同于病灶真实边界或病理证据。",
            "当 AI 结果与 BI-RADS、临床红旗或病理结果冲突时，应以人工复核和正式诊疗流程为准。",
        ],
        clinical_use="作为智能体安全护栏，帮助解释为什么快速度输出不代表诊断更确定。",
        keywords=["AI质控", "适用边界", "图像质量", "热力图", "Grad-CAM", "训练分布", "人工复核"],
        references=[
            "ACR BI-RADS Atlas: reporting context.",
        ],
    ),
    KnowledgeEntry(
        id="patient-communication-shared-decision",
        category="诊疗流程",
        title="患者沟通与共同决策要点",
        summary="辅助诊断平台输出应服务于医患沟通，表达不确定性、下一步检查目的和风险收益。",
        details=[
            "对低风险病例，应说明随访目的、观察指标和何时需要提前复诊。",
            "对可疑恶性病例，应解释穿刺活检的目的在于取得组织学和免疫组化证据。",
            "对治疗相关问题，应避免承诺疗效或替代专科会诊，重点说明还需哪些资料来决策。",
            "输出中应保留“辅助评估”定位，避免使用绝对化诊断或恐吓式表述。",
        ],
        clinical_use="使智能体回答更适合医生转述给患者，减少模型语言造成的误解。",
        keywords=["患者沟通", "共同决策", "不确定性", "随访", "活检目的", "风险收益"],
        references=[
            "NCI PDQ Breast Cancer Treatment: patient care context.",
        ],
    ),
]

KNOWLEDGE_ENTRIES.extend(
    [
        KnowledgeEntry(
            id="birads-report-components",
            category="影像风险分层",
            title="BI-RADS 报告组成要素",
            summary="规范乳腺影像报告应包含检查方式、乳腺组成、病灶描述、最终分类和管理建议。",
            details=[
                "病灶描述应尽量使用标准化词汇，避免“考虑恶性可能”等缺少结构化依据的笼统表达。",
                "最终分类需要与管理建议一致，例如可疑恶性分类通常应指向组织学确认路径。",
                "报告还应记录病灶位置、大小、可比既往影像和是否存在多灶或双侧异常。",
            ],
            clinical_use="帮助前端和智能体把超声文字、模型概率和 BI-RADS 分类放在同一套报告逻辑中解释。",
            keywords=["BI-RADS", "报告结构", "最终分类", "管理建议", "病灶描述"],
            references=["ACR BI-RADS Manual v2025.", "ACR BI-RADS Atlas."],
        ),
        KnowledgeEntry(
            id="birads-category-0-workup",
            category="影像风险分层",
            title="BI-RADS 0 与补充检查",
            summary="BI-RADS 0 表示当前资料不足，重点是补齐影像或临床信息后重新评估。",
            details=[
                "常见原因包括缺少既往片、显示不充分、需要局部加压或需要超声/MRI 补充。",
                "BI-RADS 0 不是良恶性结论，也不适合作为模型训练标签的终点。",
                "若平台病例显示 0 类，应优先提示“评估不完整”的具体缺口。",
            ],
            clinical_use="让系统在资料不足时避免强行输出风险结论，强调补充检查和复核。",
            keywords=["BI-RADS 0", "补充检查", "评估不完整", "既往影像", "复核"],
            references=["ACR BI-RADS Manual v2025."],
        ),
        KnowledgeEntry(
            id="birads-3-follow-up-dynamics",
            category="影像风险分层",
            title="BI-RADS 3 随访中的动态变化",
            summary="BI-RADS 3 的管理重点是短期复查和动态比较，而不是永久低风险标签。",
            details=[
                "随访中病灶增大、边界变差、形态改变或出现新可疑征象时，应重新分层。",
                "稳定并不等于无需记录，应保留可比切面、测量方式和检查日期。",
                "AI 分数若与影像稳定性冲突，应交由影像医生结合原图复核。",
            ],
            clinical_use="帮助医生解释为什么低风险或可能良性病例仍需要规范随访记录。",
            keywords=["BI-RADS 3", "短期随访", "动态变化", "复查", "稳定"],
            references=["ACR BI-RADS Manual v2025.", "NCI Breast Cancer Screening PDQ."],
        ),
        KnowledgeEntry(
            id="birads-4-subcategories",
            category="影像风险分层",
            title="BI-RADS 4A、4B、4C 的分层意义",
            summary="BI-RADS 4 代表可疑恶性，4A、4B、4C 用于表达从低到高的可疑程度。",
            details=[
                "4A 虽为较低可疑，但通常仍需组织学证据解释影像发现。",
                "4B 和 4C 更强调影像病理一致性，良性病理结果需要谨慎判断取材代表性。",
                "模型恶性概率可作为辅助信息，但不能直接替代 4A/4B/4C 的人工分类。",
            ],
            clinical_use="用于智能体对可疑恶性病例输出更细的取材和复核建议。",
            keywords=["BI-RADS 4A", "BI-RADS 4B", "BI-RADS 4C", "可疑恶性", "穿刺"],
            references=["ACR BI-RADS Manual v2025.", "ACR Appropriateness Criteria Palpable Breast Masses."],
        ),
        KnowledgeEntry(
            id="imaging-pathology-discordance-actions",
            category="诊疗流程",
            title="影像病理不一致后的处理思路",
            summary="影像高度可疑但病理良性时，应评估取材、定位和病灶代表性。",
            details=[
                "复核重点包括穿刺针道、标本量、病灶是否命中、病理是否能解释影像征象。",
                "不一致时可考虑重复粗针、真空辅助活检或手术活检，由临床团队决定。",
                "AI 输出不能解决不一致问题，只能提示需要人工复核和多学科讨论。",
            ],
            clinical_use="让平台在高风险病例中优先提醒医生不要把单次良性病理或模型结果作为终点。",
            keywords=["影像病理不一致", "重复穿刺", "真空辅助活检", "取材代表性", "MDT"],
            references=["NCI Breast Cancer Treatment PDQ.", "ACR Appropriateness Criteria Palpable Breast Masses."],
        ),
        KnowledgeEntry(
            id="ultrasound-shape-orientation",
            category="超声征象",
            title="超声形态与方位：规则、平行和非平行",
            summary="形态和方位是乳腺超声描述中最常用的风险线索之一。",
            details=[
                "椭圆形、规则、平行生长常支持良性倾向，但需要结合边界和内部回声。",
                "不规则形态或非平行生长提示病灶可能突破正常组织层面，风险权重更高。",
                "AI 热力图若集中在非病灶区域，应降低其对形态判断的解释权重。",
            ],
            clinical_use="用于把模型概率与可解释的超声词汇连接起来，提升报告可读性。",
            keywords=["形态", "椭圆形", "不规则", "平行", "非平行", "纵横比"],
            references=["ACR BI-RADS Ultrasound lexicon.", "ACR BI-RADS Manual v2025."],
        ),
        KnowledgeEntry(
            id="ultrasound-margin-boundary",
            category="超声征象",
            title="超声边缘与边界的风险含义",
            summary="边缘清楚、微分叶、角状、毛刺样或边界模糊会显著影响风险评估。",
            details=[
                "边界清楚常见于良性病灶，但部分恶性肿瘤也可表现相对清楚。",
                "角状、毛刺样、边缘不清或周围组织反应可提高恶性可疑程度。",
                "描述边界时应避免只写“欠清”，应尽量说明具体边缘形态。",
            ],
            clinical_use="帮助智能体在解释可疑超声描述时指出边缘信息的重要性和缺失项。",
            keywords=["边缘", "边界", "毛刺", "角状", "微分叶", "边界不清"],
            references=["ACR BI-RADS Ultrasound lexicon."],
        ),
        KnowledgeEntry(
            id="ultrasound-echo-posterior",
            category="超声征象",
            title="内部回声与后方声学特征",
            summary="低回声、不均匀回声、后方声影或增强可帮助区分病灶性质和组织成分。",
            details=[
                "不均匀低回声合并后方声影常增加可疑程度，但纤维化良性病变也可出现声影。",
                "后方增强常见于囊性或细胞丰富病变，应结合形态和边缘判断。",
                "内部回声描述应与灰阶图像质量、增益设置和探头压力共同复核。",
            ],
            clinical_use="让知识库支持对超声描述中“低回声、声影、增强”等词语的规范解释。",
            keywords=["低回声", "不均匀", "后方声影", "后方增强", "灰阶"],
            references=["ACR BI-RADS Ultrasound lexicon."],
        ),
        KnowledgeEntry(
            id="ultrasound-calcifications",
            category="超声征象",
            title="超声可见钙化与钼靶互补",
            summary="超声可见钙化具有提示意义，但钼靶通常更适合系统评估微钙化分布。",
            details=[
                "可疑钙化的形态和分布可与导管原位癌或浸润性病变相关。",
                "超声阴性不能排除钙化相关病变，尤其是单纯钙化或非肿块样病变。",
                "报告提示钙化时，智能体可建议结合钼靶或既往影像复核。",
            ],
            clinical_use="帮助平台在超声图像外提醒医生不要忽略钼靶在钙化评估中的价值。",
            keywords=["钙化", "微钙化", "钼靶", "DCIS", "导管原位癌"],
            references=["ACR BI-RADS Manual v2025.", "NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="doppler-vascularity",
            category="超声征象",
            title="彩色多普勒血流信息",
            summary="血流丰富、穿入性血流或异常血流分布可作为辅助风险线索。",
            details=[
                "血流信息受设备设置、探头压力和病灶大小影响，不能单独决定良恶性。",
                "炎症和部分良性病变也可血流丰富，需要结合临床症状和灰阶征象。",
                "可疑病灶合并异常血流时，可增强进一步取材或复核的理由。",
            ],
            clinical_use="用于智能体解释报告中的血流描述，避免把无血流简单等同于良性。",
            keywords=["多普勒", "血流", "穿入性血流", "炎症", "灰阶"],
            references=["ACR BI-RADS Ultrasound lexicon."],
        ),
        KnowledgeEntry(
            id="elastography-limitations",
            category="影像检查选择",
            title="乳腺弹性成像的用途与限制",
            summary="弹性成像可补充组织硬度信息，但不应脱离常规超声和 BI-RADS 分类单独决策。",
            details=[
                "较硬的病灶可能提示纤维化或恶性风险，但部分良性瘢痕和炎症也可偏硬。",
                "不同设备、ROI 选取和操作者会影响弹性指标的可比性。",
                "弹性结果与灰阶征象冲突时，应优先回到标准超声词典和临床背景复核。",
            ],
            clinical_use="支持医生把弹性成像作为补充证据，而不是替代组织学或 BI-RADS。",
            keywords=["弹性成像", "硬度", "剪切波", "ROI", "BI-RADS"],
            references=["ACR BI-RADS Manual v2025."],
        ),
        KnowledgeEntry(
            id="nonmass-ultrasound-findings",
            category="超声征象",
            title="非肿块样超声异常",
            summary="部分乳腺病变不形成清晰肿块，可表现为局灶低回声区、导管改变或结构扭曲。",
            details=[
                "非肿块样异常可能与 DCIS、炎症、纤维化或术后改变相关。",
                "评估时应关注分布范围、是否伴钙化、血流和导管内成分。",
                "单图 AI 分类模型对非肿块样异常可能不稳定，应强调人工复核。",
            ],
            clinical_use="拓展知识库对非典型超声表现的覆盖，减少只围绕“肿块”的模板化输出。",
            keywords=["非肿块", "结构扭曲", "导管改变", "DCIS", "低回声区"],
            references=["ACR BI-RADS Ultrasound lexicon.", "NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="architectural-distortion",
            category="超声征象",
            title="结构扭曲和牵拉征象",
            summary="结构扭曲可见于恶性病变、术后改变、放疗改变和部分高危良性病变。",
            details=[
                "若无明确手术或创伤史，新的结构扭曲应提高可疑程度。",
                "术后瘢痕和放疗改变需与既往影像比较，避免过度或不足诊断。",
                "结构扭曲常需要结合钼靶、超声和必要时 MRI 共同评估范围。",
            ],
            clinical_use="用于病例中出现牵拉、变形、扭曲等描述时的智能体解释。",
            keywords=["结构扭曲", "牵拉", "瘢痕", "放疗后", "范围评估"],
            references=["ACR BI-RADS Manual v2025."],
        ),
        KnowledgeEntry(
            id="ductal-findings",
            category="超声征象",
            title="导管扩张和导管内异常",
            summary="导管扩张、导管内实性回声和血性溢液相关表现需要结合临床和病理评估。",
            details=[
                "单纯对称导管扩张可为良性背景，但局灶性导管内实性成分需谨慎。",
                "血性溢液、单孔溢液或乳头改变时，应考虑乳管内病变和恶性风险。",
                "导管内异常可需要靶向超声、钼靶、MRI 或病理取材进一步确认。",
            ],
            clinical_use="帮助系统处理乳头溢液、导管内乳头状瘤和 DCIS 相关病例。",
            keywords=["导管扩张", "导管内", "乳头溢液", "血性溢液", "乳头状瘤"],
            references=["NCI Breast Cancer Treatment PDQ.", "ACR BI-RADS Ultrasound lexicon."],
        ),
        KnowledgeEntry(
            id="skin-nipple-areolar-signs",
            category="超声征象",
            title="皮肤、乳头和乳晕改变",
            summary="皮肤增厚、乳头回缩、乳晕区异常和橘皮样改变是重要临床和影像线索。",
            details=[
                "皮肤增厚可见于炎症、淋巴回流受阻、放疗后改变或炎性乳腺癌。",
                "新发乳头回缩或乳头血性溢液需要与导管内病变和恶性病变鉴别。",
                "局部超声图像不能替代查体和双侧对比，应记录临床外观。",
            ],
            clinical_use="让智能体在主诉含皮肤或乳头改变时自动提示风险升级和补充信息。",
            keywords=["皮肤增厚", "乳头回缩", "乳晕", "橘皮样", "炎性乳腺癌"],
            references=["CDC Breast Cancer Symptoms.", "NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="simple-cyst",
            category="病理知识",
            title="单纯乳腺囊肿",
            summary="单纯囊肿通常表现为无回声、边界清楚、后方增强，是典型良性影像模式。",
            details=[
                "典型单纯囊肿通常无需按实性肿块路径处理，但症状明显时可临床处理。",
                "囊肿内部出现实性成分、厚壁、分隔或血流时，应转入复杂囊实性病变评估。",
                "平台应避免把囊肿误解为“肿瘤概率低但仍需穿刺”的固定模板。",
            ],
            clinical_use="丰富良性病变解释，帮助低风险输出更符合影像语境。",
            keywords=["单纯囊肿", "无回声", "后方增强", "良性", "囊性"],
            references=["ACR BI-RADS Ultrasound lexicon."],
        ),
        KnowledgeEntry(
            id="complicated-complex-cyst",
            category="病理知识",
            title="复杂囊肿和复杂囊实性病变",
            summary="复杂囊性表现的风险取决于内部回声、壁结节、实性成分和血流。",
            details=[
                "复杂囊肿可因出血、感染或蛋白性内容物产生内部回声。",
                "囊实性病变若有实性结节、乳头状突起或血流，应提高病理确认需求。",
                "描述中应区分“复杂囊肿”和“复杂囊实性肿块”，二者管理含义不同。",
            ],
            clinical_use="帮助智能体在囊性病灶场景中输出更细的鉴别和取材建议。",
            keywords=["复杂囊肿", "囊实性", "壁结节", "乳头状突起", "血流"],
            references=["ACR BI-RADS Ultrasound lexicon.", "NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="fibroadenoma-management",
            category="病理知识",
            title="纤维腺瘤与随访管理",
            summary="纤维腺瘤是常见良性肿瘤，典型表现可随访，非典型或快速增大需复核。",
            details=[
                "典型超声表现包括椭圆形、边界清楚、平行生长和均匀低回声。",
                "短期快速增大、体积较大或影像不典型时，应鉴别叶状肿瘤或其他病变。",
                "病理已证实良性但影像不一致时，仍需影像病理一致性判断。",
            ],
            clinical_use="让系统处理年轻患者良性结节时能给出随访和升级复核条件。",
            keywords=["纤维腺瘤", "良性肿瘤", "快速增大", "叶状肿瘤", "随访"],
            references=["ACR BI-RADS Ultrasound lexicon.", "NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="lactating-adenoma",
            category="特殊类型",
            title="哺乳期腺瘤样病变",
            summary="妊娠和哺乳期可出现腺瘤样良性病变，但持续增大或不典型表现需评估。",
            details=[
                "哺乳期组织血供和腺体增生会改变超声背景，可能增加解释难度。",
                "持续性实性肿块不能简单归因于哺乳，应结合影像和必要时病理。",
                "用药和检查建议需考虑妊娠或哺乳状态，由临床医生决定。",
            ],
            clinical_use="补充妊娠哺乳期病例的良性鉴别和安全边界。",
            keywords=["哺乳期", "妊娠", "腺瘤", "实性肿块", "安全边界"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="mastitis-abscess",
            category="病理知识",
            title="乳腺炎和乳腺脓肿",
            summary="乳腺炎和脓肿可造成红肿热痛、局部肿块和复杂液性区，需要与恶性表现鉴别。",
            details=[
                "炎症可导致皮肤增厚、血流增多和反应性淋巴结，影像表现可能较复杂。",
                "治疗后不消退或反复发作的局灶异常，应考虑进一步影像和病理评估。",
                "炎性乳腺癌可模拟炎症，快速进展或皮肤改变明显时需提高警惕。",
            ],
            clinical_use="帮助平台避免把炎症样表现机械归为良性或恶性，强调复查和临床响应。",
            keywords=["乳腺炎", "脓肿", "红肿热痛", "炎性乳腺癌", "复查"],
            references=["CDC Breast Cancer Symptoms.", "NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="fat-necrosis",
            category="病理知识",
            title="脂肪坏死和外伤术后改变",
            summary="脂肪坏死可由外伤、手术或放疗后改变引起，影像上可能模拟恶性。",
            details=[
                "表现可包括油囊、钙化、低回声肿块、声影或结构扭曲。",
                "明确外伤、手术或放疗史有助于解释影像，但不能替代必要复核。",
                "随访稳定或典型良性表现支持保守管理，进展或不典型需进一步评估。",
            ],
            clinical_use="用于解释疗后或外伤后病例中 AI 高分但影像背景复杂的情况。",
            keywords=["脂肪坏死", "外伤", "术后", "放疗后", "油囊", "钙化"],
            references=["ACR BI-RADS Manual v2025."],
        ),
        KnowledgeEntry(
            id="radial-scar",
            category="病理知识",
            title="放射状瘢痕和复杂硬化性病变",
            summary="放射状瘢痕可产生结构扭曲并模拟恶性，常需要影像病理一致性评估。",
            details=[
                "影像上可表现为牵拉、扭曲、声影或星芒样改变。",
                "穿刺结果需结合取材充分性和是否伴不典型增生或恶性成分。",
                "管理通常由影像、病理和外科根据一致性共同决定。",
            ],
            clinical_use="让知识库覆盖高危良性病变，减少把结构扭曲简单等同恶性。",
            keywords=["放射状瘢痕", "复杂硬化性病变", "结构扭曲", "高危病变", "一致性"],
            references=["NCI Breast Cancer Treatment PDQ.", "ACR BI-RADS Manual v2025."],
        ),
        KnowledgeEntry(
            id="intraductal-papilloma-risk",
            category="病理知识",
            title="导管内乳头状瘤的风险分层",
            summary="导管内乳头状瘤可表现为导管内结节或乳头溢液，风险取决于是否伴不典型增生。",
            details=[
                "中央型病变常与乳头溢液相关，外周型或多发病变需要结合影像范围评估。",
                "若伴不典型增生、影像病理不一致或取材不足，处理策略会更积极。",
                "血流、实性成分和导管扩张应在报告中具体描述。",
            ],
            clinical_use="帮助智能体在乳头溢液或导管内病变场景中解释病理升级风险。",
            keywords=["导管内乳头状瘤", "不典型增生", "乳头溢液", "导管内结节", "血流"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="atypical-ductal-hyperplasia",
            category="病理知识",
            title="不典型导管增生 ADH",
            summary="ADH 属于高危病变，穿刺诊断后常需评估是否低估邻近 DCIS 或浸润癌。",
            details=[
                "ADH 与低级别 DCIS 在病理上存在连续谱，取材范围会影响诊断。",
                "影像表现若为可疑钙化或结构扭曲，应重视影像病理一致性。",
                "后续管理需结合病灶范围、取材方式、病理意见和患者风险背景。",
            ],
            clinical_use="让系统对“良性但高危”病理结果给出更准确的复核提示。",
            keywords=["ADH", "不典型导管增生", "高危病变", "DCIS", "低估"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="lobular-neoplasia",
            category="病理知识",
            title="小叶原位病变和小叶肿瘤谱",
            summary="小叶原位癌和非典型小叶增生常作为风险标志，需要结合影像病理一致性。",
            details=[
                "小叶原位病变可能不形成明确肿块，常在其他取材中偶然发现。",
                "若影像目标可疑而病理仅提示小叶原位病变，应判断是否充分解释影像。",
                "风险管理和随访策略需由乳腺专科结合个人风险评估。",
            ],
            clinical_use="帮助智能体处理“病理不是浸润癌但提示高风险”的复杂表达。",
            keywords=["小叶原位癌", "非典型小叶增生", "小叶肿瘤", "风险标志", "一致性"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="invasive-ductal-carcinoma-details",
            category="病理知识",
            title="浸润性导管癌的影像病理关联",
            summary="浸润性导管癌是常见浸润性乳腺癌类型，常与不规则低回声肿块和声影相关。",
            details=[
                "病理报告需提供组织学分级、ER/PR/HER2、Ki-67 和淋巴脉管侵犯等信息。",
                "超声表现可较典型，但仍不能用影像直接替代组织学诊断。",
                "治疗路径需结合肿瘤大小、淋巴结状态、分子分型和患者情况。",
            ],
            clinical_use="作为智能体解释恶性高风险超声表现和病理补充项的常用条目。",
            keywords=["浸润性导管癌", "低回声", "声影", "分级", "免疫组化"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="invasive-lobular-carcinoma-details",
            category="病理知识",
            title="浸润性小叶癌的范围评估",
            summary="浸润性小叶癌可能呈弥漫浸润、多灶或影像低估范围，需重视多模态评估。",
            details=[
                "临床和影像表现可能不如导管癌形成清楚肿块，范围评估更具挑战。",
                "MRI 在部分术前范围评估中有价值，但阳性发现仍需病理确认。",
                "双侧、多灶或影像临床不一致时，应提醒进一步复核。",
            ],
            clinical_use="让智能体在疑似或已证实小叶癌场景中提示范围评估和影像低估风险。",
            keywords=["浸润性小叶癌", "多灶", "范围评估", "MRI", "影像低估"],
            references=["NCI Breast Cancer Treatment PDQ.", "ACR BI-RADS Manual v2025."],
        ),
        KnowledgeEntry(
            id="mucinous-carcinoma",
            category="病理知识",
            title="黏液癌等特殊型乳腺癌",
            summary="部分特殊型乳腺癌影像表现可相对边界清楚，不能因外观温和而忽略病理确认。",
            details=[
                "黏液癌可表现为边界相对清楚或后方增强，可能与良性结节混淆。",
                "病理特殊类型与预后和治疗策略相关，需要组织学明确。",
                "AI 模型若训练样本不足，对特殊型癌的泛化能力需谨慎。",
            ],
            clinical_use="用于提示系统不要只依据典型恶性征象判断所有恶性风险。",
            keywords=["黏液癌", "特殊型乳腺癌", "边界清楚", "后方增强", "泛化"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="metaplastic-carcinoma",
            category="病理知识",
            title="化生性乳腺癌",
            summary="化生性乳腺癌少见，常呈高级别或三阴性表型，影像和病理均需谨慎。",
            details=[
                "影像可表现为快速增大的实性或囊实性肿块，部分边界可相对清楚。",
                "治疗决策依赖病理类型、分期和分子标志物，不能由超声图像直接推断。",
                "平台输出应提示特殊类型样本不足可能造成模型不确定性。",
            ],
            clinical_use="补充少见恶性类型知识，强化 AI 适用边界说明。",
            keywords=["化生性乳腺癌", "三阴性", "快速增大", "少见类型", "模型不确定性"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="tubular-cribriform-carcinoma",
            category="病理知识",
            title="管状癌和筛状癌",
            summary="部分低级别特殊型浸润癌预后相对较好，但仍需完整病理和分期评估。",
            details=[
                "管状癌和筛状癌可表现为小病灶，影像上有时不典型。",
                "低级别并不代表无需治疗，处理仍基于分期、切缘和受体状态。",
                "智能体应避免根据病理名称直接承诺预后或治疗强度。",
            ],
            clinical_use="让知识库覆盖低级别特殊类型，辅助解释病理报告中的术语。",
            keywords=["管状癌", "筛状癌", "低级别", "特殊型", "预后"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="paget-disease",
            category="特殊类型",
            title="乳头 Paget 病",
            summary="乳头 Paget 病常表现为乳头乳晕湿疹样改变，可能伴 DCIS 或浸润癌。",
            details=[
                "临床上可出现乳头瘙痒、糜烂、脱屑、渗出或疼痛。",
                "影像可能需要钼靶、超声或 MRI 寻找潜在导管内或浸润性病灶。",
                "持续乳头改变不应长期按普通皮肤病处理，应考虑病理确认。",
            ],
            clinical_use="用于主诉含乳头皮肤改变时的风险提示和补充检查建议。",
            keywords=["Paget 病", "乳头", "乳晕", "湿疹样", "DCIS"],
            references=["NCI Breast Cancer Treatment PDQ.", "CDC Breast Cancer Symptoms."],
        ),
        KnowledgeEntry(
            id="occult-breast-cancer-axillary",
            category="特殊类型",
            title="腋窝转移提示隐匿性乳腺癌",
            summary="腋窝淋巴结转移而乳腺原发灶不明确时，需要系统寻找隐匿性乳腺癌。",
            details=[
                "可疑腋窝淋巴结应结合超声形态、穿刺病理和免疫组化判断来源。",
                "乳腺 MRI、钼靶和靶向超声可用于寻找隐匿原发灶。",
                "单张乳腺肿块模型无法覆盖这种临床路径，需专科评估。",
            ],
            clinical_use="提醒平台在腋窝信息异常时不要只聚焦上传的乳腺局部图像。",
            keywords=["隐匿性乳腺癌", "腋窝转移", "淋巴结", "MRI", "原发灶"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="breast-lymphoma-and-metastasis",
            category="特殊类型",
            title="乳腺淋巴瘤和转移性病变鉴别",
            summary="乳腺肿块少数可来自淋巴瘤或其他肿瘤转移，病史和病理非常关键。",
            details=[
                "既往恶性肿瘤史、系统症状或多部位病灶可提示非原发乳腺癌可能。",
                "影像表现可能缺乏典型乳腺癌征象，病理和免疫组化是鉴别核心。",
                "AI 模型多以常见乳腺病变训练，对罕见来源输出需谨慎。",
            ],
            clinical_use="扩展特殊病例覆盖，避免智能体把所有乳腺肿块默认为常见乳腺癌。",
            keywords=["乳腺淋巴瘤", "转移性肿瘤", "既往肿瘤史", "免疫组化", "罕见"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="hormone-receptor-interpretation",
            category="分子标志物",
            title="激素受体 ER/PR 判读意义",
            summary="ER 和 PR 状态用于判断肿瘤是否可能从内分泌治疗中获益。",
            details=[
                "ER/PR 阳性支持进入内分泌治疗评估，但方案选择取决于绝经状态和分期。",
                "低表达或异质性结果需结合病理报告和肿瘤专科意见。",
                "影像或 AI 概率不能推断激素受体状态。",
            ],
            clinical_use="让智能体在缺少 ER/PR 时明确提示免疫组化缺口。",
            keywords=["ER", "PR", "激素受体", "内分泌治疗", "免疫组化"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Hormone Therapy."],
        ),
        KnowledgeEntry(
            id="her2-ihc-ish-interpretation",
            category="分子标志物",
            title="HER2 IHC 与 ISH/FISH 判读路径",
            summary="HER2 状态通常依赖 IHC 和必要时 ISH/FISH 检测，决定抗 HER2 治疗评估。",
            details=[
                "IHC 3+ 通常支持 HER2 阳性，IHC 2+ 常需要 ISH/FISH 等进一步检测。",
                "HER2 低表达、异质性和检测质量会影响治疗讨论，应以病理标准为准。",
                "平台只能提示补全 HER2 信息，不能基于超声预测靶向治疗适应证。",
            ],
            clinical_use="用于 HER2 相关问题的智能体解释和知识库检索。",
            keywords=["HER2", "IHC", "ISH", "FISH", "靶向治疗", "HER2-low"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Targeted Drug Therapy."],
        ),
        KnowledgeEntry(
            id="ki67-interpretation-limits",
            category="分子标志物",
            title="Ki-67 增殖指数的解释限制",
            summary="Ki-67 反映增殖活性，但不同实验室和判读标准可能存在差异。",
            details=[
                "Ki-67 可辅助区分增殖活性高低，但不应作为单一治疗决策依据。",
                "数值解释需结合组织学分级、ER/PR/HER2 和临床分期。",
                "智能体应提示 Ki-67 阈值存在机构差异，避免写成固定绝对标准。",
            ],
            clinical_use="帮助输出更规范地解释 Ki-67，而不制造过度确定性。",
            keywords=["Ki-67", "增殖指数", "分级", "阈值差异", "免疫组化"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="genomic-assays-recurrence",
            category="分子标志物",
            title="复发风险基因表达检测",
            summary="部分早期 HR 阳性、HER2 阴性乳腺癌可评估多基因检测以辅助系统治疗决策。",
            details=[
                "Oncotype DX、MammaPrint 等检测用于特定人群的复发风险和化疗获益评估。",
                "是否适用取决于分期、淋巴结状态、受体状态和当地可及性。",
                "影像平台只能提示可能需要肿瘤专科评估，不能自行决定检测或治疗。",
            ],
            clinical_use="用于智能体回答“为什么还需要分子检测或基因表达检测”。",
            keywords=["Oncotype DX", "MammaPrint", "复发风险", "HR阳性", "HER2阴性"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="pdl1-testing-tnbc",
            category="分子标志物",
            title="三阴性乳腺癌中的 PD-L1 检测",
            summary="部分三阴性乳腺癌治疗评估会涉及 PD-L1 等免疫相关检测。",
            details=[
                "PD-L1 检测平台、评分体系和适用药物需按具体治疗场景判断。",
                "高危早期或转移性三阴性病例可能进入免疫治疗评估路径。",
                "AI 图像结果不能推断 PD-L1 状态，应提示补全病理和分子资料。",
            ],
            clinical_use="支持三阴性病例中对免疫治疗相关问题的规范回答。",
            keywords=["PD-L1", "三阴性", "免疫治疗", "检查点抑制剂", "分子检测"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="germline-brca-palb2",
            category="筛查与风险因素",
            title="BRCA1/2、PALB2 等胚系风险基因",
            summary="部分胚系致病变异会显著影响乳腺癌风险、筛查强度和治疗讨论。",
            details=[
                "早发、双侧、多原发、男性乳腺癌或明显家族史可提示遗传咨询需求。",
                "BRCA1/2 信息可影响筛查、手术讨论和 PARP 抑制剂等系统治疗评估。",
                "遗传检测结果涉及家属风险和伦理沟通，需由专业团队解释。",
            ],
            clinical_use="用于病例导入缺少家族史或年轻高危病例时的补充提示。",
            keywords=["BRCA1", "BRCA2", "PALB2", "胚系", "遗传咨询", "家族史"],
            references=["NCI BRCA Gene Changes Fact Sheet.", "USPSTF BRCA-Related Cancer Recommendation."],
        ),
        KnowledgeEntry(
            id="vus-genetic-testing",
            category="筛查与风险因素",
            title="遗传检测中的意义未明变异 VUS",
            summary="VUS 不能等同于致病变异，管理通常不应仅基于 VUS 升级。",
            details=[
                "VUS 需要随数据库和证据积累重新分类，短期内不应过度解释。",
                "临床决策仍应依据个人病史、家族史和已确认致病变异。",
                "智能体应提醒医生避免把 VUS 写成明确遗传高风险。",
            ],
            clinical_use="提高平台对遗传检测报告的表达准确性。",
            keywords=["VUS", "意义未明变异", "遗传检测", "致病变异", "再分类"],
            references=["NCI BRCA Gene Changes Fact Sheet.", "USPSTF BRCA-Related Cancer Recommendation."],
        ),
        KnowledgeEntry(
            id="tumor-grade-lvi",
            category="病理知识",
            title="组织学分级与淋巴脉管侵犯",
            summary="组织学分级和淋巴脉管侵犯是复发风险、分期讨论和治疗规划中的重要病理要素。",
            details=[
                "高级别肿瘤通常提示增殖和侵袭性更强，但需结合分子分型和分期。",
                "淋巴脉管侵犯可提示转移风险增加，应在病理报告中明确记录。",
                "平台应把这些作为缺失项提示，而不是通过影像推断。",
            ],
            clinical_use="用于智能体列出病理报告还缺哪些关键信息。",
            keywords=["组织学分级", "淋巴脉管侵犯", "复发风险", "病理报告", "分期"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="surgical-margins",
            category="局部治疗",
            title="切缘状态和局部复发风险",
            summary="切缘状态是保乳或手术后局部治疗计划中的关键病理信息。",
            details=[
                "阳性或过近切缘可能需要再次切除或调整局部治疗计划。",
                "DCIS 和浸润癌的切缘讨论重点不同，应以外科和病理标准为准。",
                "影像平台可提示记录切缘信息，但不能决定是否再次手术。",
            ],
            clinical_use="帮助智能体解释术后病例为什么需要完整病理报告。",
            keywords=["切缘", "保乳", "DCIS", "局部复发", "再次切除"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="residual-cancer-burden",
            category="病理知识",
            title="新辅助治疗后的残余病灶评估",
            summary="新辅助治疗后残余肿瘤负荷与预后和后续治疗讨论相关。",
            details=[
                "病理完全缓解和残余病灶需通过术后病理确认，影像评估不能完全替代。",
                "残余病灶大小、细胞密度、淋巴结状态和受体变化都应关注。",
                "治疗前后影像应建立可比基线，AI 分数不能作为疗效终点。",
            ],
            clinical_use="用于治疗中病例的智能体解释，强调影像和病理终点差异。",
            keywords=["新辅助治疗", "残余病灶", "病理完全缓解", "疗效评估", "淋巴结"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="tnm-tumor-size",
            category="分期评估",
            title="TNM 中肿瘤大小和范围评估",
            summary="T 分期关注原发肿瘤大小和局部侵犯范围，需要影像、查体和病理结合。",
            details=[
                "超声可测量可见肿块大小，但多灶、非肿块样或弥漫性病变可能低估范围。",
                "术前范围评估可结合钼靶、超声、MRI 和临床查体。",
                "模型输入单张图像时，不应把单图 ROI 等同于完整肿瘤范围。",
            ],
            clinical_use="帮助平台在高风险病例中提示分期资料缺失。",
            keywords=["TNM", "T分期", "肿瘤大小", "多灶", "范围评估"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="nodal-staging-ultrasound-pathology",
            category="分期评估",
            title="淋巴结分期：影像和病理证据",
            summary="N 分期需要结合腋窝查体、影像、穿刺和术中/术后病理。",
            details=[
                "超声可提示可疑淋巴结，但最终分期通常需要病理依据。",
                "治疗前阳性淋巴结会影响新辅助治疗、腋窝处理和放疗范围讨论。",
                "若上传图像未包含腋窝区域，模型结果不应覆盖淋巴结状态。",
            ],
            clinical_use="用于智能体提醒医生补充腋窝评估和病理信息。",
            keywords=["N分期", "腋窝淋巴结", "穿刺", "前哨淋巴结", "分期"],
            references=["NCI Breast Cancer Treatment PDQ.", "ACR Appropriateness Criteria Palpable Breast Masses."],
        ),
        KnowledgeEntry(
            id="metastatic-workup",
            category="分期评估",
            title="远处转移评估的触发场景",
            summary="远处转移检查通常基于分期、症状、实验室异常和高风险临床背景决定。",
            details=[
                "骨痛、呼吸症状、肝功能异常或神经症状可提示针对性评估。",
                "早期低风险病例不一定需要广泛全身影像，避免过度检查。",
                "疑似局部晚期或高负荷疾病时，分期影像对治疗计划有重要意义。",
            ],
            clinical_use="让智能体在建议分期检查时说明触发依据，而不是默认所有病例做全身筛查。",
            keywords=["M分期", "远处转移", "骨痛", "分期影像", "局部晚期"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="recurrence-risk-factors",
            category="分期评估",
            title="复发风险综合因素",
            summary="复发风险与肿瘤大小、淋巴结、分级、分子分型、切缘和治疗反应共同相关。",
            details=[
                "不能只用超声恶性概率或单个病理指标判断复发风险。",
                "HR/HER2 状态、Ki-67、基因表达检测和治疗反应可进入综合评估。",
                "智能体应把复发风险表达为需要专科综合评估的议题。",
            ],
            clinical_use="帮助平台把病理、分期和治疗信息整合进病例摘要。",
            keywords=["复发风险", "淋巴结", "分子分型", "治疗反应", "综合评估"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="breast-conserving-therapy",
            category="局部治疗",
            title="保乳治疗的基本条件",
            summary="保乳治疗通常需要考虑肿瘤范围、切缘、乳房肿瘤比例和术后放疗条件。",
            details=[
                "单灶、可完整切除并获得可接受外观的病例更可能进入保乳讨论。",
                "多灶、多中心或范围广泛病变可能限制保乳可行性。",
                "最终选择需由外科、放疗科和患者共同决策。",
            ],
            clinical_use="用于解释为什么影像范围评估会影响手术方式讨论。",
            keywords=["保乳", "切缘", "术后放疗", "肿瘤范围", "共同决策"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="mastectomy-reconstruction",
            category="局部治疗",
            title="全乳切除和乳房重建讨论",
            summary="全乳切除可用于特定范围或风险背景，重建选择需结合治疗计划和患者意愿。",
            details=[
                "多中心病变、广泛 DCIS、无法保乳或个人选择都可能进入全乳切除讨论。",
                "即刻或延迟重建需考虑放疗需求、基础疾病和患者偏好。",
                "影像平台不应推荐具体手术方式，只提示需要外科评估的信息。",
            ],
            clinical_use="帮助智能体在高风险病例中保持手术建议边界。",
            keywords=["全乳切除", "乳房重建", "多中心", "广泛DCIS", "外科评估"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="sentinel-lymph-node-biopsy",
            category="局部治疗",
            title="前哨淋巴结活检概念",
            summary="前哨淋巴结活检用于评估乳腺癌是否已累及区域淋巴引流通路。",
            details=[
                "临床腋窝阴性早期病例常会讨论前哨淋巴结活检。",
                "若治疗前腋窝已证实阳性，腋窝处理路径会不同。",
                "影像可提示淋巴结可疑，但病理和手术策略需专科决定。",
            ],
            clinical_use="让智能体能解释腋窝评估和前哨淋巴结术语。",
            keywords=["前哨淋巴结", "腋窝", "区域分期", "淋巴引流", "病理"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="axillary-dissection-morbidity",
            category="局部治疗",
            title="腋窝淋巴结清扫和并发症风险",
            summary="腋窝清扫可提供控制和分期信息，但与淋巴水肿、感觉异常和肩部功能影响相关。",
            details=[
                "是否清扫取决于淋巴结负荷、术前治疗、前哨结果和局部治疗计划。",
                "淋巴水肿风险需要术前沟通和术后康复管理。",
                "平台只能提示腋窝状态重要性，不能替代外科决策。",
            ],
            clinical_use="用于患者沟通和智能体解释腋窝处理为何需要谨慎。",
            keywords=["腋窝清扫", "淋巴水肿", "肩部功能", "前哨淋巴结", "并发症"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="radiation-after-lumpectomy",
            category="局部治疗",
            title="保乳术后放疗",
            summary="保乳术后常需评估全乳放疗、部分乳腺照射或瘤床加量等方案。",
            details=[
                "放疗计划取决于年龄、肿瘤大小、切缘、分级、淋巴结状态和复发风险。",
                "部分低风险老年患者可能有个体化讨论空间，需放疗科评估。",
                "智能体不应给出剂量分割，只能提示需专科制定计划。",
            ],
            clinical_use="帮助术后病例解释为什么需要放疗科评估。",
            keywords=["保乳术后", "放疗", "瘤床加量", "切缘", "复发风险"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Radiation Therapy."],
        ),
        KnowledgeEntry(
            id="postmastectomy-radiation",
            category="局部治疗",
            title="全乳切除后放疗评估",
            summary="全乳切除后是否放疗取决于肿瘤大小、淋巴结受累、切缘和其他高危因素。",
            details=[
                "淋巴结阳性、肿瘤较大或切缘问题常会触发放疗讨论。",
                "区域淋巴结照射范围需结合腋窝、锁骨上和内乳区风险。",
                "放疗决策需基于完整病理和治疗计划，不能由 AI 图像输出决定。",
            ],
            clinical_use="用于解释术后病理信息和局部治疗计划之间的关系。",
            keywords=["全乳切除后放疗", "淋巴结阳性", "切缘", "区域淋巴结", "放疗科"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Radiation Therapy."],
        ),
        KnowledgeEntry(
            id="chemotherapy-principles",
            category="系统治疗",
            title="乳腺癌化疗原则概览",
            summary="化疗适用性取决于分期、分子分型、复发风险、治疗阶段和患者基础情况。",
            details=[
                "早期、局部晚期和转移性场景的化疗目的不同。",
                "蒽环类、紫杉类和其他方案选择需肿瘤专科结合风险收益判断。",
                "影像平台不能输出具体方案、剂量或疗程。",
            ],
            clinical_use="帮助智能体回答治疗相关问题时保持辅助和边界表达。",
            keywords=["化疗", "蒽环", "紫杉", "系统治疗", "风险收益"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="ovarian-function-suppression",
            category="药物治疗",
            title="卵巢功能抑制与绝经前内分泌治疗",
            summary="绝经前 HR 阳性乳腺癌可能涉及卵巢功能抑制联合内分泌治疗评估。",
            details=[
                "是否需要卵巢功能抑制取决于年龄、复发风险、治疗阶段和耐受性。",
                "常与他莫昔芬或芳香化酶抑制剂等策略讨论，但需专科制定。",
                "平台可提示绝经状态是关键缺失项，不能替代用药决策。",
            ],
            clinical_use="让病例导入和智能体输出更重视绝经状态。",
            keywords=["卵巢功能抑制", "绝经前", "HR阳性", "他莫昔芬", "芳香化酶抑制剂"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Hormone Therapy."],
        ),
        KnowledgeEntry(
            id="cdk46-inhibitors",
            category="药物治疗",
            title="CDK4/6 抑制剂相关知识",
            summary="CDK4/6 抑制剂常与内分泌治疗联合，用于部分 HR 阳性、HER2 阴性乳腺癌场景。",
            details=[
                "适用阶段包括部分晚期、转移性以及特定高危早期场景，需按指南和适应证评估。",
                "常见安全性问题包括骨髓抑制、肝功能异常或间质性肺病风险等，需监测。",
                "智能体只能提示可能进入专科评估，不能推荐具体药物和剂量。",
            ],
            clinical_use="扩充 HR 阳性治疗知识，使药物问题回答更规范。",
            keywords=["CDK4/6", "帕博西尼", "瑞波西尼", "阿贝西利", "HR阳性", "HER2阴性"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Targeted Drug Therapy."],
        ),
        KnowledgeEntry(
            id="antibody-drug-conjugates",
            category="药物治疗",
            title="抗体偶联药物 ADC",
            summary="ADC 将靶向抗体和细胞毒载荷结合，用于特定 HER2 或其他靶点相关乳腺癌场景。",
            details=[
                "T-DM1、T-DXd、戈沙妥珠单抗等药物适用人群和治疗线数不同。",
                "间质性肺病、骨髓抑制等风险需要专科监测。",
                "HER2-low 或三阴性等概念需要病理和既往治疗资料支持。",
            ],
            clinical_use="帮助智能体解释 ADC 概念和为什么需要完整病理/治疗史。",
            keywords=["ADC", "T-DM1", "T-DXd", "戈沙妥珠单抗", "HER2-low", "三阴性"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Targeted Drug Therapy."],
        ),
        KnowledgeEntry(
            id="immune-checkpoint-therapy",
            category="药物治疗",
            title="免疫检查点治疗边界",
            summary="免疫治疗在部分三阴性乳腺癌场景中具有价值，但依赖分期、风险和检测结果。",
            details=[
                "高危早期或转移性三阴性病例可能进入免疫治疗讨论。",
                "PD-L1、治疗阶段和联合化疗方案等信息会影响评估。",
                "免疫相关不良反应涉及多个器官系统，需专科监测。",
            ],
            clinical_use="用于回答三阴性和免疫治疗问题，同时避免给出个体化处方。",
            keywords=["免疫治疗", "检查点抑制剂", "三阴性", "PD-L1", "不良反应"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="bone-modifying-agents",
            category="支持治疗",
            title="骨转移和骨保护治疗",
            summary="骨转移或骨丢失风险管理可涉及双膦酸盐、地舒单抗和钙/维生素 D 评估。",
            details=[
                "骨转移治疗目标包括降低骨相关事件和控制症状，需要影像和临床确认。",
                "内分泌治疗或卵巢功能抑制可能影响骨密度，应考虑骨健康管理。",
                "用药需评估肾功能、低钙风险和颌骨坏死等安全性问题。",
            ],
            clinical_use="补充支持治疗知识，让智能体能提示骨健康和骨转移资料缺口。",
            keywords=["骨转移", "双膦酸盐", "地舒单抗", "骨密度", "骨健康"],
            references=["NCI Breast Cancer Treatment PDQ.", "American Cancer Society Treatment Options."],
        ),
        KnowledgeEntry(
            id="fertility-preservation",
            category="支持治疗",
            title="年轻患者生育力保护沟通",
            summary="育龄期患者在系统治疗前可能需要生育力保护咨询。",
            details=[
                "化疗、内分泌治疗和卵巢功能抑制可能影响月经和生育计划。",
                "治疗前应尽早沟通卵子/胚胎冷冻等选择和时间窗口。",
                "智能体可提示需要专科咨询，但不应替代生殖医学或肿瘤专科建议。",
            ],
            clinical_use="用于年轻患者病例的辅助建议和缺失信息提示。",
            keywords=["生育力保护", "年轻患者", "化疗", "卵巢功能", "治疗前咨询"],
            references=["NCI Breast Cancer Treatment PDQ."],
        ),
        KnowledgeEntry(
            id="screening-age-interval",
            category="筛查与风险因素",
            title="普通风险人群筛查年龄和间隔",
            summary="筛查建议随国家和组织不同而变化，应按当地规范和个体风险执行。",
            details=[
                "USPSTF 建议 40 至 74 岁女性每两年进行乳腺癌筛查乳腺 X 线摄影。",
                "高危人群可能需要更早或更强化筛查，由专科根据风险评估决定。",
                "平台应把筛查建议表达为参考信息，避免替代当地指南。",
            ],
            clinical_use="用于知识库展示筛查背景，并提醒按本地规范执行。",
            keywords=["筛查", "40岁", "74岁", "乳腺X线", "USPSTF", "普通风险"],
            references=["USPSTF Breast Cancer Screening Recommendation 2024.", "NCI Breast Cancer Screening PDQ."],
        ),
        KnowledgeEntry(
            id="tomosynthesis-screening",
            category="影像检查选择",
            title="数字乳腺断层摄影 DBT",
            summary="DBT 可作为乳腺 X 线筛查或诊断的一种技术，有助于减少组织重叠影响。",
            details=[
                "DBT 对致密乳腺或组织重叠造成的假影有一定帮助。",
                "钙化、结构扭曲和双侧比较仍需由放射科系统判读。",
                "超声 AI 平台不能替代钼靶或 DBT 的筛查作用。",
            ],
            clinical_use="帮助医生解释超声、钼靶和 DBT 的互补关系。",
            keywords=["DBT", "数字乳腺断层", "钼靶", "组织重叠", "筛查"],
            references=["ACR BI-RADS Manual v2025.", "NCI Breast Cancer Screening PDQ."],
        ),
        KnowledgeEntry(
            id="mri-high-risk-screening",
            category="影像检查选择",
            title="高危人群乳腺 MRI 筛查",
            summary="乳腺 MRI 常用于遗传高危或特定高风险人群筛查和范围评估。",
            details=[
                "BRCA 等遗传高危人群可能进入 MRI 强化筛查讨论。",
                "MRI 敏感性较高，但也可能增加假阳性和额外取材。",
                "是否采用 MRI 需结合个人风险、既往病史和当地可及性。",
            ],
            clinical_use="用于智能体解释为什么高危人群可能需要 MRI，而普通风险不默认追加。",
            keywords=["MRI", "高危筛查", "BRCA", "假阳性", "范围评估"],
            references=["NCI Breast Cancer Screening PDQ.", "ACR Appropriateness Criteria Supplemental Screening."],
        ),
        KnowledgeEntry(
            id="supplemental-ultrasound-screening",
            category="影像检查选择",
            title="致密乳腺补充超声筛查",
            summary="致密乳腺中补充超声可增加检出，但也可能增加假阳性和不必要取材。",
            details=[
                "USPSTF 认为致密乳腺阴性钼靶后补充超声或 MRI 的健康结局证据仍不足。",
                "ACR 等影像建议会根据风险水平和乳腺密度分层考虑补充筛查。",
                "平台应提示患者与医生讨论个体风险，而不是固定推荐补充超声。",
            ],
            clinical_use="让筛查相关回答更平衡，兼顾检出率和假阳性。",
            keywords=["致密乳腺", "补充超声", "假阳性", "MRI", "USPSTF", "ACR"],
            references=["USPSTF Breast Cancer Screening Recommendation 2024.", "ACR Appropriateness Criteria Supplemental Screening.", "CDC About Dense Breasts."],
        ),
        KnowledgeEntry(
            id="risk-models-chemoprevention",
            category="筛查与风险因素",
            title="风险模型与药物预防讨论",
            summary="Gail、Tyrer-Cuzick 等风险模型可辅助筛查强度和预防讨论，但不能替代临床判断。",
            details=[
                "模型输入包括年龄、家族史、既往病理、遗传因素和生育史等。",
                "部分高风险人群可由专科评估药物预防或强化筛查。",
                "风险模型结果需要解释不确定性和适用人群限制。",
            ],
            clinical_use="帮助病例导入提示风险字段，支持智能体提出结构化补充问题。",
            keywords=["风险模型", "Gail", "Tyrer-Cuzick", "药物预防", "强化筛查"],
            references=["NCI Breast Cancer Risk Assessment Tool.", "NCI Breast Cancer Screening PDQ."],
        ),
    ]
)


def _tokens(text: str) -> set[str]:
    """Tokenize Chinese/English mixed clinical text for simple local retrieval."""
    chunks = findall(r"[A-Za-z0-9+\-/]+|[\u4e00-\u9fff]{2,}", text.lower())
    return set(chunks)


def list_knowledge_entries(category: str | None = None) -> list[KnowledgeEntry]:
    """Return all knowledge entries, optionally filtered by category."""
    if not category:
        return KNOWLEDGE_ENTRIES
    return [entry for entry in KNOWLEDGE_ENTRIES if entry.category == category]


def search_knowledge(query: str, *, limit: int = 5) -> list[KnowledgeEntry]:
    """Return knowledge entries ranked by keyword and text overlap."""
    normalized_query = query.strip().lower()
    if not normalized_query:
        return KNOWLEDGE_ENTRIES[:limit]

    query_tokens = _tokens(normalized_query)
    results: list[KnowledgeSearchResult] = []
    for entry in KNOWLEDGE_ENTRIES:
        searchable = " ".join(
            [
                entry.category,
                entry.title,
                entry.summary,
                entry.clinical_use,
                " ".join(entry.keywords),
                " ".join(entry.details),
            ]
        ).lower()
        score = 0
        for keyword in entry.keywords:
            if keyword.lower() in normalized_query:
                score += 8
        if normalized_query in searchable:
            score += 5
        score += len(query_tokens.intersection(_tokens(searchable)))
        if score > 0:
            results.append(KnowledgeSearchResult(entry=entry, score=score))

    ranked = sorted(results, key=lambda item: (-item.score, item.entry.category, item.entry.title))
    return [item.entry for item in ranked[:limit]]


def build_knowledge_context(entries: list[KnowledgeEntry]) -> str:
    """Format retrieved entries for LLM context injection."""
    sections: list[str] = []
    for index, entry in enumerate(entries, start=1):
        details = "\n".join(f"- {detail}" for detail in entry.details)
        sections.append(
            f"[{index}] {entry.title}（{entry.category}）\n"
            f"摘要：{entry.summary}\n"
            f"临床用途：{entry.clinical_use}\n"
            f"要点：\n{details}"
        )
    return "\n\n".join(sections)
