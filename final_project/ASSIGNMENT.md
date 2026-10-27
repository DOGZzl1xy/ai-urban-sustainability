# Final Project 作业要求

MUSA 6950 — AI for Urban Sustainability

> 原文为课程发布的说明（见文末），这里整理成清单。说明里只详细写了 proposal 的要求；report 和 presentation 的具体格式、篇幅、截止日期都没有给出，标为 **待确认**。

---

## 1. 项目总体要求

期末项目是一个**完整项目**，不只是 proposal。要用课上学到的 AI、机器学习和计算方法解决一个现实问题，完整走完下面几步：

- [ ] **找问题**：确定一个有意义的研究问题或实际问题
- [ ] **定方法**：设计基于 AI 的方法
- [ ] **做分析**：分析相关数据
- [ ] **讲结果**：用两种形式交流发现
  - 书面报告（written report）
  - 口头展示（oral presentation）

完整的动手过程应包括：problem formulation → data preparation → model development → evaluation → interpretation。

## 2. 个人 / 小组

- 鼓励**独立完成**，这样能完整经历 AI 应用的全过程
- 只有当项目范围或复杂度确实需要合作时才组队
- 组队时，每个成员都要有**实质性且能清楚区分**的贡献

## 3. 选题范围

题目自由，只要和人工智能、城市系统、地理空间分析、环境可持续性或其他现实应用相关。课程列出的方向（不限于此）：

| 方向 | 示例 |
|---|---|
| Urban Environment and Climate | 城市树木制图、热暴露预测、土地覆盖分析、气候适应策略评估 |
| Climate Risk and Resilience | 洪涝建模、机器学习、遥感评估洪涝、极端高温等灾害 |
| Remote Sensing and Computer Vision | 航拍、卫星、街景影像的目标检测、语义分割、城市要素提取 |
| **Urban Mobility and Infrastructure** | **交通分析、交通预测、基础设施监测、城市可达性** ← 本项目所属方向 |
| Urban Streetscape and Built Environment | 用计算机视觉量化绿化、步行性、街道品质 |
| Public Health and Environmental Justice | 环境条件、基础设施、健康结果和社会经济不平等的关系 |
| Ecology and Wildlife Conservation | 栖息地制图、生物多样性监测、野生动物保护 |
| LLMs and Generative AI | 文本分析、规划文件、环境政策、地理空间 AI agent |
| High-Performance and GPU Computing | GPU 加速、深度学习、可扩展的城市与环境计算 |

## 4. 交付物

### 4.1 Proposal（1–2 页）

必须包含 5 部分：

- [ ] **Project Title and Student Name(s)**：描述性标题；如果组队，列出所有成员
- [ ] **Background and Research Question**：简要介绍现实问题、说明为什么重要、明确要回答的具体问题或挑战
- [ ] **Data Sources**：计划使用的数据集，包括来源、地理覆盖范围和相关特征
- [ ] **Proposed AI Methods**：打算用的 AI / ML / 深度学习 / 计算方法，以及为什么适合这个问题
- [ ] **Expected Outcomes**：预期成果，比如预测模型、地图、可视化、分析结论或原型应用

提交后老师会给反馈。数据可得性、方法选择、可行性方面有疑问，可以找老师讨论。

### 4.2 Written Report

- [ ] 内容：应覆盖问题、数据、方法、评估、结论和解释
- 格式 / 篇幅 / 截止日期：**待确认**

### 4.3 Oral Presentation

- [ ] 内容：展示研究问题、方法和主要发现
- 时长 / 形式 / 日期：**待确认**

---

## 5. 本项目对应情况

- 题目：**Teaching Aerial Imagery to See Height: Tracking Philadelphia's Gentle Densification with LiDAR-Supervised Foundation Models**（个人项目，Xuanyu Zhou）
- Proposal：`proposal/`（2 页 PDF）

| Proposal 要求 | 对应位置 |
|---|---|
| Title and Name | 标题 + 第二行 |
| Background and Research Question | §1，RQ1–RQ3 |
| Data Sources | §2 |
| Proposed AI Methods | §3 |
| Expected Outcomes | §5（§4 是可行性检验结果） |

---

## 附：课程原文

> The final project provides an opportunity for students to apply the artificial intelligence (AI), machine learning, and computational techniques learned in this course to address real-world challenges. Students are expected to identify a meaningful research question or practical problem, develop an AI-based approach, analyze relevant data, and communicate their findings through a written report and oral presentation.
>
> Students are encouraged to complete their projects independently, as this will provide the most comprehensive hands-on experience in applying AI techniques, from problem formulation and data preparation to model development, evaluation, and interpretation. However, group projects are permitted when collaboration is necessary due to the scope or complexity of the proposed work. For group projects, each member should make a substantial and clearly identifiable contribution.
>
> **Project Topics**
>
> Project topics are flexible and can address any issue related to artificial intelligence, urban systems, geospatial analysis, environmental sustainability, or other real-world applications.
>
> Potential topics include, but are not limited to:
>
> - Urban Environment and Climate: Using AI to map urban trees, predict urban heat exposure, analyze land cover, or evaluate climate adaptation strategies.
> - Climate Risk and Resilience: Applying flood modeling, machine learning, or remote sensing to assess flooding, extreme heat, and other environmental hazards.
> - Remote Sensing and Computer Vision: Using aerial imagery, satellite imagery, or street-view images for object detection, semantic segmentation, and urban feature extraction.
> - Urban Mobility and Infrastructure: Applying AI to transportation analysis, traffic prediction, infrastructure monitoring, or urban accessibility.
> - Urban Streetscape and Built Environment: Quantifying urban greenery, walkability, street quality, and other built-environment characteristics using computer vision.
> - Public Health and Environmental Justice: Investigating relationships between environmental conditions, urban infrastructure, health outcomes, and socioeconomic inequalities.
> - Ecology and Wildlife Conservation: Using AI for habitat mapping, biodiversity monitoring, or wildlife protection.
> - Large Language Models (LLMs) and Generative AI: Applying LLMs to text analysis, urban planning documents, environmental policy, or developing AI agents for geospatial applications.
> - High-Performance and GPU Computing: Exploring GPU acceleration, deep learning, and scalable computational methods for urban and environmental analysis.
>
> Students should submit a brief project proposal outlining their planned work. The proposal should include:
>
> 1. Project Title and Student Name(s): Provide a descriptive title and identify all team members, if applicable.
> 2. Background and Research Question: Briefly introduce the real-world problem, explain its importance, and identify the specific question or challenge your project will address.
> 3. Data Sources: Describe the datasets you plan to use, including their sources, geographic coverage, and relevant characteristics.
> 4. Proposed AI Methods: Explain the AI, machine learning, deep learning, or computational techniques you intend to apply and why they are appropriate for the problem.
> 5. Expected Outcomes: Describe the anticipated results, such as prediction models, maps, visualizations, analytical findings, or prototype applications.
>
> Suggested proposal length: 1–2 pages.
>
> Students who have already submitted their proposals will receive feedback and suggestions. You are encouraged to discuss your project ideas with the instructor, particularly if you need guidance on data availability, methodological choices, or project feasibility.
