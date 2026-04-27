## 一、认知

### 1. AI 科研绘图的适用场景

- **定量数据图**：正确的做法是继续使用Python、Origin等专业软件作图，**AI仅可用于提供配色建议或编写绘图代码**。
- **实证影像图**：此类图像通常不允许进行包括AI去噪或放大等生成式填充修改，**AI仅允许全图线性的亮度/对比度调整**。
- **定性示意图**：理解复杂的结构化文本，将抽象逻辑快速转化为高质量的视觉草案，**为科研人员提供成熟的构图思路与风格参考，从而显著降低后续人工绘制与修改的时间成本**。
- 优秀的科研图需要同时满足三个核心维度：**科学性、逻辑性和艺术性**。这三者共同决定了图片的专业度与说服力。

### 2. 领域风格的定向适配

- 一种是偏向物理、计算机与人工智能领域的**极简主义风格**。这类插图偏好**扁平化设计与矢量感**。它们多使用低饱和度的莫兰迪色系，线条硬朗清晰，强调拓扑结构与逻辑流的直接表达，排斥不必要的三维渲染与光影修饰。
- 另一种则是偏向生物、医学与材料科学的**拟真风格**。这类插图更青睐**三维质感与真实环境**的复现。它们强调微观细节的丰富度，常利用环境光遮蔽(AO)与次表面散射(SSS)等渲染技术来模拟细胞、蛋白质或纳米材料的真实质感，通过极强的视觉冲击力来营造沉浸式的微观世界。

## 二、工具

不需要成为提示词专家，只需要掌握两种足以覆盖绝大多数科研绘图场景的**对话模式**：结构化生成的**文生图**，以及引入视觉约束的**图生图**。

### 1. 文生图

- 需要尽量减少模型的自由想象空间，将其严格约束在明确的目标边界内。

- 推荐采用一种模块化、分层级的指令组织方式。用户可以显式地向模型声明绘图的关键约束条件，如整体风格、空间布局、拓扑关系。

- 高度逻辑化的输入方式，可以在很大程度上抑制模型的幻觉行为，确保生成结果在科学逻辑上是自洽的。可以直接在提示词中指定HEX色值。

- 在实际工作流程中，提示词的初稿完全可以借助大语言模型自动生成

### 2. 图生图

- 引入参考图将成为决定体验上限的关键。它并不仅仅是一个辅助选项，更是科研绘图从可用迈向好用的重要分界点。
- **参考图主要承担两种核心功能：提供空间拓扑的结构骨架，以及定义视觉特征的目标风格。**
- 无论是用鼠标在画图工具中随手勾勒的线条，在PPT中粗略摆放的框架，还是在纸上快速画下的草图，只要其拓扑结构是正确的，即明确了谁与谁相邻、谁指向谁，AI 就可以将其视为整个画面的“骨架”。
- 通过上传这样的参考图，原本抽象而冗长的文字描述便获得了清晰的空间约束。模型会保留草图中的结构关系，同时对线条、配色和质感进行系统化优化。一张结构正确但外观粗糙的草图，往往胜过冗长且难以精确约束空间关系的文字描述。
- 除了结构参考，图生图模式还可以用于风格迁移。遇到配色、线条与整体质感都高度符合目标期刊风格的插图，可以将其作为风格上的参考图提供给模型。
- 参考图像与文字指令之间应保持一致，避免在结构或风格层面出现相互冲突的约束。

### 3. 前处理工具

| 工具名称                 | 适用场景                    | 说明 / 链接                                                  |
| :----------------------- | :-------------------------- | :----------------------------------------------------------- |
| **Excalidraw / draw.io** | 轻量级草图与拓扑结构绘制    | 绘制逻辑骨架作为图生图(Image-to-Image)的骨架参考。 [Excalidraw 在线端](https://excalidraw.com/) \| [draw.io网页版](https://app.diagrams.net/) |
| **PPT / Visio**          | 常见形状结构勾勒            | 本地客户端内快速构建论文主体的基础形状布局。                 |
| **colorgram.py**         | 配色提取工具 (Python开源库) | 辅助从高水平论文插图中提炼稳定审美的提取 HEX 色值。[colorgram.py](https://github.com/obskyr/colorgram.py) |

#### （1）草图绘制工具

- 用于绘制草图的工具并不存在统一的最优解，真正重要的是能否够快速、明确地表达结构意图。
- 无论是操作系统自带的画图软件，Power Point中最基础的形状与箭头工具，还是Excalidraw、draw.io这类轻量级流程或示意图工具，都足以承担这一阶段的任务。在很多情况下，甚至只需要一张A4纸和一支笔，就能完成对结构关系的初步表达。

#### （2）配色提取工具

- 通过配色提取工具，则可以直接从这些插图中获取对应的HEX色值，用于后续的提示词编写或统一整篇论文的视觉风格。

### 4. 后处理工具

| 工具名称                     | 适用场景                  | 说明 / 链接                                                  |
| :--------------------------- | :------------------------ | :----------------------------------------------------------- |
| **gemini-watermark-remover** | 图像去水印 (开源项目)     | 通过反向 Alpha 混合算法清理水印，还原基础素材。[GitHub 相关开源仓](https://github.com/GargantuaX/gemini-watermark-remover) |
| **Real-ESRGAN 系列模型**     | 高清放大与超分辨率模型    | 在自动描摹前提升低分辨率位图的锐度与细节。[GitHub 项目](https://github.com/xinntao/Real-ESRGAN) |
| **Vectorizer**               | 在线基础矢量化            | 快速将位图转化为基础 SVG 文件。[Vectorizer在线版](https://vectorizer.ai/) |
| **Adobe Illustrator / AI**   | 专业矢量化“图像描摹”      | 高质量转换，支持精细参数调节（推荐设置：颜色20%、路径50%、边角30%）。 |
| **ChemDraw / VESTA**         | 分子构型与晶体结构生成    | 化学生生物学必装专业工具，用于生成结构精确的局部模块或组件。 |
| **Matplotlib**               | 代码辅助矢量图形绘制      | 适合绘制拥有严格参数控制的科学坐标架构体系。[官方文档](https://matplotlib.org/) |
| **Edit-Banana / Paper2Any**  | 基于 VLM/OCR 的结构化生成 | 尝试将静态图表转化为DrawIO等可编辑文件的预研项目。[Edit-Banana](https://github.com/BIT-DataLab/Edit-Banana);[Paper2Any](https://github.com/OpenDCAI/Paper2Any) |

#### （1）去水印工具

#### （2）高清放大工具

#### （3）矢量化和可编辑工具

## 三、提示词

### 1. 定制化元提示词

- 当通用领域的提示词无法满足特定交叉学科需要时，可以利用本提示词，附带心仪的一张“顶级发表图”（作为视觉参考垫图），让 LLM 反向破译并适配领域的排版与专用语义。
- **让AI基于提供的参考图进行布局与风格层面的逆向分析，并结合目标领域的知识表达范式，对既有的基础提示词进行针对性微调**。快速得到一套兼具参考图视觉特征与目标领域逻辑的定制化提示词。

```textplain
你是一名资深提示词工程专家，熟悉学术论文插图的生成逻辑，对计算机领域以及【你所在的领域名称】领域的研究范式、图示风格与视觉表达均有系统理解。 
 
我将提供一个目标领域插图的成品示例。该示例代表我希望最终生成结果所遵循的整体风格。请你对该示例进行逆向分析，重点关注以下方面：整体布局结构、信息层级组织方式、模块之间的空间关系、配色方案及其在信息表达中的作用、图形元素的抽象程度与表达习惯。 
 
在此基础上，请对下方给定的两个提示词分别进行微调优化，使其在实际使用时，能够稳定生成与示例在视觉风格与表达逻辑上高度一致的插图提示词。 
 
需要注意的是，这两条原始提示词均是为计算机领域论文内容抽取与示意图绘制所设计的。你的任务是将其调整为适用于【所在的领域名称】的版本。请保持原有提示词的整体结构、步骤逻辑和控制维度，仅结合【所在的领域名称】常见的图示布局特征、学科语义重点以及视觉表达习惯进行针对性的细化和替换。 
 
最终输出应为两条对应的完整、可直接使用的提示词，其生成结果在风格上与目标示例保持一致，同时在内容表达上自然适配【所在的领域名称】。 

---  
提示词A：  【复制粘贴“阶段一：逻辑构建”完整提示词】  
---  
提示词B：  【复制粘贴“阶段二：绘图渲染”完整提示词】
```

### 2. 数据领域专属提示词

> **核心特征**：偏向抽象的拓扑结构，强调信息流、网络输入输出关系。

#### （1）阶段一：逻辑构建

- **对论文内容进行重组与抽象，用以明确插图应当呈现的核心信息、模块划分方式以及整体画面设计思路。**
- 提示词中通常会引入对应领域的顶级会议或期刊作为风格参照，同时提供若干常见的布局原型供模型选择，以降低生成结果的不确定性。
- 如果对所要绘制内容的逻辑关系已经非常清楚，可以将现有的文字说明、流程图、伪代码或相关笔记，与论文正文一并提供给LLM，作为结构构建的输入依据；如果对本领域中常见的插图布局形式缺乏把握，则可以先收集几篇具有代表性的论文插图，让LLM对其进行归纳与总结，再将得到的布局类型与设计特征补充进提示词中。
- **为避免最终成图与逻辑脱节，建议在进入阶段二前，仔细检查Schema中的结构与风格描述是否准确。**

```textplain
# Role 
你是一位 CVPR/NeurIPS 顶会的**视觉架构师**。你的核心能力是将抽象的论文逻辑转化为**具体的、结构化的、几何级的视觉指令**。 
 
# Objective 
阅读我提供的论文内容，输出一份 **[VISUAL SCHEMA]**。这份 Schema 将被直接发送给 AI 绘图模型，因此必须使用**强硬的物理描述**。 
 
# Phase 1: Layout Strategy Selector (关键步骤：布局决策) 
在生成 Schema 之前，请先分析论文逻辑，从以下**布局原型**中选择最合适的一个（或组合）： 
1.  **Linear Pipeline**: 左→右流向 (适合 Data Processing, Encoding-Decoding)。 
2.  **Cyclic/Iterative**: 中心包含循环箭头 (适合 Optimization, RL, Feedback Loops)。 
3.  **Hierarchical Stack**: 上→下或下→上堆叠 (适合 Multiscale features, Tree structures)。 
4.  **Parallel/Dual-Stream**: 上下平行的双流结构 (适合 Multi-modal fusion, Contrastive Learning)。 
5.  **Central Hub**: 一个核心模块连接四周组件 (适合 Agent-Environment, Knowledge Graphs)。 
 
# Phase 2: Schema Generation Rules 
1.  **Dynamic Zoning**: 根据选择的布局，定义 2-5 个物理区域 (Zones)。不要局限于 3 个。 
2.  **Internal Visualization**: 必须定义每个区域内部的“物体” (Icons, Grids, Trees)，禁止使用抽象概念。 
3.  **Explicit Connections**: 如果是循环过程，必须明确描述 "Curved arrow looping back from Zone X to Zone Y"。 
 
# Output Format (The Golden Schema) 
请严格遵守以下 Markdown 结构输出： 
 
---BEGIN PROMPT--- 
 
[Style & Meta-Instructions]  High-fidelity scientific schematic, technical vector illustration, clean white background, distinct boundaries, academic textbook style. High resolution 4k, strictly 2D flat design with subtle isometric elements. 
 
[LAYOUT CONFIGURATION] 
* **Selected Layout**: [例如：Cyclic Iterative Process with 3 Nodes] 
* **Composition Logic**: [例如：A central triangular feedback loop surrounded by input/output panels] 
* **Color Palette**: Professional Pastel (Azure Blue, Slate Grey, Coral Orange, Mint Green). 
 
[ZONE 1: LOCATION - LABEL] 
* **Container**: [形状描述, e.g., Top-Left Panel] 
* **Visual Structure**: [具体描述, e.g., A stack of documents] 
* **Key Text Labels**: "[Text 1]" 
 
[ZONE 2: LOCATION - LABEL] 
* **Container**: [形状描述, e.g., Central Circular Engine] 
* **Visual Structure**: [具体描述, e.g., A clockwise loop connecting 3 internal modules: A (Gear), B (Graph), C (Filter)] 
* **Key Text Labels**: "[Text 2]", "[Text 3]" 
 
[ZONE 3: LOCATION - LABEL]  ... (Add Zone 4/5 if necessary based on layout) 
 
[CONNECTIONS] 
1.  [描述连接线, e.g., A curved dotted arrow looping from Zone 2 back to Zone 1 labeled "Feedback"] 
2.  [描述连接线, e.g., A wide flow arrow from Zone 2 to Zone 3] 
 
---END PROMPT--- 
 
# Input Data 
【论文相关内容】
```

#### （2）阶段二：绘图渲染

- 这一阶段，核心目标是抑制模型的幻觉行为与过度发挥，确保生成结果尽可能精准地还原Schema中定义的空间位置与几何关系。绘图提示词应重点强调**空间布局、视觉元素类型与整体风格约束**。同时，结合具体学科领域的顶级期刊审美标准，**对画面的风格进行再次确认**。

```textplain
**Style Reference & Execution Instructions:** 
 
1.  **Art Style (Visio/Illustrator Aesthetic):** 
    Generate a **professional academic architecture diagram** suitable for a top-tier computer science paper (CVPR/NeurIPS). 
    * **Visuals:** Flat vector graphics, distinct geometric shapes, clean thin outlines, and soft pastel fills (Azure Blue, Slate Grey, Coral Orange). 
    * **Layout:** Strictly follow the spatial arrangement defined below. 
    * **Vibe:** Technical, precise, clean white background. NOT hand-drawn, NOT photorealistic, NOT 3D render, NO shadows/shading. 
 
2.  **CRITICAL TEXT CONSTRAINTS (Read Carefully):** 
    * **DO NOT render meta-labels:** Do not write words like "ZONE 1", "LAYOUT CONFIGURATION", "Input", "Output", or "Container" inside the image. These are structural instructions for YOU, not text for the image. 
    * **ONLY render "Key Text Labels":** Only text inside double quotes (e.g., "[Text]") listed under "Key Text Labels" should appear in the diagram. 
    * **Font:** Use a clean, bold Sans-Serif font (like Roboto or Helvetica) for all labels. 
 
3.  **Visual Schema Execution:** 
    Translate the following structural blueprint into the final image: 
 
【阶段一输出的全部内容】
```

#### （3）阶段三：交互迭代

- 对初稿进行有针对性的修改与优化，依然是不可或缺的步骤。从实践经验来看，常见问题主要可以归为两类：**整体布局正确但局部细节或风格存在偏差**，以及**整体布局本身出现错误**。
- 当插图的整体结构、模块关系与空间布局已经满足预期，仅在局部元素、配色或视觉风格上存在瑕疵时，建议采用自然语言编辑策略进行修正。
- 如果文字错误较为集中或影响整体观感，更稳妥的处理方式是直接移除图中所有文字标签，在后处理阶段通过矢量编辑或排版软件统一添加说明文字。
- 在每一次生成或修改完成后，立即去除当次产生的可见水印，避免累积。
- 如果发现生成结果在结构层面出现明显偏差，更有效的做法是回到阶段一，对Schema进行检查与修订，重点确认是否选择了合适的布局策略。
- 可以直接与LLM进行交互，要求它基于你的修正意见重新生成Schema作为新的结构蓝图，再重新执行阶段二的绘图渲染流程。
- **当一张插图在整体结构和风格层面已经达到较高完成度时，并不建议频繁重新生成。**
- 在实际工作流程中，**结构性或幅度较大的调整更适合通过优化阶段一的提示词来完成；细节层面的微调则可以直接通过自然语言编辑完成**。

### 3. 注意点

#### （1）复杂图形的拆解与生成

- 在科研写作中，真正需要引入模块化策略的复杂图形，主要集中在两类典型形态：复杂长图，以及由多个子图构成的复合型大图。
- 更稳妥的做法是**在逻辑层面先将整张长图拆分为输入、处理、输出等若干相对独立的子模块**。
- 复合图更适合**在生成阶段将每一个子图视为独立模块分别绘制**，使每个子图都能在自身最合适的视觉范式下完成表达，**再通过统一的风格约束与人工排版完成整体整合**。

#### （2）基于母图的风格锚定

- 在完成第一张令人满意的插图后，后续工作的重点便从“创造”转向了“对齐”。
- 更有效的做法，是将你最满意的那张生成结果作为“母图”引入后续生成流程。通过图生图的方式，在生成新的模块或子图时，要**求模型严格参考该母图的笔触密度、配色策略、线条粗细与整体视觉节奏**。这种基于具体图像特征的风格锚定方式，能够显著降低模块化生成过程中出现风格漂移的风险。

#### （3）图像结果的矢量化

- 将图片拖入Illustrator主界面后，点击右侧面板中的“图像描摹”按钮。首先在预设一栏选择“高保真度图片”以加载系统默认的彩色追踪方案，此时手动修改参数(推荐参数：颜色20%、路径50%、边角30%、杂色10px)，预设会自动切换为“自定”。处理完成后，依次点击“扩展”与“取消编组”，即可对各个矢量对象进行灵活拖动或单独编辑。
- 在实际测试中，描摹效果对参数设置和图像风格高度敏感，需要通过反复微调颜色、路径、边角、杂色，才能获得相对可用的结果，也需要一定的反复试验。
- 在执行自动描摹前，建议先通过高清放大工具将原始位图进行预处理。理论上，输入的位图分辨率越高、边缘锐度越强，自动描摹生成的路径就越平滑且贴合原貌。

#### （4）把流程固化为 **skill**

- 可以使用中文的输入和输出，但国外模型对英文的输入和输出效果会更好，在使用时可以可以两种方法都尝试，根据实际情况进行取舍。

### 4. 效果尝试

我随便找了一篇 2018 年的深度学习论文，看看效果会怎么样。

![](https://cdn.jsdelivr.net/gh/don2vito/picgo_warehouse@main/202604230904101.png)

输入阶段一的提示词，返回提取出的论文相关内容。

```textplain
---BEGIN PROMPT---

[Style & Meta-Instructions]
High-fidelity scientific schematic, technical vector illustration, clean white background, distinct boundaries, academic textbook style. High resolution 4k, strictly 2D flat design with subtle isometric elements. Strict geometric alignment, no photographic textures.

[LAYOUT CONFIGURATION]

Selected Layout: Hierarchical Stack combined with Central Hub.

Composition Logic: Bottom-up processing flow. Zone 1 (Input Embedding) at bottom, feeding up into Zone 2 (Central Transformer Tower). Zone 2 outputs to two parallel branches in Zone 3 (Pre-training Heads). Zone 4 (Downstream Tasks) is positioned at the top, connected via a dashed vertical line indicating the "Fine-Tuning" parameter transfer.

Color Palette:

Background: Pure White (#FFFFFF)

Zone 1: Soft Amber (#FDEBD0)

Zone 2: Azure Blue (#D6EAF8)

Zone 3: Mint Green (#D5F5E3)

Zone 4: Slate Grey (#EAECEE)

Arrows/Connectors: Coral Orange (#E59866)

[ZONE 1: BOTTOM - INPUT REPRESENTATION]

Container: A wide rectangular panel spanning the lower section. The panel has sharp 90-degree corners and a thin Coral Orange outline.

Visual Structure: Inside the panel, three distinct horizontal layers (plates) stacked precisely on top of one another.

Layer 1 (Bottom): A grid of small, uniform squares (representing WordPiece Token Embeddings).

Layer 2 (Middle): A single wide bar split vertically into two halves colored different shades of Amber (representing Segment Embedding A and B).

Layer 3 (Top): A waveform-like zigzag line or a gradient bar fading from left to right (representing Positional Embeddings).

Key Text Labels: "Token Embeddings", "Segment Embeddings", "Position Embeddings".

Output Node: A large plus symbol "+" positioned to the right of the three stacked layers, followed by a single unified block labeled "Input Vector".

[ZONE 2: CENTER - BIDIRECTIONAL TRANSFORMER ENCODER]

Container: A dominant vertical tower constructed from stacked horizontal plates. The tower sits directly above the unified block of Zone 1.

Visual Structure:

The tower consists of repeating identical layers (T=1 to T=24). We explicitly visualize 3-4 representative plates labeled "Transformer Block" to imply depth.

Internal Geometry (Crucial): Within each block, visualize a dense Bipartite Graph (two columns of nodes). The left column represents "Left Context Tokens", the right column represents "Right Context Tokens". Connect every node on the left to every node on the right with fine, straight lines (Full Attention). Crucially, draw a curved, bidirectional arrow wrapping around the entire block, labeled "Context Fusion".

On the left side of the tower, attach a narrow vertical panel containing a vertical sequence of the letters "T₁", "T₂", "...", "Tₙ" (representing final hidden states).

Key Text Labels: "L=24 Layers", "H=1024 Hidden", "A=16 Heads", "Deep Bidirectional Self-Attention".

[ZONE 3: TOP-LEFT & TOP-RIGHT - PRE-TRAINING HEADS]

Container: Two distinct rectangular boxes positioned immediately above Zone 2, branching left and right like a "Y" junction.

Left Branch (Masked LM):

Visual Structure: A text string "[CLS] Tok 1 [MASK] Tok 3 [SEP]" where the "[MASK]" token is highlighted with a glowing Coral Orange border. An arrow points from this token to a Softmax Probability Cloud (a collection of overlapping transparent circles containing words like "Hairy", "Apple", "Cat").

Key Text Labels: "Task #1: Masked LM (MLM)", "Predict Masked Token".

Right Branch (Next Sentence Prediction):

Visual Structure: Two document icons labeled "Sentence A" and "Sentence B". A simple switch icon (toggle) between them points either to a Green Checkmark (IsNext) or a Red Cross (NotNext).

Key Text Labels: "Task #2: NSP", "50% IsNext / 50% Random".

[ZONE 4: TOP - FINE-TUNING DOWNSTREAM TASKS]

Container: A floating, wide rectangular panel spanning the top of the image. It is visually disconnected from the lower structure but connected by a single thick Dashed Vertical Conduit coming from the central tower (Zone 2) labeled "Transfer Learned Parameters".

Visual Structure: A horizontal gallery of four distinct mini-icons:

Two overlapping speech bubbles with a double-ended arrow between them (MNLI / Sentence Pair).
A magnifying glass over a paragraph with start/end brackets around a phrase (SQuAD QA).
A single document with a tag attached to a specific word (NER Tagging).
A thumbs up / thumbs down toggle (Sentiment SST-2).
Key Text Labels: "GLUE", "SQuAD", "NER", "SWAG".

[CONNECTIONS]

Zone 1 to Zone 2: A thick, solid upward arrow transitioning from Amber to Azure Blue.

Zone 2 to Zone 3: Two branching solid arrows from the top of the tower to the left and right heads.

Zone 2 to Zone 4: A dashed, vertical line (conduit) running from the center of the Transformer tower straight up through the gap between Zone 3 Left and Right, terminating at the bottom of Zone 4. Labeled "Fine-Tuning (All Parameters Updated)".

Internal Loop (Zone 2): A distinct Coral Orange curved arrow starting at the right side of a Transformer Block, arching up and over to the left side, labeled "Bidirectional Context Flow".

---END PROMPT---
```

把以上返回内容，贴入阶段二的提示词，输出图片初稿，效果还真不错呢。

![](https://cdn.jsdelivr.net/gh/don2vito/picgo_warehouse@main/202604230909801.png)

在此基础上，后续再进行调整和迭代。