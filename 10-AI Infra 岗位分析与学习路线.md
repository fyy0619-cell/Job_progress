# 10 · AI Infra 岗位分析与学习路线(2026-05-31 新增)

> **起草背景:** 2026-05-30 用户面试华为昇腾(杭州滨江)"AI + C++/Python 软件开发"实习岗,业务线含 **AI Infra & 大模型训推优化**(CANN/算子调优/异构调度) + AI Agent。
> 面完后用户问:"AI Infra 岗位如何?发展怎么样?完整学习路线?"——本文档为答复正稿。
>
> **写作原则(遵循 feedback_career_direction):** 严格对比、不单方面拔高当下风口、给出 EE 硕士可执行版,**避免再次踩"AI 风口偏见"的坑**。
>
> **时间线基准:** 3 年制学硕、2028 届、秋招 2027.9、毕业 2028.7(以 [09 文档](09-完整时间线修正%203年制学硕版.md) 为准)。

---

## 0. TL;DR(先看这一段)

1. **AI Infra ≠ 端侧 AI ≠ AI 应用算法**。它是**支撑大模型训练/推理跑得快、跑得稳、跑得省**的整套底层系统软件 —— CUDA/算子/编译器/分布式/调度/存储/网络。**离硬件比"端侧 AI"更近一层,但离 EE 硬件直觉更远一层**(战场是 GPU/NPU 芯片+编译器,不是 datasheet+寄存器)。
2. **AI Infra 是当下最贵、最热、天花板最高的方向之一,但门槛也极高:** 应届顶薪 60–100w+(SP/SSP),3–5 年 100–200w,头部公司(英伟达 DevTech / OpenAI / Anthropic / 华为昇腾 / 字节 AML / DeepSeek)的 staff/principal 200w+。**但坑也最深** —— CS 科班+OS/编译/HPC 三门课底子+ACM/OI/Kaggle 背景是默认入场券,EE 硕士裸跳风险大。
3. **对 EE 硕士的建议:不要把 AI Infra 当主线"裸冲",而是把它当作"方案 C(AI BSP/NPU 系统软件)的上行路径"**。先把方案 C 的 NPU 系统软件做出深度(应届 45–60w 稳),工作 2–3 年后**从 AI BSP 端横跳到 AI Infra**(算子/编译/runtime 的技能在两边是通的),薪资可上 100w+。这条路风险可控、且充分利用 EE 硬件背景做差异化。
4. **如果非要应届直冲 AI Infra:** 必须在 2026.7–2027.7 这一年补齐 **CUDA + 算子调优 + MLIR/编译器 + 分布式训练** 四件套,并且**至少有一段 AI Infra 方向的实习**(华为昇腾/字节 AML/商汤/旷视/摩尔线程/壁仞/沐曦/燧原任意一家)。否则秋招拼不过 CS 科班+ACM 背景的同分位竞争者。
5. **昇腾面试是一个高质量信号:** 不论这次实习是否拿到,华为昇腾这条线**值得长期保持联系**(它和方案 C 的 AscendC 算子开发是一脉相承的)。

---

## 1. 什么是 AI Infra(先澄清概念)

### 1.1 AI Infra 的边界

AI Infra(AI Infrastructure)是**让 AI 模型在硬件上跑起来、跑得高效、跑得规模化**的整套系统软件。它**不是**:
- ❌ 不是炼丹(那是算法岗/研究员)
- ❌ 不是部署个 ONNX 跑一下(那是 ML 工程师/部署岗)
- ❌ 不是写 PyTorch 代码训模型(那是算法工程师)

它**是**:
- ✅ 写 CUDA / AscendC / Triton kernel,让 GEMM/Attention/Softmax 跑得更快
- ✅ 写编译器 pass(MLIR/TVM/XLA),做 fusion/tiling/layout 优化
- ✅ 写 runtime 框架(vLLM / SGLang / TensorRT-LLM),管理 KV cache / 调度请求 / 推理服务
- ✅ 写分布式训练框架(Megatron / DeepSpeed / FSDP),做 TP/PP/EP/DP 切分
- ✅ 调优集群网络(NCCL/HCCL/RDMA/IB),搞定万卡训练的通信瓶颈
- ✅ 设计存储/数据加载(GDS/Alluxio),让 TB 级数据集喂得动 GPU

### 1.2 AI Infra 的六大子方向(选一个深耕,不要全抓)

| 子方向 | 核心技术 | 代表公司/团队 | EE 硕士契合度 |
|--------|----------|--------------|-------------|
| **算子 Kernel 优化** ★ | CUDA / AscendC / Triton / TensorCore / Tile / Warp 编程 | 英伟达 DevTech、华为昇腾算子组、字节 AML | ★★★★(最易切入) |
| **AI 编译器** | MLIR / TVM / XLA / Halide / 图优化 / 自动调优 | 谷歌 XLA、华为 MindSpore、智源、模力 | ★★(需要补编译原理) |
| **推理引擎/Serving** ★ | vLLM / SGLang / TensorRT-LLM / KV cache / PagedAttention / 调度 | OpenAI、Anthropic、DeepSeek、月之暗面、字节 | ★★★ |
| **分布式训练框架** | Megatron / DeepSpeed / FSDP / 通信原语(NCCL/HCCL) | OpenAI、Meta、字节、阿里 PAI、华为昇思 | ★★(需要分布式系统底子) |
| **集群/网络/存储** | RDMA / IB / NVLink / GDS / 调度器 / 万卡训练运维 | 阿里云、字节、华为云、AWS | ★★(更偏运维) |
| **NPU/GPU 系统软件** ★ | 驱动 + runtime + 编译器全栈(NPU 厂商专属) | 华为昇腾 CANN、寒武纪、地平线、壁仞、沐曦 | ★★★★★(与方案 C 直接接轨) |

> **EE 硕士最容易上手的两个子方向:** ★ 算子 Kernel 优化、★ NPU/GPU 系统软件。第二个**和方案 C 是同一条赛道,只是从端侧扩到了数据中心**。

---

## 2. AI Infra 发展怎么样(认真分析,不夸大)

### 2.1 薪资天花板(2026 市场)

| 等级 | 应届(校招) | 3 年(P6/T6) | 5 年(P7/T7) | 10 年(P8/staff) |
|------|------------|--------------|--------------|-------------------|
| **AI Infra 头部** | 60–100w+(SP/SSP) | 100–150w | 150–250w | 300–500w+ |
| **AI Infra 中部** | 40–60w | 60–100w | 100–150w | 150–250w |
| **方案 C(AI BSP/NPU 系统软件)** | 45–60w | 70–100w | 100–180w | 180–300w |
| **纯端侧 AI** | 50–70w | 70–100w | 100–150w | 150–200w |
| **Linux 驱动/BSP(纯)** | 30–45w | 50–70w | 70–100w | 100–150w |

**结论:**
- AI Infra **头部薪资明显高于方案 C 和纯端侧 AI**(尤其 5 年后差距拉大)
- 但 **AI Infra 中部 ≈ 方案 C**(45-60w 一档)—— **薪资优势主要在头部公司**
- 头部公司门槛极高(英伟达 DevTech / OpenAI / Anthropic / DeepSeek 应届入职率 < 1%)

### 2.2 长远稳定性(10–20 年视角)

**✅ AI Infra 的稳定性优势:**
- 大模型只要还在用 GPU/NPU 跑,AI Infra 就有饭吃。**这是大模型时代的"卖铲子"**。
- 技术栈深(CUDA/编译器/分布式),AI 工具替代风险极低(写不出 GEMM kernel 的 LLM 暂时取代不了写 CUDA 的人)。
- 头部需求长期存在(英伟达/华为/各 NPU 厂)。

**⚠ AI Infra 的稳定性隐忧:**
- **大厂训练集群"赢家通吃":** 真正烧钱训大模型的公司不到 50 家,需求集中。中小公司养不起 AI Infra 团队。
- **架构变动剧烈:** Transformer 5 年了,但 MoE/Mamba/diffusion 都在挑战。如果某天 Transformer 被颠覆,大量 Flash/Paged/KV cache 的优化经验会**部分作废**(底层 CUDA/编译器技能仍可迁移)。
- **35 岁焦虑:** **存在但弱于纯算法岗、强于 Linux 驱动**。算子调优本质是体力活+经验活,顶尖人不焦虑,中位数有压力。
- **国产 NPU 厂商的不确定性:** 寒武纪/壁仞/沐曦/燧原都在烧钱,2027–2028 行业洗牌可能很激烈,选错厂等于踩雷。**华为昇腾是当下最稳的国产选择。**

### 2.3 AI 替代风险

| 工作内容 | AI 替代风险 | 原因 |
|----------|------------|------|
| 写 CUDA kernel 调优 | **低** | 涉及硬件直觉、profiling 经验、tradeoff,LLM 短期写不出 SOTA 算子 |
| 写编译器 pass | **低** | 涉及大量数学+硬件知识,LLM 辅助但不替代 |
| 推理引擎架构设计 | **极低** | 系统级 tradeoff |
| Serving 业务逻辑 | **中** | 模板代码 LLM 能写 |
| 文档/调用样板 | **高** | 已经被 LLM 写得很好 |

**结论:** AI Infra 是 AI 时代 AI 工具最难替代的工程岗之一,**底层越深越安全**。这一点优于纯端侧 AI(部署工作的模板化部分容易被 LLM 替代)。

### 2.4 跟方案 C / 纯端侧 AI / 算法岗 严格对比

| 维度 | AI Infra | 方案 C(AI BSP/NPU) | 纯端侧 AI | 大模型算法岗 |
|------|----------|---------------------|-----------|--------------|
| 应届起薪 | 60–100w+(头部)/ 40–60w(中部) | 45–60w | 50–70w | 60–100w+(头部) |
| 10 年薪资 | 150–250w(头部 300w+) | 100–180w | 100–150w | 150–300w |
| EE 硕士契合度 | ★★ (需补 CUDA/编译/HPC) | ★★★★★(最契合) | ★★★(补 Python/DL) | ★(基本不契合) |
| 入场门槛 | 极高(CS+ACM+实习) | 中等(EE 背景+项目) | 中等 | 极高(顶会论文) |
| 35 岁焦虑 | 中(头部不焦虑) | **低** | 中 | 高(论文跟不上就死) |
| AI 替代风险 | 极低 | 极低 | 中(部署模板化) | 中(prompt engineer 化) |
| 长期稳定性 | 中-高(架构变动风险) | **高** | 中 | 中-低 |
| 头部公司机会 | 多但门槛极高 | 中等(国产 NPU 厂) | 中等 | 极少且门槛极高 |
| 跳槽议价权 | **强** | 强 | 中 | 中(论文光环褪色快) |
| **EE 硕士综合性价比** | ★★★(高风险高回报) | **★★★★★(最优)** | ★★★ | ★★(不推荐) |

> **关键判断:** 对你这个具体情况 —— EE 硕士、3 年制、研一末、当前主项目 WS63 TinyML + 在学 MP157 —— **方案 C 仍是最稳的主线**,AI Infra 应该作为**方案 C 工作 2–3 年后的上行通道**,不是应届直冲的目标。理由:
> 1. 应届直冲 AI Infra 你拼不过 CS 科班+ACM 选手
> 2. 方案 C 的算子/runtime 技能 5 年后能直接跳到 AI Infra(技术栈连通)
> 3. 华为昇腾本身就是横跨方案 C 和 AI Infra 的桥梁公司(端侧 NPU + 数据中心 CANN 是同一套)

---

## 3. 对 EE 硕士的具体建议(分三档执行)

### 档 A:**主线方案 C,把 AI Infra 当 5 年后的上行路径(推荐 ★★★★★)**

**适用:** 你不想赌、希望 2027 秋招稳拿 45–60w、长远再图。

**执行:**
- 2026.5 – 2027.9:严格按 [09 文档](09-完整时间线修正%203年制学硕版.md) 走方案 C
- 简历主项目:WS63 跌倒检测 + LuckFox NPU 驱动深耕 + MP157 异构多核
- 2027 秋招目标公司:**华为昇腾(终端+车 BU)/ 地平线 / 寒武纪 / 海光 / 海思**
- 入职后 2–3 年深耕算子/runtime,**自然就在 AI Infra 路上了**
- 5 年后跳:英伟达 DevTech / 字节 AML / DeepSeek Infra

### 档 B:**复合路 —— 方案 C 主线 + AI Infra 暑期实习(可选 ★★★)**

**适用:** 你想冲 AI Infra 但又怕踩空,愿意多花精力。

**执行:**
- 2026.5 – 2026.12:方案 C 路径 + 额外补 CUDA + 算子调优(每周加 6–8 小时)
- 2027 暑期实习:**优先冲 AI Infra 实习**(华为昇腾 / 字节 AML / 商汤 / 壁仞 / 沐曦 / 燧原)
- 2027 秋招:**双轨投递** —— AI BSP 岗 + AI Infra 岗,以 offer 结果定方向
- 风险:学习负载高(方案 C 已经满载,再叠 CUDA 是 1.3 倍工作量)

### 档 C:**应届直冲 AI Infra(不推荐 ★)**

**适用:** 你确认要赌头部公司、放弃方案 C 的稳定性。

**执行:**
- 2026.5 – 2026.9:全面切换学习重点 —— CUDA C++ / GEMM 调优 / PyTorch 内核
- 2026.7 – 2026.9:**必须**拿到一段 AI Infra 实习
- 2026.10 – 2027.6:开源贡献(vLLM/TensorRT-LLM/Megatron) + 顶会读论文
- 2027.7 – 2027.9:第二段 AI Infra 实习(冲华为昇腾 / 字节 AML 转正)
- 2027 秋招:双轨投递,以 SSP 为目标
- 风险:**这条路 EE 硕士的胜率 < 30%**,失败后回退到普通 AI 应用岗(40w 一档),收益比方案 C 低。

> ⚠ **强烈建议选档 A 或档 B。档 C 只对你确实热爱底层硬核计算且学习能力极强的情况成立。**

---

## 4. AI Infra 完整学习路线(共 6 阶段,适配 3 年制学硕)

> 以下路线**按档 B(复合路)规划**。如果你选档 A,只需把"AI Infra 增量部分(★)"忽略,其余与 [09 文档](09-完整时间线修正%203年制学硕版.md) 重叠。

### 阶段 0:基础前置(到 2026.06,已完成大部分)

| 必备 | 当前状态 | 备注 |
|------|---------|------|
| C 语言扎实 | ✅ | WS63 项目 |
| Python | ✅ 基础 | 训练/部署都要 |
| Linux 命令行 + Git | ✅ | |
| 一次完整的端侧部署 | ✅(进行中) | WS63 TinyML |
| PyTorch 训练一个小模型 | ⚠ 部分 | Edge Impulse 算用了一半 |

### 阶段 1:C++ + CUDA 入门(2026.06 – 2026.08,~10 周)★

**目标:** 能写出可编译运行的 GEMM/Conv2D CUDA kernel,能讲清楚 warp / shared memory / coalesced access。

**必学:**
- C++ 现代特性:RAII / 模板 / move / lambda / std::thread / 智能指针(Effective Modern C++ 任选 30 项)
- CUDA C++ 基础:thread/block/grid / shared memory / sync / atomic
- **关键题目:** 手写 SGEMM naive → tiled → vectorized,达到 cuBLAS 30–50%
- profiler:nsys / ncu(必须会读!)
- 你已经在 desktop 写过 CPU GEMM 优化(gemm-optim 项目,2026-05-29 给你建的备战项目),**直接把它移植到 CUDA 上**

**资源(精选,不贪多):**
- 《CUDA C++ Programming Guide》官方文档(读 chapter 1–6)
- Mark Harris 的 [Optimizing CUDA reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)(经典)
- B 站搜 "CUDA 编程入门" 找一个完整系列
- GitHub 项目 `siboehm/SGEMM_CUDA`(7 步从 naive 到 cuBLAS 90%)★ 强推

**硬件:** 你有 GTX1650 → 跑 CUDA 12.x 没问题(算力 7.5,无 TensorCore,但 GEMM/conv 都能跑)。**优先把这卡用起来,不要等买新卡。**

### 阶段 2:算子调优 + 推理引擎入门(2026.08 – 2026.12,~16 周)★

**目标:** 能独立优化一个算子(GEMM/Softmax/LayerNorm/Attention 任选),能讲清楚 vLLM / PagedAttention 原理。

**算子方向:**
- FlashAttention 1/2 原理(读论文 + 至少跑通官方实现)
- TensorRT-LLM / vLLM 的 KV cache 管理
- Triton 语言入门(GPU kernel 的 Python DSL,比 CUDA 上手快)
- 优化目标:**写一个 INT8 GEMM kernel 跑到 cuBLAS 60%+**

**推理引擎方向:**
- vLLM 源码精读:scheduler / kv cache / paged attention 三大模块
- TensorRT-LLM 文档 + 跑通一个 LLaMA 7B 推理
- 读 SGLang 论文(RadixAttention)

**与方案 C 的衔接:** 这一阶段 80% 的内容 **方案 C 09 文档也涵盖了**(只是侧重不同)。AI Infra 增量部分主要是 CUDA 深度 + 大模型推理特化。

### 阶段 3:暑期实习(2027.07 – 2027.09,~10 周)★★★

**目标:** 拿到一段 AI Infra 实习,转正/留下推荐信。

**目标公司分档:**

| 档次 | 公司 | 难度 | 备注 |
|------|------|------|------|
| 顶级 | 英伟达 DevTech(上海) | 极难 | 顶会论文 + CUDA 高手 |
| 顶级 | 字节 AML / Seed-Foundation | 难 | 算子/编译/推理实习 |
| **现实首选** | **华为昇腾(杭州/上海/深圳)** | **中-难** | 已面过,保持联系 |
| 现实首选 | 阿里 PAI / 蚂蚁 AI Infra | 中 | 国产替代+商业化 |
| 现实首选 | 寒武纪 / 海光 / 摩尔线程 / 壁仞 / 沐曦 | 中 | 国产 NPU 厂,机会多但波动大 |
| 现实首选 | 商汤 / 旷视 / 智源 | 中 | 偏 CV 推理 |
| 现实首选 | DeepSeek / 月之暗面 / MiniMax / 智谱 / 阶跃 | 中-难 | AI 公司 Infra 团队 |

**投递时间:** 2027.03 – 2027.05(春招/提前批)

**面试关键准备(基于昇腾面试反推):**
- 手撕 CUDA GEMM 是常考
- KV cache / PagedAttention / FlashAttention 必问
- 项目:WS63 端侧 AI 经验包装为"严苛资源约束下的推理优化"
- C++ 基础:虚函数表、智能指针、move 语义、模板

### 阶段 4:秋招冲刺(2027.09 – 2027.12)★★★

**双轨投递策略(档 B):**

| 投递档 | 岗位 | 公司 | 简历版本 |
|--------|------|------|---------|
| AI Infra 档 | 算子开发 / Infra 工程师 | 华为昇腾、字节 AML、商汤、AI Lab | 突出 CUDA/algorithm |
| 方案 C 档 | AI BSP / NPU 系统软件 | 华为海思车 BU、地平线、寒武纪 | 突出 BSP/驱动 |

**简历三个项目(理想配置):**
1. WS63 跌倒检测端侧 AI(基线项目,所有岗位都用)
2. LuckFox/RV1106 NPU 驱动 + 端侧推理(方案 C 主项目)
3. **CUDA GEMM 优化 / FlashAttention 复现** 或 **vLLM 二次开发**(AI Infra 增量项目)★

### 阶段 5:入职 + 头 3 年深耕(2028.07 – 2031.07)

**如果应届进 AI Infra:**
- P5 → P6:专精一个子方向(算子/编译/推理引擎)
- 3 年内目标:成为团队里"算子优化最强"或"编译器最熟"的那个

**如果应届进方案 C:**
- 2–3 年后从 AI BSP 端横跳到 AI Infra
- 跳板:**华为昇腾内部从车 BU → CANN 编译器组、地平线从 BSP → 编译器组**

---

## 5. 必修知识地图(技能矩阵)

### 5.1 AI Infra 核心技能矩阵

| 层 | 技能 | 必精通 | 教材/资源 |
|----|------|--------|----------|
| 硬件 | GPU 架构(SM/warp/shared mem/L1/L2) | ✅ | CUDA Programming Guide + GPU Gems |
| 硬件 | TensorCore / Tile / WMMA / TMA | ✅ | NVIDIA 架构白皮书(Ampere/Hopper) |
| 硬件 | 国产 NPU 架构(达芬奇/MLU/BPU) | A 级 | 昇腾 CANN 文档 / 寒武纪 BANG-C |
| 编程 | CUDA C++ | ✅ | 官方 + siboehm/SGEMM_CUDA |
| 编程 | Triton / AscendC | ✅ | OpenAI Triton tutorial + 华为 AscendC 文档 |
| 编程 | C++17/20 现代特性 | ✅ | Effective Modern C++ |
| 编程 | Python(PyTorch 自定义算子) | ✅ | PyTorch C++ Extension |
| 算法 | GEMM 优化(naive→tile→TC→split-K) | ✅ | siboehm 系列博客 |
| 算法 | FlashAttention 1/2/3 | ✅ | 论文 + 官方实现 |
| 算法 | KV cache / PagedAttention / RadixAttention | ✅ | vLLM / SGLang 论文 |
| 算法 | 量化(INT8/FP8/AWQ/GPTQ) | ✅ | TensorRT 量化 / GPTQ 论文 |
| 框架 | PyTorch 内部机制(autograd/dispatch) | A 级 | PyTorch dev podcast |
| 框架 | vLLM / SGLang / TensorRT-LLM 源码 | ✅(选一吃透) | 官方文档 + 源码 |
| 框架 | TVM / XLA / MLIR(选一) | A 级 | MLIR tutorial |
| 框架 | Megatron-LM / DeepSpeed(分布式) | A 级 | 官方文档 |
| 系统 | Linux 系统编程 | ✅(方案 C 共享) | APUE |
| 系统 | 通信原语(NCCL/HCCL/RDMA) | A 级 | NCCL 文档 |
| 系统 | profile 工具(nsys/ncu/Triton autotune) | ✅ | NVIDIA 官方教程 |

### 5.2 学习优先级(EE 硕士版,前 10 项必学)

1. **CUDA C++ 基础**(必学,3 周)
2. **GEMM 7 步优化**(必学,4 周,siboehm 项目)
3. **C++ 现代特性**(必学,持续)
4. **profiler:nsys / ncu 用法**(必学,1 周)
5. **FlashAttention 论文 + 实现阅读**(必学,2 周)
6. **vLLM 源码精读**(必学,3 周)
7. **量化基础(INT8/FP16)**(必学,1 周)
8. **PyTorch 自定义 C++/CUDA 算子**(必学,2 周)
9. **AscendC 入门(若冲昇腾)**(必学,1 周)
10. **GPU 架构图细节(Ampere/Hopper)**(必学,1 周)

### 5.3 与方案 C 的技能重合度

**完全重合(80% 学方案 C 等于学 AI Infra 基础):**
- C++ 现代特性
- 量化基础
- ONNX / PyTorch
- profile 思维
- 算子优化方法论(端侧和数据中心都通)
- AscendC(昇腾端侧+数据中心同套)

**AI Infra 独有(需要额外补):**
- CUDA 深度
- FlashAttention / PagedAttention 等大模型特化算法
- 编译器 pass(MLIR)
- 分布式训练原语

**结论:** 方案 C 是 AI Infra 的良好预备。**先深耕方案 C,2-3 年后补 CUDA 深度即可横跳。**

---

## 6. 目标公司清单(2027 秋招 AI Infra 方向)

### 6.1 顶级(应届进入难度 ★★★★★)
- 英伟达 DevTech / 中国研发(算子组)—— 上海
- OpenAI / Anthropic Infra —— 海外,英文+顶会论文
- DeepSeek Infra —— 北京/杭州,极难
- 字节 Seed-Foundation / AML —— 北京/上海/杭州

### 6.2 第一梯队(应届进入难度 ★★★★)
- **华为昇腾 CANN / MindSpore 团队 —— 杭州滨江 / 上海 / 深圳 ★ 你刚面过的线**
- 阿里 PAI / 通义 Infra —— 杭州
- 蚂蚁 AGI 基础设施 —— 杭州
- 百度文心 Infra —— 北京
- 腾讯 TEG / 优图 Infra —— 深圳/北京
- 阶跃 / 月之暗面 / MiniMax / 智谱 Infra —— 北京

### 6.3 第二梯队(应届进入难度 ★★★)
- 寒武纪 / 海光 / 摩尔线程 / 壁仞 / 沐曦 / 燧原 —— 北京/上海(国产 NPU 厂,波动大)
- 商汤 / 旷视 / 智源 / 上海 AI Lab —— 上海/北京
- 美团 NLP Infra / 滴滴 AI Infra —— 北京

### 6.4 与方案 C 重合的公司(双向投递)
华为昇腾、地平线、寒武纪、海光、海思车 BU —— **这五家可以一次投递,简历互相通用**。

---

## 7. 风险与对冲

### 7.1 主要风险
1. **应届直冲胜率低:** EE 硕士打 CS 科班+ACM,正面竞争吃亏。
2. **国产 NPU 厂洗牌:** 2027–2028 一定有公司活不下来,选错踩雷。
3. **大模型架构突变:** Transformer 时代红利可能在 2028+ 减退,Attention 优化经验贬值。
4. **AI 工具长期蚕食:** 中低端算子调优 5 年后 LLM 自动化的概率不低。
5. **学习负载过重:** 方案 C + AI Infra 双修 = 1.3 倍工作量,容易两边都做不深。

### 7.2 对冲策略
1. **主线锁方案 C,AI Infra 当上行通道** —— 不裸冲,降低应届失败成本。
2. **集中投华为昇腾/字节/头部国产 NPU,不投小厂** —— 降低踩雷概率。
3. **专精底层(CUDA 架构、profiler、汇编级)** —— 这部分 LLM 替代最难,反而抗周期。
4. **保留 Linux 驱动底子做最终保底** —— 方案 C 自带这层。
5. **2026 暑期不上 CUDA,优先完成 LuckFox 项目** —— 先保方案 C 简历,再叠 AI Infra。

---

## 8. 你的下一步(可立即执行的清单)

**到 2026-06-30 之前(1 个月):**
- [ ] 跑通 desktop 上 `c:\Users\yyfu\Desktop\gemm-optim\` 的 C++ CPU GEMM(装 w64devkit,见 [华为面试备战手册.docx](file:///C:/Users/yyfu/Desktop/华为昇腾面试备战手册.docx))
- [ ] **可选** CUDA 入门(2 周):装 CUDA 12.x + 在 GTX1650 跑通官方 sample
- [ ] 等华为昇腾面试结果,**不论结果保持邮件礼貌跟进**

**到 2026-08-31 之前(暑期):**
- [ ] **优先完成方案 C 09 文档的暑期任务**(LuckFox 项目 / WS63 端侧 AI 收尾)
- [ ] 如果方案 C 任务有富余时间:跑通 siboehm/SGEMM_CUDA 7 步优化
- [ ] **不要**在 LuckFox 项目还没立项前就转 CUDA(主次颠倒)

**2026-09 ~ 2027-03:**
- [ ] 方案 C 路径全速推进,完成 LuckFox NPU 驱动深耕
- [ ] AI Infra 额外学习:CUDA GEMM 完成 + vLLM 跑通 + FlashAttention 读完
- [ ] 2027 春招提前批:**双轨投递华为昇腾 + 地平线**

**2027-04 ~ 2027-09:**
- [ ] 拿到 AI Infra 暑期实习(目标:华为昇腾)
- [ ] 实习期内确认最终方向:**Infra 还是 BSP**

**2027 秋招:**
- [ ] 按实习结果决定主投档(AI Infra 或方案 C)

---

## 9. 写在最后:对 AI Infra 的诚实判断

**(本节最重要,请认真读)**

我必须诚实告诉你:

1. **AI Infra 是 2026 最热的方向,但热度 ≠ 适合你**。EE 硕士裸冲 AI Infra,90% 的概率会被 CS 科班+ACM+顶会论文的候选人按在地上。我之前如果直接对你说"AI Infra 真香,冲就完事了",那是在拿你的职业生涯赌一个不属于你的赛道。

2. **AI Infra 跟方案 C 不是对立的**。AI Infra 的"算子+编译器+runtime"是数据中心版本,方案 C 的"NPU 系统软件"是端侧版本,**底层算子优化方法论是同一套**。先做端侧深度,工作 2-3 年后横跳到数据中心,胜率比应届直冲高 3 倍。

3. **华为昇腾这次面试是最重要的信号**:它**同时**是 AI Infra 路径和方案 C 路径上**最理想的中继公司**。不论这次实习是否拿到,**昇腾这条线值得作为你 2026–2028 的核心目标**。备战手册里准备的算子调优、CANN、达芬奇架构,在这两条路上都不浪费。

4. **如果让我给一个押注:你**应该把 70% 精力放在方案 C(LuckFox + MP157 + WS63),20% 精力补 CUDA(为 AI Infra 留门),10% 精力补 C++/算法/简历**。**不要把比例反过来,会两边都做不深**。

5. **30 岁前的择业选择,稳定性优先于天花板**。年轻时你以为天花板重要,30 岁后你会发现稳定的工作经验+深度才是议价权。方案 C 在这点上完胜应届裸冲 AI Infra。

---

## 10. 参考(可后续追加)

- [siboehm 的 SGEMM_CUDA 七步博客](https://siboehm.com/articles/22/CUDA-MMM) ★ 必读
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- vLLM 论文:Efficient Memory Management for Large Language Model Serving with PagedAttention(SOSP'23)
- FlashAttention 2 论文(2023)
- 华为 CANN 官方文档 / 昇腾 AscendC 编程指南
- 《CUDA C++ Programming Guide》(NVIDIA 官方)
- 《Effective Modern C++》Scott Meyers
- 极客时间:《CUDA 编程》系列(中文较好的入门)

---

> 文档归属:`fyy0619-cell/qian_niu/10-AI Infra 岗位分析与学习路线.md`
> 起草:2026-05-31 · 关联:[08-方案C](08-方案C%20完整规划%20AI%20BSP路径.md) · [09-时间线](09-完整时间线修正%203年制学硕版.md) · 触发:华为昇腾面试(2026-05-30)
