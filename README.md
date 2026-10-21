# Paper to MD for fine-tuning data

将论文 PDF 转换为高质量多语言 SFT（监督微调）问答数据集的自动化流水线。

## 架构

```
本地 PDF  ──>  MinerU VLM 解析  ──>  Markdown 落盘  ──>  Heading-aware 切分  ──>  LLM 生成 QA  ──>  JSONL
              (mineru.net/api/v4)     output/mineru_md/    (黑名单+长度过滤)      (MiniMax)        output/sft/
```

| 阶段 | 实现位置 | 说明 |
|---|---|---|
| 1. 解析 | `run_mineru()` | 批量申请预签名上传链接 → PUT 上传 → 轮询 `extract-results/batch` → 下载 md |
| 2. 切分 | `chunk_markdown()` | 按 2-4 级标题切分，向上回溯最近一级标题作上下文，黑名单过滤 |
| 3. 生成 | `generate_qa()` | 中英双语 prompt，按 CJK 比例自动选模板，输出 Alpaca 格式 |
| 4. 断点续传 | `output/state.json` | 记录已完成 chunk key，失败自动重试，支持增量写盘 |

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# 2. 配置密钥
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
# 编辑 .env 填入 MINERU_API_KEY 与 LLM_API_KEY

# 3. 运行
python build_sft_pipeline.py             # 全流程 (解析 + 切分 + 生成)
python build_sft_pipeline.py --from-md   # 跳过 MinerU, 复用已有 md
```

## 输出

| 路径 | 内容 |
|---|---|
| `output/mineru_md/*.md` | MinerU 解析出的 Markdown |
| `output/sft/rice_endosperm_sft.jsonl` | 最终 SFT 数据集，Alpaca 格式 |
| `output/state.json` | 断点续传状态（**已 gitignore**，含预签名 URL） |

单行 JSONL 结构：

```json
{"instruction": "问题", "input": "", "output": "答案", "source": "文档 > 一级标题 > 章节"}
```

## 配置项（`.env`）

| 变量 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `MINERU_API_KEY` | 是 | — | MinerU 平台密钥 |
| `LLM_API_KEY` | 是 | — | LLM API 密钥 |
| `LLM_BASE_URL` | 否 | `https://api.minimaxi.com/anthropic` | Anthropic 兼容端点 |
| `LLM_MODEL` | 否 | `MiniMax-M3.1-Flash` | 生成用模型 |

## 已知限制

以下问题在当前实现中已确认存在，使用数据集前请务必了解：

1. **约 47% 的解析文本未进入数据集。** `chunk_markdown()` 只遍历标题节点，首个二级标题之前的文本（标题、作者、**摘要、关键词**）永不进入任何 chunk。
2. **同名章节会被永久跳过。** 断点续传 key 为 `doc::h1::section`，而 MinerU 输出的标题层级是扁平的，同一文档内存在同名同级标题时，第二个 chunk 命中 `done` 被跳过。
3. **表格与图片语义不完整。** 表格为 HTML 形式保留；图片引用被剥离但图注文字保留，出现"如图 2 所示"却无图的情况。
4. **存在幻觉风险。** LLM 仅基于单一 chunk 生成，未做忠实度校验；抽样统计中约 155/252 条答案含源文本中不存在的实词。
5. **采样非确定。** `temperature=0.3`，相同输入重跑会得到不同数据集。

修复路线见 [ROADMAP.md](ROADMAP.md)。

## License

见 [LICENSE](LICENSE)。
