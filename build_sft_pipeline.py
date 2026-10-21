# -*- coding: utf-8 -*-
"""
SFT 数据集构建流水线（全流程可断点续传）

本地 PDF ──> MinerU VLM 解析(/api/v4/file/upload, 无需公网链接)
        ──> Markdown 落盘 output/mineru_md/
        ──> Heading-aware 切分(黑名单过滤参考文献等)
        ──> MiniMax LLM 生成 Alpaca 格式 QA
        ──> output/sft/rice_endosperm_sft.jsonl

用法:
    python build_sft_pipeline.py            # 全流程
    python build_sft_pipeline.py --from-md  # 跳过 MinerU, 直接从已有 md 生成
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

# Windows 控制台中文/emoji 安全输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ==================== 路径与常量 ====================
ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "Review on Rice Endosperm"
OUT_DIR = ROOT / "output"
MD_DIR = OUT_DIR / "mineru_md"
SFT_DIR = OUT_DIR / "sft"
STATE_FILE = OUT_DIR / "state.json"
SFT_JSONL = SFT_DIR / "rice_endosperm_sft.jsonl"

MINERU_BASE = "https://mineru.net/api/v4"
POLL_INTERVAL = 15          # 秒
POLL_TIMEOUT = 60 * 60      # 单文件最长等待 1 小时
LLM_TIMEOUT = 300           # 秒
LLM_MAX_TOKENS = 16384
QA_PER_CHUNK = 3
MIN_CHUNK_CHARS = 300       # 过短章节直接丢弃

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
# 参考文献/致谢/基金等不参与 QA 生成
BLACKLIST_RE = re.compile(
    r"(参考文献|引用文献|致谢|謝辞|acknowledg|references?|corresponding author|"
    r"author contribution|利益冲突|conflict of interest|funding|基金|资助|"
    r"supplementary|appendix|附录|图表|figure legend|摘要|abstract|keywords?|关键词)",
    re.IGNORECASE,
)


# ==================== 配置加载 ====================
def load_env() -> dict:
    env = {}
    env_path = ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


ENV = load_env()
MINERU_TOKEN = ENV.get("MINERU_API_KEY", "")
LLM_KEY = ENV.get("LLM_API_KEY", "")
LLM_BASE = ENV.get("LLM_BASE_URL", "https://api.minimaxi.com/anthropic").rstrip("/")
LLM_MODEL = ENV.get("LLM_MODEL", "MiniMax-M3.1-Flash")

if not MINERU_TOKEN:
    sys.exit("[FATAL] .env 中缺少 MINERU_API_KEY")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"mineru": {}, "chunks_done": [], "chunks_failed": []}


def save_state(st: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def has_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s))


def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(CJK_RE.findall(s)) / max(len(s), 1)


# ==================== 1. MinerU 解析模块 ====================
def mineru_headers() -> dict:
    return {"Authorization": f"Bearer {MINERU_TOKEN}"}


def upload_batch(pdf_list: list) -> tuple:
    """为一批本地 PDF 申请预签名上传链接。
    返回 (batch_id, [presigned_url, ...])。文件名转 ASCII 避免服务端编码问题,
    原始文件名放在 data_id 中回传。"""
    files_meta = [{"name": f"paper_{i:02d}.pdf", "data_id": p.stem}
                  for i, p in enumerate(pdf_list, 1)]
    resp = requests.post(
        f"{MINERU_BASE}/file-urls/batch",
        headers={**mineru_headers(), "Content-Type": "application/json"},
        json={"files": files_meta, "model_version": "vlm"},
        timeout=120,
    )
    resp.raise_for_status()
    j = resp.json()
    if j.get("code") != 0:
        raise RuntimeError(f"MinerU 申请上传链接失败: {json.dumps(j, ensure_ascii=False)}")
    data = j.get("data") or {}
    batch_id, urls = data.get("batch_id"), data.get("file_urls") or []
    if not batch_id or len(urls) != len(pdf_list):
        raise RuntimeError(f"MinerU 返回异常: {json.dumps(j, ensure_ascii=False)}")
    return batch_id, urls


def query_batch(batch_id: str) -> list:
    resp = requests.get(
        f"{MINERU_BASE}/extract-results/batch/{batch_id}",
        headers=mineru_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    j = resp.json()
    if j.get("code") != 0:
        raise RuntimeError(f"MinerU 查询失败: {json.dumps(j, ensure_ascii=False)}")
    results = (j.get("data") or {}).get("extract_result") or []
    if not results:
        raise RuntimeError(f"MinerU 查询结果为空: {json.dumps(j, ensure_ascii=False)}")
    return results


def put_file(pdf_path: Path, presigned_url: str, label: str) -> None:
    """纯 PUT 上传二进制。严禁带 Content-Type 等额外 header(OSS 签名校验会 403)。"""
    with pdf_path.open("rb") as fh:
        r = requests.put(presigned_url, data=fh, timeout=600)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"[{label}] PUT 上传失败 HTTP {r.status_code}: {r.text[:200]}")


def wait_batch(batch_id: str, labels: dict) -> dict:
    """轮询 batch 直到所有文件 done/failed。返回 {file_name: result_item}。"""
    wanted = set(labels)
    start = time.time()
    finished: dict = {}
    failed: dict = {}
    while time.time() - start < POLL_TIMEOUT:
        try:
            results = query_batch(batch_id)
        except Exception as e:
            print(f"  [warn] 查询异常(将继续重试): {e}")
            time.sleep(POLL_INTERVAL)
            continue
        for item in results:
            name = item.get("file_name", "")
            if name not in wanted or name in finished or name in failed:
                continue
            state = item.get("state", "")
            if state == "done":
                finished[name] = item
                print(f"  [done] {labels[name]}")
            elif state == "failed":
                failed[name] = item
                print(f"  [FAILED] {labels[name]}: {item.get('err_msg', '')}")
        if len(finished) + len(failed) >= len(wanted):
            if failed:
                raise RuntimeError(f"MinerU 有文件解析失败: {list(failed)}")
            return finished
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"MinerU 轮询超时(1h), batch={batch_id}")


def _get_with_retry(url: str, timeout: int, retries: int = 3) -> bytes:
    """requests 优先; 遇 SSL 握手问题(该 CDN 会掐 Python 的 TLS 握手)换 curl 兜底。
    curl 为 Schannel 构建时追加 --ssl-no-revoke(吊销服务器不可达会导致失败)。"""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.exceptions.SSLError:
            break  # SSL 层被断开, requests 重试无意义, 换 curl
        except Exception as e:
            print(f"  [warn] 下载重试 {attempt}/{retries}: {e}")
            time.sleep(3 * attempt)

    import shutil
    import subprocess
    import tempfile
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError(f"下载失败且系统无 curl: {url}")
    ver = subprocess.run([curl, "--version"], capture_output=True, text=True).stdout
    extra = ["--ssl-no-revoke"] if "Schannel" in ver else []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tf:
        tmp = Path(tf.name)
    try:
        subprocess.run(
            [curl, "-sS", "-L", *extra, "--max-time", str(timeout),
             "-o", str(tmp), url],
            check=True, capture_output=True, text=True,
        )
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def download_md(item: dict, out_md: Path) -> None:
    """优先 md_link, 其次 raw_md_link, 最后解压 full_zip_url。"""
    link = item.get("md_link") or item.get("raw_md_link")
    if link:
        out_md.write_bytes(_get_with_retry(link, 300))
        return
    zip_url = item.get("full_zip_url")
    if not zip_url:
        raise RuntimeError(f"结果中无任何下载链接: {item}")
    import io
    import zipfile
    buf = io.BytesIO(_get_with_retry(zip_url, 600))
    with zipfile.ZipFile(buf) as zf:
        md_names = [n for n in zf.namelist() if n.lower().endswith(".md")]
        if not md_names:
            raise RuntimeError(f"zip 中没有 .md 文件: {zf.namelist()}")
        # 选最大的 md(主文档)
        md_names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        out_md.write_bytes(zf.read(md_names[0]))


def run_mineru(st: dict) -> dict:
    """返回 {pdf_stem: md_path}。已有本地 md 的直接跳过。"""
    MD_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"[FATAL] {PDF_DIR} 下没有 PDF")
    print(f"共发现 {len(pdfs)} 个 PDF")

    def md_of(p: Path) -> Path:
        return MD_DIR / f"{p.stem}.md"

    for p in pdfs:
        if md_of(p).exists() and md_of(p).stat().st_size > 500:
            print(f"已有本地 md, 跳过: {p.name}")
    pending = [p for p in pdfs if not (md_of(p).exists() and md_of(p).stat().st_size > 500)]

    if pending:
        batch = st["mineru"].get("batch")
        if not batch:
            batch_id, urls = upload_batch(pending)
            batch = {
                "batch_id": batch_id,
                "files": [{"name": f"paper_{i:02d}.pdf", "stem": p.stem,
                           "url": urls[i - 1]}
                          for i, p in enumerate(pending, 1)],
            }
            st["mineru"] = {"batch": batch}
            save_state(st)
            print(f"已创建解析批次 {batch_id} ({len(pending)} 个文件)")
        else:
            print(f"复用已有批次 {batch['batch_id']} (预签名链接 24h 内有效)")

        labels = {f["name"]: f["stem"] for f in batch["files"]}
        pending_stems = {p.stem for p in pending}
        # PUT 上传(幂等, 断点续传时重复 PUT 无副作用)
        for f in batch["files"]:
            if f["stem"] not in pending_stems:
                continue
            src = next(p for p in pending if p.stem == f["stem"])
            put_file(src, f["url"], f["stem"])
            print(f"  已上传: {f['stem']}")

        results = wait_batch(batch["batch_id"], labels)
        errors = []
        for f in batch["files"]:
            if f["stem"] not in pending_stems:
                continue
            item = results[f["name"]]
            out_md = MD_DIR / f"{f['stem']}.md"
            try:
                download_md(item, out_md)
                print(f"  解析完成 -> {out_md.relative_to(ROOT)} "
                      f"({out_md.stat().st_size/1024:.0f} KB)")
            except Exception as e:
                errors.append(f["stem"])
                print(f"  [error] 结果下载失败 {f['stem']}: {e}")
        if errors:
            saved = [p for p in pending if (MD_DIR / f"{p.stem}.md").exists()]
            if not saved:
                raise RuntimeError(f"全部结果下载失败: {errors}")
            print(f"[warn] {len(errors)} 个文件下载失败, 已跳过: {errors}")

    return {p.stem: MD_DIR / f"{p.stem}.md" for p in pdfs
            if (MD_DIR / f"{p.stem}.md").exists()}


# ==================== 2. Markdown 切分模块 ====================
def clean_content(text: str) -> str:
    text = IMG_RE.sub("", text)               # 去图片引用(本地无对应文件)
    lines = [ln.rstrip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if ln.strip():
            out.append(ln)
            blank = 0
        elif blank < 1:
            out.append("")
            blank += 1
    return "\n".join(out).strip()


def chunk_markdown(md_text: str, doc_slug: str) -> list:
    """按 2-4 级标题切分; 记录最近一级标题作上下文; 黑名单+长度过滤。"""
    matches = list(HEADING_RE.finditer(md_text))
    chunks = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        if level < 2:        # 一级标题是文档题目, 不作为切分点
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = clean_content(md_text[start:end])
        if len(content) < MIN_CHUNK_CHARS:
            continue
        if BLACKLIST_RE.search(title):
            continue
        # 向上找最近的一级标题
        h1 = ""
        for j in range(i, -1, -1):
            if len(matches[j].group(1)) == 1:
                h1 = matches[j].group(2).strip()
                break
        chunks.append({
            "doc": doc_slug,
            "h1": h1,
            "section": title,
            "content": content,
            "lang": "zh" if cjk_ratio(title + content) > 0.05 else "en",
        })
    return chunks


# ==================== 3. LLM 生成模块 (MiniMax, Anthropic 协议) ====================
def call_llm(prompt: str, retries: int = 3) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": LLM_MODEL,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(f"{LLM_BASE}/v1/messages", headers=headers,
                              json=body, timeout=LLM_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(5 * attempt)
                continue
            r.raise_for_status()
            j = r.json()
            texts = [b.get("text", "") for b in j.get("content", [])
                     if b.get("type") == "text"]
            text = "".join(texts).strip()
            if text:
                return text
            last_err = f"空回复: {json.dumps(j, ensure_ascii=False)[:300]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(5 * attempt)
    raise RuntimeError(f"LLM 调用失败: {last_err}")


def parse_json_array(text: str) -> list:
    """从模型输出中稳健提取 JSON 数组。"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    idx = text.find("[")
    if idx == -1:
        raise ValueError("输出中没有 JSON 数组")
    arr, _ = json.JSONDecoder().raw_decode(text[idx:])
    if not isinstance(arr, list):
        raise ValueError("解析结果不是数组")
    return arr


PROMPT_ZH = """你是一位资深的水稻分子生物学与作物科学专家, 正在为科研领域大模型构建高质量微调(SFT)数据集。

【文献信息】
文档: {h1}
章节: {section}

【章节正文】
{content}

【任务要求】
1. 基于且仅基于上述正文, 生成 {n} 个高质量中文问答对。
2. 问题要有科研深度: 涉及分子机制、调控通路、时空表达、因果关系、实验方法或结论对比; 避免字面检索式的浅问题。
3. 答案必须严格忠实于正文, 禁止编造或引入外部知识; 答案完整、专业、信息密度高。
4. 只输出一个 JSON 数组, 不要任何解释文字或代码块标记, 格式:
[{{"instruction": "问题", "output": "答案"}}]"""

PROMPT_EN = """You are a senior expert in plant molecular biology and crop science, building a high-quality SFT (fine-tuning) dataset for a scientific domain LLM.

[Paper context]
Document: {h1}
Section: {section}

[Section text]
{content}

[Requirements]
1. Based strictly on the text above, generate {n} high-quality English QA pairs.
2. Questions must have scientific depth: molecular mechanisms, regulatory pathways, spatio-temporal expression, causal relationships, experimental methods, or comparison of conclusions. Avoid shallow literal-retrieval questions.
3. Answers must be strictly faithful to the text; no fabrication or external knowledge. Answers should be complete, professional and information-dense.
4. Output ONLY a JSON array, no explanations or code fences, in the format:
[{{"instruction": "question", "output": "answer"}}]"""


def generate_qa(chunk: dict) -> list:
    tpl = PROMPT_ZH if chunk["lang"] == "zh" else PROMPT_EN
    prompt = tpl.format(h1=chunk["h1"] or chunk["doc"], section=chunk["section"],
                        content=chunk["content"], n=QA_PER_CHUNK)
    text = call_llm(prompt)
    arr = parse_json_array(text)
    qas = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        instr, outp = item.get("instruction"), item.get("output")
        if not instr or not outp:
            continue
        qas.append({
            "instruction": str(instr).strip(),
            "input": "",
            "output": str(outp).strip(),
            "source": f"{chunk['doc']} > {chunk['h1']} > {chunk['section']}" if chunk["h1"]
                      else f"{chunk['doc']} > {chunk['section']}",
        })
    return qas


# ==================== 4. 主流程 ====================
def main():
    from_md = "--from-md" in sys.argv
    st = load_state()
    OUT_DIR.mkdir(exist_ok=True)
    SFT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: MinerU 解析 ----
    if from_md:
        md_map = {p.stem: p for p in MD_DIR.glob("*.md")}
        if not md_map:
            sys.exit("[FATAL] --from-md 但 output/mineru_md 下没有 md, 请先跑全流程")
        print(f"--from-md 模式: 复用 {len(md_map)} 个本地 md")
    else:
        md_map = run_mineru(st)

    # ---- Step 2: 切分 ----
    all_chunks = []
    for slug in sorted(md_map):
        md_text = md_map[slug].read_text(encoding="utf-8", errors="replace")
        cs = chunk_markdown(md_text, slug)
        print(f"切分 {slug}: {len(cs)} 个有效章节")
        all_chunks.extend(cs)
    print(f"合计 {len(all_chunks)} 个待生成章节\n")

    # ---- Step 3: LLM 生成 + 增量写盘 ----
    done = set(st.get("chunks_done", []))
    failed = set(st.get("chunks_failed", []))
    total_new = 0
    t0 = time.time()
    for i, chunk in enumerate(tqdm(all_chunks, desc="生成SFT", ncols=80), 1):
        key = f"{chunk['doc']}::{chunk['h1']}::{chunk['section']}"
        if key in done:
            continue
        try:
            qas = generate_qa(chunk)
            if not qas:
                raise ValueError("解析出 0 条 QA")
            with SFT_JSONL.open("a", encoding="utf-8") as f:
                for qa in qas:
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            total_new += len(qas)
            done.add(key)
            failed.discard(key)
        except Exception as e:
            failed.add(key)
            print(f"\n[warn] 失败 {key}: {e}")
        if i % 5 == 0:
            st["chunks_done"] = sorted(done)
            st["chunks_failed"] = sorted(failed)
            save_state(st)
    st["chunks_done"] = sorted(done)
    st["chunks_failed"] = sorted(failed)
    save_state(st)

    # ---- 汇总 ----
    n_lines = sum(1 for _ in SFT_JSONL.open(encoding="utf-8")) if SFT_JSONL.exists() else 0
    mins = (time.time() - t0) / 60
    print(f"\n完成: 本次新增 {total_new} 条, 累计 {n_lines} 条 QA")
    print(f"耗时 {mins:.1f} 分钟; 失败章节 {len(failed)} 个(见 state.json chunks_failed)")
    print(f"数据集: {SFT_JSONL}")


if __name__ == "__main__":
    main()
