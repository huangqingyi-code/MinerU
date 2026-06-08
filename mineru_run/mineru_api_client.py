"""
mineru_api_client.py
====================
mineru-api 的 Python 客户端。把这条 curl 命令翻译成 Python:

    curl -X POST http://127.0.0.1:8000/file_parse \\
      -F "files=@xxx.pdf" \\
      -F "backend=vlm-auto-engine" \\
      -F "formula_enable=true" \\
      -F "table_enable=true" \\
      -F "image_analysis=true" \\
      -F "return_md=true" \\
      -F "return_middle_json=true" \\
      -F "return_content_list=true" \\
      -F "return_images=true" \\
      -F "return_original_file=true" \\
      -F "response_format_zip=true" \\
      --output result.zip

支持两种调用风格:
1) /file_parse —— 同步,阻塞等结果(适合脚本、小批量)
2) /tasks       —— 异步,提交任务后轮询(适合长任务、批量、不想占住连接)

依赖:
    pip install requests
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

import requests


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# 默认请求参数。和你 curl 里的参数一一对应。
# 注意:HTTP form 字段值需要是字符串,布尔写成 "true"/"false"。
DEFAULT_OPTIONS: dict = {
    "backend": "vlm-auto-engine",
    "formula_enable": "true",
    "table_enable": "true",
    "image_analysis": "false",
    "return_md": "true",
    "return_middle_json": "true",
    "return_content_list": "true",
    "return_images": "true",
    "return_original_file": "true",
    "response_format_zip": "true",
    # 其他可选:
    # "parse_method": "auto",            # 仅 pipeline/hybrid 生效
    # "lang_list": "ch",                 # 仅 pipeline/hybrid 生效
    # "start_page_id": "0",
    # "end_page_id": "9",
    # "server_url": "http://127.0.0.1:30000",  # 仅 *-http-client 生效
}


class MinerUClient:
    """轻量封装 mineru-api 的两个核心端点"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # ---------- 健康检查 ----------
    def health(self) -> dict:
        r = self.session.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        # /health 可能返回空体也可能返回 JSON,做兼容
        return r.json() if r.content else {"status": "ok"}

    # ---------- 同步:/file_parse ----------
    def parse_sync(
        self,
        file_paths: list[str | Path],
        out_zip: str | Path,
        options: dict | None = None,
    ) -> Path:
        """
        同步解析多个文件,把返回的 ZIP 保存到本地。
        阻塞直到服务端解析完(可能要几分钟,看文件大小和 backend)。
        """
        opts = {**DEFAULT_OPTIONS, **(options or {})}

        # files 参数:多个 file 字段都叫 "files",值是 (filename, fileobj) 元组
        files_payload = []
        opened = []
        try:
            for fp in file_paths:
                fp = Path(fp)
                fh = open(fp, "rb")
                opened.append(fh)
                files_payload.append(("files", (fp.name, fh, "application/octet-stream")))

            url = f"{self.base_url}/file_parse"
            print(f"[POST] {url}  files={[Path(p).name for p in file_paths]}  opts={opts}")
            t0 = time.time()
            r = self.session.post(url, files=files_payload, data=opts, timeout=self.timeout)
            dt = time.time() - t0
            print(f"[done] {r.status_code} in {dt:.1f}s, body={len(r.content)} bytes")
            r.raise_for_status()
        finally:
            for fh in opened:
                fh.close()

        # response_format_zip=true 时是二进制 zip;否则是 JSON
        if opts.get("response_format_zip") == "true":
            out_zip = Path(out_zip)
            out_zip.write_bytes(r.content)
            print(f"[ok] saved zip -> {out_zip}")
            return out_zip
        else:
            # JSON 模式,把 JSON 落盘
            out_json = Path(str(out_zip)).with_suffix(".json")
            out_json.write_bytes(r.content)
            print(f"[ok] saved json -> {out_json}")
            return out_json

    # ---------- 异步:/tasks ----------
    def submit_task(
        self,
        file_paths: list[str | Path],
        options: dict | None = None,
    ) -> str:
        """提交一个异步解析任务,返回 task_id"""
        opts = {**DEFAULT_OPTIONS, **(options or {})}

        files_payload = []
        opened = []
        try:
            for fp in file_paths:
                fp = Path(fp)
                fh = open(fp, "rb")
                opened.append(fh)
                files_payload.append(("files", (fp.name, fh, "application/octet-stream")))

            url = f"{self.base_url}/tasks"
            print(f"[POST] {url}  files={[Path(p).name for p in file_paths]}")
            r = self.session.post(url, files=files_payload, data=opts, timeout=60)
            r.raise_for_status()
        finally:
            for fh in opened:
                fh.close()

        data = r.json()
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise RuntimeError(f"no task_id in response: {data!r}")
        print(f"[submitted] task_id={task_id}")
        return task_id

    def get_task(self, task_id: str) -> dict:
        r = self.session.get(f"{self.base_url}/tasks/{task_id}", timeout=10)
        r.raise_for_status()
        return r.json()

    def get_result(self, task_id: str, out_zip: str | Path | None = None) -> Path | dict:
        """拿任务结果。response_format_zip=true 时是 zip(走二进制流);否则是 JSON。"""
        r = self.session.get(f"{self.base_url}/tasks/{task_id}/result",
                             timeout=self.timeout, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "application/zip" in ctype or "octet-stream" in ctype:
            if out_zip is None:
                out_zip = f"{task_id}.zip"
            out_zip = Path(out_zip)
            with open(out_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    f.write(chunk)
            print(f"[ok] saved zip -> {out_zip}")
            return out_zip
        else:
            return r.json()

    def wait_task(
        self,
        task_id: str,
        out_zip: str | Path | None = None,
        interval: float = 3.0,
        timeout: float = 1800,
    ) -> Path | dict:
        """轮询任务状态直到完成,然后下载结果"""
        t0 = time.time()
        last_status = None
        while time.time() - t0 < timeout:
            info = self.get_task(task_id)
            status = info.get("status")
            if status != last_status:
                print(f"  [{int(time.time() - t0)}s] status={status}")
                last_status = status
            if status in ("completed", "success", "done"):
                return self.get_result(task_id, out_zip=out_zip)
            if status in ("failed", "error"):
                raise RuntimeError(f"task {task_id} failed: {info.get('error') or info}")
            time.sleep(interval)
        raise TimeoutError(f"task {task_id} not done after {timeout}s")


# ---------------------------------------------------------------------------
# zip 自动解包,方便对接可视化
# ---------------------------------------------------------------------------
def unzip_result(zip_path: str | Path, out_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"[ok] unzipped -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# 演示:跟 demo.py 一样的 parse_doc(doc_path_list, output_dir, backend=...) 形式
# ---------------------------------------------------------------------------
def parse_doc(
    doc_path_list: list[str | Path],
    output_dir: str | Path,
    backend: str = "vlm-auto-engine",
    server_url: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 600,
    use_async: bool = False,
    extra_options: dict | None = None,
):
    """
    类似 official demo.py 的 parse_doc 接口,但走 HTTP API。
    每个文档解析一次,结果 zip 自动解包到 output_dir/{文件名}/
    """
    client = MinerUClient(base_url=base_url, timeout=timeout)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = {"backend": backend}
    if server_url:
        opts["server_url"] = server_url
    if extra_options:
        opts.update(extra_options)

    for doc_path in doc_path_list:
        doc_path = Path(doc_path)
        stem = doc_path.stem
        out_zip = output_dir / f"{stem}.zip"
        sub_dir = output_dir / stem

        print(f"\n=== parse {doc_path} ===")
        if use_async:
            tid = client.submit_task([doc_path], options=opts)
            client.wait_task(tid, out_zip=out_zip)
        else:
            client.parse_sync([doc_path], out_zip=out_zip, options=opts)

        unzip_result(out_zip, sub_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="+", help="一个或多个待解析文件")
    ap.add_argument("-o", "--output-dir", default="./output", help="结果输出目录")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="mineru-api 地址")
    ap.add_argument("-b", "--backend", default="vlm-auto-engine",
                    choices=["pipeline", "vlm-auto-engine", "vlm-http-client",
                             "hybrid-auto-engine", "hybrid-http-client"])
    ap.add_argument("-u", "--server-url", default=None,
                    help="*-http-client backend 才需要,指向 openai-server")
    ap.add_argument("--async", dest="use_async", action="store_true",
                    help="走 /tasks 异步端点(适合大文件)")
    ap.add_argument("--lang", default=None,
                    help="pipeline/hybrid backend 才用得到,例如 ch / en")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="HTTP 请求超时秒数(默认 600,大文件建议 1800+)")
    ap.add_argument("--start-page", type=int, default=None)
    ap.add_argument("--end-page", type=int, default=None)
    args = ap.parse_args()

    extra = {}
    if args.lang:
        extra["lang_list"] = args.lang
    if args.start_page is not None:
        extra["start_page_id"] = str(args.start_page)
    if args.end_page is not None:
        extra["end_page_id"] = str(args.end_page)

    parse_doc(
        doc_path_list=args.input,
        output_dir=args.output_dir,
        backend=args.backend,
        server_url=args.server_url,
        base_url=args.base_url,
        timeout=args.timeout,
        use_async=args.use_async,
        extra_options=extra,
    )
# 等价于你那条 curl
# python mineru_api_client.py \
#     /mnt/data/qyhuang/agentic_rag/PageIndex/examples/documents/attention-residuals.pdf \
#     -o ./output \
#     -b vlm-auto-engine