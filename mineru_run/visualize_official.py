"""
基于 MinerU 官方 mineru.utils.draw_bbox 的可视化封装。
依赖: pip install -U mineru pymupdf

用法:
    python visualize_official.py <pdf_path> <middle_json_path> <out_dir> [--png] [--span] [--no-labels]

输出:
    out_dir/layout.pdf      —— 带类别文字标签的块级布局(默认,显示 TITLE / IMAGE / TABLE 等)
    out_dir/span.pdf        —— span 级细分(可选,加 --span;官方在某些 middle.json 上有 bug)
    可选:out_dir/png/...   —— PNG 渲染,加 --png 启用
    --no-labels             —— 退回官方版(只有色块,无文字)

"""
import os
import json
import sys
import argparse
from io import BytesIO

from pypdf import PdfReader, PdfWriter, PageObject
from reportlab.pdfgen import canvas

from draw_bbox import (
    draw_layout_bbox,
    draw_span_bbox,
    cal_canvas_rect,
    BlockType,
    SplitFlag,
)


# 颜色取自官方 draw_layout_bbox,文字标签便于一眼识别类别
TYPE_STYLE = {
    BlockType.TITLE:              ((102, 102, 255), "TITLE"),
    BlockType.TEXT:               ((153, 0,   76 ), "TEXT"),
    BlockType.REF_TEXT:           ((153, 0,   76 ), "REF_TEXT"),
    BlockType.ABSTRACT:           ((153, 0,   76 ), "ABSTRACT"),
    BlockType.INTERLINE_EQUATION: ((0,   180, 0  ), "EQUATION"),
    BlockType.LIST:               ((40,  169, 92 ), "LIST"),
    BlockType.INDEX:              ((40,  169, 92 ), "INDEX"),
    BlockType.SEAL:               ((153, 255, 51 ), "SEAL"),
    # 容器型 block 展开后的叶子类型
    BlockType.IMAGE_BODY:         ((153, 255, 51 ), "IMAGE"),
    BlockType.IMAGE_CAPTION:      ((102, 178, 255), "IMG_CAPTION"),
    BlockType.IMAGE_FOOTNOTE:     ((255, 178, 102), "IMG_FOOTNOTE"),
    BlockType.TABLE_BODY:         ((204, 204, 0  ), "TABLE"),
    BlockType.TABLE_CAPTION:      ((218, 165, 32 ), "TBL_CAPTION"),
    BlockType.TABLE_FOOTNOTE:     ((180, 200, 120), "TBL_FOOTNOTE"),
    BlockType.CHART_BODY:         ((255, 140, 0  ), "CHART"),
    BlockType.CHART_CAPTION:      ((102, 178, 255), "CHART_CAPTION"),
    BlockType.CHART_FOOTNOTE:     ((255, 178, 102), "CHART_FOOTNOTE"),
    BlockType.CODE_BODY:          ((102, 0,   204), "CODE"),
    BlockType.CODE_CAPTION:       ((204, 153, 255), "CODE_CAPTION"),
    BlockType.CODE_FOOTNOTE:      ((229, 204, 255), "CODE_FOOTNOTE"),
}
DISCARDED_STYLE = ((158, 158, 158), "DROP")


def _iter_labeled_bboxes(page):
    """对单页 para_blocks 展平,产出 (bbox, type) 序列。"""
    for blk in page.get("para_blocks", []):
        t = blk["type"]
        if t in (BlockType.IMAGE, BlockType.TABLE, BlockType.CHART, BlockType.CODE):
            for sub in blk.get("blocks", []):
                if sub.get(SplitFlag.CROSS_PAGE, False):
                    continue
                yield sub["bbox"], sub["type"]
        elif t == BlockType.LIST:
            yield blk["bbox"], t
            # 列表内部的 list item 不再单独贴标签,避免标签互相覆盖
        else:
            yield blk["bbox"], t


def draw_layout_bbox_with_labels(pdf_info, pdf_bytes, out_path, filename,
                                 draw_discarded=True):
    """
    画出每个 block 的 bbox,并在左上角贴一个该 block 的类别名标签。
    输出与官方 draw_layout_bbox 一样的 PDF,但增加文字标签。
    """
    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        page_w, page_h = float(page.cropbox[2]), float(page.cropbox[3])
        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=(page_w, page_h))

        page_info = pdf_info[i]

        # 被丢弃的 block(页眉页脚等)
        if draw_discarded:
            for blk in page_info.get("discarded_blocks", []):
                _draw_box_with_tag(c, page, blk["bbox"], DISCARDED_STYLE[0],
                                   DISCARDED_STYLE[1], dashed=True)

        # 正文 block
        for bbox, t in _iter_labeled_bboxes(page_info):
            rgb, label = TYPE_STYLE.get(t, ((128, 128, 128), t.upper()))
            _draw_box_with_tag(c, page, bbox, rgb, label)

        c.save()
        packet.seek(0)
        overlay = PdfReader(packet)

        if len(overlay.pages) > 0:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            new_page.merge_page(overlay.pages[0])
            output.add_page(new_page)
        else:
            output.add_page(page)

    with open(os.path.join(out_path, filename), "wb") as f:
        output.write(f)


def _draw_box_with_tag(c, page, bbox, rgb, label, dashed=False):
    r, g, b = [v / 255.0 for v in rgb]
    rect = cal_canvas_rect(page, bbox)
    x, y, w, h = rect

    # 边框
    c.setStrokeColorRGB(r, g, b)
    c.setLineWidth(1.2)
    if dashed:
        c.setDash(3, 2)
    else:
        c.setDash()
    c.rect(x, y, w, h, stroke=1, fill=0)
    c.setDash()

    # 左上角的类别标签
    font_name, font_size = "Helvetica-Bold", 7
    tag_w = c.stringWidth(label, font_name, font_size) + 4
    tag_h = font_size + 2
    tag_x = x
    tag_y = y + h - tag_h
    if tag_y < 0:
        tag_y = y  # 极端情况兜底

    c.setFillColorRGB(r, g, b, 1.0)
    c.rect(tag_x, tag_y, tag_w, tag_h, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1, 1.0)
    c.setFont(font_name, font_size)
    c.drawString(tag_x + 2, tag_y + 2, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("middle_json_path")
    ap.add_argument("out_dir")
    ap.add_argument("--png", action="store_true",
                    help="额外把 layout.pdf 渲成 PNG(需要 pymupdf)")
    ap.add_argument("--span", action="store_true",
                    help="额外画 span 级细分(官方函数在嵌套容器型 block 上可能崩,默认关闭)")
    ap.add_argument("--no-labels", action="store_true",
                    help="退回官方版,只画色块不写类别文字")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.middle_json_path, "r", encoding="utf-8") as f:
        middle = json.load(f)
    with open(args.pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_info = middle["pdf_info"]

    # 块级可视化
    if args.no_labels:
        draw_layout_bbox(pdf_info, pdf_bytes, args.out_dir, "layout.pdf")
    else:
        draw_layout_bbox_with_labels(pdf_info, pdf_bytes, args.out_dir, "layout.pdf")
    print("[ok]", os.path.join(args.out_dir, "layout.pdf"))

    # span 级可视化(可选,且容错)
    if args.span:
        try:
            draw_span_bbox(pdf_info, pdf_bytes, args.out_dir, "span.pdf")
            print("[ok]", os.path.join(args.out_dir, "span.pdf"))
        except KeyError as e:
            # 官方 draw_span_bbox 在嵌套容器型 block(image/table/chart/code)上会因为
            # 直接访问 block['lines'] 而 KeyError。这里给一份打过补丁的 middle.json:
            # 把没有 lines 的容器 block 跳过(或临时塞个空 lines)。
            print(f"[warn] draw_span_bbox 原生失败: {e},改用补丁版重试")
            patched = _patch_middle_for_span(middle)
            draw_span_bbox(patched["pdf_info"], pdf_bytes, args.out_dir, "span.pdf")
            print("[ok]", os.path.join(args.out_dir, "span.pdf"), "(补丁版)")

    if args.png:
        import fitz
        png_dir = os.path.join(args.out_dir, "png")
        os.makedirs(png_dir, exist_ok=True)
        zoom = args.dpi / 72
        for src_name in ("layout.pdf", "span.pdf"):
            src = os.path.join(args.out_dir, src_name)
            if not os.path.exists(src):
                continue
            tag = src_name.replace(".pdf", "")
            doc = fitz.open(src)
            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                out = os.path.join(png_dir, f"{tag}_page_{i + 1:03d}.png")
                pix.save(out)
            doc.close()
            print(f"[ok] rendered {src_name} -> {png_dir}/{tag}_page_*.png")


def _patch_middle_for_span(middle):
    """
    深拷贝并修复 middle.json,使其能跑过官方 draw_span_bbox:
      - 容器型 block(image/table/chart/code,内部有 'blocks' 子数组、没有 'lines')
        被 draw_span_bbox 当成普通 block 处理时会崩。
      - 把容器 block 的子 block 提到同级,这样每个 block 都有 lines 字段。
    """
    import copy
    new = copy.deepcopy(middle)
    for page in new["pdf_info"]:
        for key in ("preproc_blocks", "para_blocks"):
            if key not in page:
                continue
            flat = []
            for blk in page[key]:
                if "blocks" in blk and "lines" not in blk:
                    # 把内层叶子 block 提到同级
                    flat.extend(blk["blocks"])
                else:
                    flat.append(blk)
            # 再兜底:依然没有 lines 的塞个空列表
            for b in flat:
                b.setdefault("lines", [])
            page[key] = flat
    return new

# python visualize_official.py  origin.pdf middle.json ./viz_out --png
if __name__ == "__main__":
    main()
