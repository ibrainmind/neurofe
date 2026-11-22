"""Shared reportlab styles and helpers for syn_brain PDF reports.

Used by scripts/single_node_tms_report.py and scripts/*_report.py. Keeping this in one
place means every report looks the same and font/style fixes only need to be made once.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle

# DejaVu (shipped with matplotlib) covers Greek letters and math symbols.
_fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for _name, _file in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                     ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf"),
                     ("DVM", "DejaVuSansMono.ttf")):
    try:
        pdfmetrics.registerFont(TTFont(_name, str(_fd / _file)))
    except Exception:
        pass
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B"); addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9, leading=12.6, alignment=TA_JUSTIFY,
                      spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.6, leading=9.8, alignment=0, spaceAfter=0)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.8, leading=10, textColor=colors.HexColor("#444444"), spaceAfter=9)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DV-B", fontSize=14, leading=18, spaceBefore=8, spaceAfter=6,
                    textColor=colors.HexColor("#1d3557"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DV-B", fontSize=10.5, leading=14, spaceBefore=7, spaceAfter=4,
                    textColor=colors.HexColor("#1d3557"))
TITLE = ParagraphStyle("title", parent=H1, fontSize=19, leading=24, spaceAfter=4)
BOX = ParagraphStyle("box", parent=BODY, backColor=colors.HexColor("#eef3f8"), borderColor=colors.HexColor("#9db4cc"),
                     borderWidth=0.6, borderPadding=7, spaceBefore=4, spaceAfter=12)
VERDICT_OK = ParagraphStyle("vok", parent=BOX, backColor=colors.HexColor("#e6f4ea"), borderColor=colors.HexColor("#6aa876"))
VERDICT_BAD = ParagraphStyle("vbad", parent=BOX, backColor=colors.HexColor("#fbeaea"), borderColor=colors.HexColor("#c97b7b"))
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)

W = A4[0] - 3.6 * cm


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items, style=BUL):
    return [Paragraph(t, style, bulletText="•") for t in items]


def fig(path, caption, width=W):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, CAP)])


def table(rows, widths, header=True, zebra=True, font=7.6):
    st = ParagraphStyle("t", parent=SMALL, fontSize=font, leading=font * 1.28)
    sth = ParagraphStyle("th", parent=st, fontName="DV-B", textColor=colors.white)
    data = [[Paragraph(str(c), sth if (header and i == 0) else st) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    sty = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0da")),
           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        sty.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                sty.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f6f9")))
    t.setStyle(TableStyle(sty))
    return t


def pct(v):
    return f"{v:.2g}%" if v < 10 else (f"{v:.0f}%" if v < 1000 else ">1000%")


def make_footer(text):
    def footer(c, doc):
        c.saveState()
        c.setFont("DV", 7)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(1.8 * cm, 1.1 * cm, text)
        c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"{doc.page}")
        c.restoreState()
    return footer


def build(story, path, title, footer_text, author="syn_brain study"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.7 * cm, title=title, author=author)
    footer = make_footer(footer_text)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", path)
