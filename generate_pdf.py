import subprocess
import sys
import os
import re

def ensure(package, import_name=None):
    name = import_name or package
    try:
        __import__(name)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

ensure("reportlab", "reportlab")
ensure("markdown")

from markdown import markdown
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, HRFlowable
)
from reportlab.lib.enums import TA_LEFT

CODE_BG = colors.HexColor("#f4f4f4")
HEADING1_COLOR = colors.HexColor("#1a1a1a")
HEADING2_COLOR = colors.HexColor("#2c2c2c")
ACCENT = colors.HexColor("#444444")

def build_styles():
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=22,
                             leading=28, textColor=HEADING1_COLOR,
                             spaceAfter=6, spaceBefore=12),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=16,
                             leading=22, textColor=HEADING2_COLOR,
                             spaceAfter=4, spaceBefore=20),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=13,
                             leading=18, textColor=ACCENT,
                             spaceAfter=2, spaceBefore=14),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=11,
                               leading=16, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=11,
                                 leading=16, leftIndent=18, spaceAfter=3,
                                 bulletIndent=6),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=9,
                               leading=13, backColor=CODE_BG,
                               leftIndent=12, rightIndent=12,
                               spaceBefore=6, spaceAfter=6),
    }

def parse_md(md_path):
    with open(md_path, encoding="utf-8") as f:
        return f.read()

def md_to_flowables(text, styles):
    flowables = []
    in_code = False
    code_lines = []

    def flush_code():
        if code_lines:
            block = "\n".join(code_lines)
            flowables.append(Preformatted(block, styles["code"]))
            flowables.append(Spacer(1, 4))
            code_lines.clear()

    for line in text.splitlines():
        # fenced code blocks
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        # HR
        if re.match(r"^-{3,}$", line.strip()):
            flowables.append(Spacer(1, 6))
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#cccccc")))
            flowables.append(Spacer(1, 6))
            continue

        # Headings
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            content = m.group(2).strip()
            key = f"h{level}" if level <= 3 else "body"
            flowables.append(Paragraph(content, styles[key]))
            if level == 1:
                flowables.append(HRFlowable(width="100%", thickness=1,
                                            color=colors.HexColor("#333333")))
                flowables.append(Spacer(1, 4))
            elif level == 2:
                flowables.append(HRFlowable(width="100%", thickness=0.4,
                                            color=colors.HexColor("#cccccc")))
                flowables.append(Spacer(1, 2))
            continue

        # Bullet / numbered list
        if re.match(r"^[-*]\s+", line):
            content = re.sub(r"^[-*]\s+", "", line)
            content = inline_fmt(content)
            flowables.append(Paragraph(f"• {content}", styles["bullet"]))
            continue
        if re.match(r"^\d+\.\s+", line):
            content = re.sub(r"^\d+\.\s+", "", line)
            m2 = re.match(r"^(\d+)\.", line)
            num = m2.group(1) if m2 else "1"
            content = inline_fmt(content)
            flowables.append(Paragraph(f"{num}. {content}", styles["bullet"]))
            continue

        # Blank line
        if line.strip() == "":
            flowables.append(Spacer(1, 4))
            continue

        # Normal paragraph
        flowables.append(Paragraph(inline_fmt(line), styles["body"]))

    flush_code()
    return flowables

def inline_fmt(text):
    # bold **...**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # inline code `...`
    text = re.sub(r"`([^`]+)`",
                  r'<font name="Courier" size="9">\1</font>', text)
    return text

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(root, "SETUP.md")
    pdf_path = os.path.join(root, "SETUP.pdf")

    styles = build_styles()
    text = parse_md(md_path)

    print("Building PDF...")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LETTER,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    doc.build(md_to_flowables(text, styles))
    print(f"Done — saved to {pdf_path}")

if __name__ == "__main__":
    main()
