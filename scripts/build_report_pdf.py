"""Render report/report.md to report/report.pdf using markdown_pdf."""
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "report" / "report.md"
DST = REPO / "report" / "report.pdf"

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.45; color: #1c1c1c; }
h1 { font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 4pt; margin-top: 24pt; }
h2 { font-size: 16pt; color: #2a4d7a; margin-top: 18pt; }
h3 { font-size: 13pt; color: #444; margin-top: 12pt; }
h4 { font-size: 11pt; color: #555; }
pre, code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; }
pre { background: #f4f4f4; border: 1px solid #d0d0d0; border-radius: 4px; padding: 8pt; overflow-x: auto; }
code { background: #f4f4f4; padding: 1pt 4pt; border-radius: 3px; }
blockquote { border-left: 3px solid #2a4d7a; padding: 4pt 12pt; color: #444; background: #f9f9fb; margin: 8pt 0; }
table { border-collapse: collapse; margin: 8pt 0; width: 100%; font-size: 10pt; }
th, td { border: 1px solid #b0b0b0; padding: 4pt 8pt; text-align: left; }
th { background: #ececec; }
a { color: #1a5fb4; }
hr { border: none; border-top: 1px solid #aaa; margin: 16pt 0; }
"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    pdf = MarkdownPdf(toc_level=0, optimize=True)
    pdf.meta["title"] = "DSAI 413 Assignment 2 — ColPali-RAG vs MedGemma"
    pdf.meta["author"] = "Assignment 2 Multimedia"
    pdf.add_section(Section(text), user_css=CSS)
    pdf.save(str(DST))
    print(f"Wrote {DST}  ({DST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
