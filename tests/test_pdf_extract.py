"""Parser test against a synthetic annual-report-style PDF (fpdf2-generated)."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpdf import FPDF

import pdf_extract

INCOME_PAGE = [
    "CONSOLIDATED INCOME STATEMENT",
    "For the year ended 31 December",
    "(RMB in millions, unless otherwise stated)",
    "",
    "Note 2023 2022",
    "Revenues 5 609,015 554,552",
    "Cost of revenues 7 (315,906) (315,806)",
    "Gross profit 293,109 238,746",
    "Interest income 13,808 8,592",
    "Selling and marketing expenses (34,211) (29,229)",
    "Research and development expenses 7 (64,078) (61,401)",
    "Operating profit 8 160,074 108,323",
    "Profit for the year 118,048 88,243",
    "",
    "Earnings per share (RMB per share)",
    "Diluted earnings per share 11.888 8.895",
]

BALANCE_PAGE = [
    "CONSOLIDATED STATEMENT OF FINANCIAL POSITION",
    "As at 31 December",
    "(RMB in millions)",
    "",
    "2023 2022",
    "Total assets 1,577,246 1,610,919",
    "Total liabilities 663,214 731,110",
    "Total equity 914,032 879,809",
    "Cash and cash equivalents 172,320 144,955",
]

CASHFLOW_PAGE = [
    "CONSOLIDATED STATEMENT OF CASH FLOWS",
    "(RMB in millions)",
    "",
    "2023 2022",
    "Net cash generated from operating activities 221,962 146,383",
    "Net cash used in investing activities (127,594) (60,443)",
]


def make_pdf() -> bytes:
    pdf = FPDF()
    pdf.set_font("Courier", size=9)
    for page in (INCOME_PAGE, BALANCE_PAGE, CASHFLOW_PAGE):
        pdf.add_page()
        for line in page:
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def test_parses_synthetic_annual_report():
    ext = pdf_extract.parse_pdf(make_pdf())
    assert ext.ok
    assert ext.currency == "RMB"
    assert ext.scale == 1e6
    df = ext.df

    assert df.loc[2023, "revenue"] == 609_015e6
    assert df.loc[2022, "revenue"] == 554_552e6
    assert df.loc[2023, "cost_of_revenue"] == -315_906e6  # parentheses -> negative
    assert df.loc[2023, "gross_profit"] == 293_109e6
    assert df.loc[2023, "operating_income"] == 160_074e6
    assert df.loc[2023, "net_income"] == 118_048e6
    assert df.loc[2023, "rnd"] == -64_078e6
    assert df.loc[2023, "assets"] == 1_577_246e6
    assert df.loc[2023, "liabilities"] == 663_214e6
    assert df.loc[2023, "equity"] == 914_032e6
    assert df.loc[2023, "op_cash_flow"] == 221_962e6
    # per-share value must NOT get the millions multiplier
    assert df.loc[2023, "eps_diluted"] == 11.888

    # note-reference columns ("Note 5", "Note 7") must not leak into values
    assert df.loc[2022, "cost_of_revenue"] == -315_806e6

    derived = pdf_extract.add_pdf_derived(df)
    assert round(derived.loc[2023, "net_margin"], 1) == 19.4
    assert derived.loc[2023, "revenue_growth"] > 9


def test_rejects_empty_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=9)
    pdf.cell(0, 5, "This report contains no financial tables.", new_x="LMARGIN", new_y="NEXT")
    ext = pdf_extract.parse_pdf(bytes(pdf.output()))
    assert not ext.ok
