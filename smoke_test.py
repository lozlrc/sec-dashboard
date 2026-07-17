"""Live sanity check against the SEC EDGAR API. Run: uv run python smoke_test.py"""
import edgar
import metrics


def main() -> None:
    for ticker in ("AAPL", "MSFT", "KO"):
        info = edgar.resolve_cik(ticker)
        print(f"\n=== {ticker}  {info['title']}  CIK={info['cik']} ===")
        facts = edgar.get_company_facts(info["cik"])
        df = metrics.build_financials(facts, years=6)
        cols = [c for c in ("revenue", "net_income", "net_margin", "assets", "eps_diluted") if c in df.columns]
        print(df[cols].round(2).to_string())
        filings = edgar.get_recent_filings(info["cik"], limit=3)
        print("recent:", [(f["form"], f["filed"]) for f in filings])
        assert "revenue" in df.columns, f"{ticker}: no revenue extracted"
        assert df["revenue"].notna().sum() >= 3, f"{ticker}: too few revenue points"
    print("\nOK")


if __name__ == "__main__":
    main()
