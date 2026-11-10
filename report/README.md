# Findings report

- `PHPFL_Class_Fairness_Report.pdf` — detailed 14-page findings report.
- `PHPFL_All_Results.pdf` — 12-page results-only compendium with every committed table.
- `build_report.py` — ReportLab builder for the findings report.
- `build_results_pdf.py` — ReportLab builder for the results compendium.
- `make_report_figures.py` — regenerates the report figures from committed CSVs.
- `figures/` — report figures at 250 DPI.

Rebuild:

```bash
python report/make_report_figures.py
python report/build_report.py
python report/build_results_pdf.py
```

The reports read only committed artifacts under `phofl/analysis`,
`phofl_method`, and `phofl_half/results`; they do not train models.
