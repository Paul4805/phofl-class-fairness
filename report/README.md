# Findings report

- `PHPFL_Class_Fairness_Report.pdf` — detailed 14-page report.
- `build_report.py` — ReportLab builder.
- `make_report_figures.py` — regenerates the report figures from committed CSVs.
- `figures/` — report figures at 250 DPI.

Rebuild:

```bash
python report/make_report_figures.py
python report/build_report.py
```

The report reads only committed artifacts under `phofl/analysis`,
`phofl_method`, and `phofl_half/results`; it does not train models.
