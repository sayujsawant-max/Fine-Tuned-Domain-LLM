# Report assets

Supporting assets for the FinSage-7B benchmark report exports.

- `style.css` — stylesheet applied to the HTML export (`benchmark_report.html`).
- `logo_placeholder.svg` — placeholder logo; replace with a real logo if desired.

Charts (`report_*.png`) are written to `reports/figures/` by the report
generator and referenced from the Markdown report with relative paths.

Regenerate everything with:

```bash
make report          # real report (labelled sample until a real fine-tune exists)
make report-mock     # sample report from fixtures, to /tmp
make validate-report # structural validation
```
