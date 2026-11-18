Input files for eval tasks, referenced as `{files}/name` in suite YAML.

Copied from DataSleuth's `data/samples` (generated with seed 42). Seeded traps:

- `hr_attrition_trap.csv`: `exit_interview_completed` is filled only after an employee
  has left (leakage). Real, weaker signals: low satisfaction, short tenure.
- `sensor_readings_messy.csv`: three date formats, ~15% missing humidity, ~8% of
  temperatures in Fahrenheit, ~3% duplicate rows.
- `tiny_sample.csv`: 80 rows; Pro users churn a bit less, but not significantly at this size.

From DocIntel's synthetic set (`data/synthetic`, seeded generator), with answers from its
`ground_truth/*.json`:

- `docintel_retail_receipt.png` (000000): GSTIN 33DCTOS4025A9ZM, date 2025-12-15, total 733.28
- `docintel_gst_invoice.png` (000004): INV/25-26/52773, 2026-01-17, total 1412.39
- `docintel_delivery_challan.png` (000002): DC/26-27/89354, 2026-06-10, total 3343.77
