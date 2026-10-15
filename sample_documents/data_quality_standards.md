# Retail Data Quality Standards

This document defines the data quality rules, thresholds, and ownership for the retail analytics platform. All examples use synthetic data.

## Severity Levels

Every data quality rule has one of three severity levels.

- **Critical**: the batch is stopped and the Gold layer is not refreshed. Examples are duplicate transaction identifiers and a row count drop of more than 5 percent between Bronze and Silver.
- **Warning**: the batch continues, the failed records are written to the error table, and an alert is sent to the data quality channel.
- **Informational**: the result is only logged to the data quality dashboard for trend analysis.

## Thresholds

- Mandatory columns must have a null rate of 0 percent.
- Optional columns may have a null rate of up to 2 percent before a Warning is raised.
- The row count variance between consecutive pipeline stages must stay within 0.5 percent, except for records rejected by validation rules.
- Daily sales volume is compared with the average of the same weekday over the previous four weeks. A change of more than 30 percent raises a Warning for business review.
- Late-arriving sales records older than 7 days are loaded into a separate late-arrivals table instead of the main fact table.

## Reference Data Rules

Product and store references are validated against the current dimension tables. A sales record with an unknown product identifier is assigned the placeholder product key -1 and flagged for the product data steward. Unknown store identifiers follow the same rule with store key -1.

## Duplicate Handling

A duplicate is defined as two records with the same transaction identifier and line number. Exact duplicates are removed in the Silver layer and the first occurrence is kept. If the same file is delivered twice, the file hash check in the Bronze layer detects it and skips the second file.

## Ownership

- The **data engineering team** owns the pipelines, rule implementation, and reruns.
- The **product data steward** owns product master data corrections.
- The **finance analyst** approves any manual adjustment to net sales amounts.

## Reporting

Data quality results are stored in the dq_results table and shown on the data quality dashboard, which refreshes every morning at 7:00 AM. Rule changes require a pull request reviewed by at least one senior data engineer.
