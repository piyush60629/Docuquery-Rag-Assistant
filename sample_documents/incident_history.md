# Retail Pipeline Incident History

Synthetic incident records used for troubleshooting practice. Each entry lists the symptom, root cause, resolution, and prevention step.

## INC-1042: Store file rejected because of schema drift

- **Symptom**: The store master file was moved to the quarantine folder and the store dimension was not refreshed.
- **Root cause**: The source system added a new column called region_code without notice.
- **Resolution**: The column mapping in the ingestion pipeline was updated and the quarantined file was reprocessed.
- **Prevention**: A schema change notification was added to the source system release checklist.

## INC-1057: Watermark not updated after a Gold layer timeout

- **Symptom**: The next daily run reprocessed the previous day's records.
- **Root cause**: The Gold layer notebook exceeded the 60-minute activity timeout, so the batch was marked as failed and the watermark stayed unchanged.
- **Resolution**: The timeout was increased to 90 minutes and the failed batch was rerun after approval.
- **Prevention**: The notebook was optimized by partitioning the sales fact table by transaction date.

## INC-1063: Linked service authentication failure

- **Symptom**: All copy activities failed with an authentication error at 2:15 AM.
- **Root cause**: The storage account secret in Azure Key Vault had expired.
- **Resolution**: The secret was rotated in Key Vault and the pipeline was rerun.
- **Prevention**: An expiry alert was configured 30 days before any Key Vault secret expires.

## INC-1071: Duplicate sales after a file was resent

- **Symptom**: Net sales in the Gold layer were double for one store on one day.
- **Root cause**: The source team resent the same sales file with a new file name, so the file name check did not detect it.
- **Resolution**: The duplicate rows were removed and the Gold tables were rebuilt for that date.
- **Prevention**: Duplicate detection was changed to use a file hash instead of the file name.

## INC-1088: Databricks job out of memory on a skewed join

- **Symptom**: The Silver to Gold transformation failed with an out of memory error on the largest store.
- **Root cause**: One store had far more transactions than others, which caused data skew in the join with the product dimension.
- **Resolution**: The product dimension was broadcast and adaptive query execution was enabled.
- **Prevention**: A skew check was added to the weekly performance review.

## INC-1095: Snowflake load stopped by the resource monitor

- **Symptom**: The final load into Snowflake failed and reports showed the previous day's data.
- **Root cause**: The monthly credit quota on the Snowflake resource monitor was reached and the warehouse was suspended.
- **Resolution**: The platform owner approved a quota increase and the load was rerun.
- **Prevention**: A notification was added when 80 percent of the monthly credit quota is used.
