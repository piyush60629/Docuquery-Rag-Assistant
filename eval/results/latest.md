# DocuQuery evaluation results

- Run: 2026-10-19 10:30 UTC  
- Mode: full · chunking sections · chunk size 900 · overlap 150 · top-k 4  
- Models: embeddings `gemini-embedding-001`, answers `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b` (via groq)  
- Questions: 35 (30 answerable, 5 unanswerable) · chunks indexed: 30 · errors: 0

| Metric | Score |
|---|---|
| Retrieval Hit@1 | 96.7% |
| Retrieval Hit@4 | 100.0% |
| Retrieval MRR | 0.983 |
| Answer faithfulness | 100.0% |
| Answer correctness | 98.3% |
| Refusal accuracy (unanswerable) | 100.0% |
| False refusal rate (lower is better) | 0.0% |

## Per-question results

| ID | Rank | Refused | Faithful | Correct | Note |
|---|---|---|---|---|---|
| r01 | 1 | no | 1 | 1 | The answer correctly identifies the start time as 2:15 AM, which is explicitly stated in the retrieved context and match |
| r02 | 2 | no | 1 | 1 | The answer correctly states that files with unexpected schemas are moved to a quarantine folder for review, which is ful |
| r03 | 1 | no | 1 | 1 | The answer accurately reflects the retrieved context and matches the reference answer regarding the destination of inval |
| r04 | 1 | no | 1 | 1 | The answer accurately reflects the retrieved context and matches the reference answer regarding the watermark update con |
| r05 | 1 | no | 1 | 1 | The answer correctly lists all four measures found in the retrieved context and matches the reference answer exactly. |
| r06 | 1 | no | 1 | 1 | The answer accurately reflects the retrieved context regarding Git commits, PR reviews, and Azure DevOps deployment, mat |
| r07 | 1 | no | 1 | 1 | The answer accurately reflects the steps described in Source 1 and matches the reference answer. |
| r08 | 1 | no | 1 | 1 | The answer correctly identifies the three specific metadata fields mentioned in the retrieved context and matches the re |
| d01 | 1 | no | 1 | 1 | The answer correctly identifies the 2 percent threshold for optional columns as stated in the retrieved context, matchin |
| d02 | 1 | no | 1 | 1 | The answer correctly identifies that the batch is stopped and the Gold layer is not refreshed, matching the reference an |
| d03 | 1 | no | 1 | 1 | The answer accurately reflects the retrieved context and matches the reference answer regarding the assignment of placeh |
| d04 | 1 | no | 1 | 1 | The answer accurately reflects the information in Source 1 and matches the reference answer exactly. |
| d05 | 1 | no | 1 | 0.5 | The answer correctly identifies the 0.5 percent threshold but omits the important exception for records rejected by vali |
| d06 | 1 | no | 1 | 1 | The answer correctly identifies the finance analyst as the approver, which is explicitly supported by Source 1 and match |
| d07 | 1 | no | 1 | 1 | The answer correctly states the refresh time as 7:00 AM, which is fully supported by Source 1 and matches the reference  |
| d08 | 1 | no | 1 | 1 | The answer accurately reflects the definition provided in Source 1 and matches the reference answer exactly. |
| d09 | 1 | no | 1 | 1 | The answer accurately reflects the 30 percent threshold and comparison method specified in the retrieved context and mat |
| i01 | 1 | no | 1 | 1 | The answer correctly identifies the expired storage account secret in Azure Key Vault as the root cause, which is fully  |
| i02 | 1 | no | 1 | 1 | The answer accurately reflects the resolution details (timeout increase to 90 minutes and rerun after approval) found in |
| i03 | 1 | no | 1 | 1 | The answer accurately reflects the root cause described in the retrieved context and matches the reference answer. |
| i04 | 1 | no | 1 | 1 | The answer accurately reflects the root cause described in Source 1, matching the reference answer's details about data  |
| i05 | 1 | no | 1 | 1 | The answer accurately reflects the root cause described in the retrieved context and matches the reference answer. |
| i06 | 1 | no | 1 | 1 | The answer correctly identifies 'region_code' as the new column causing the incident, which is fully supported by Source |
| i07 | 1 | no | 1 | 1 | The answer accurately reflects the prevention step described in Source 1 and matches the reference answer. |
| o01 | 1 | no | 1 | 1 | The answer correctly identifies the deadline as 6:00 AM IST, which is explicitly supported by the retrieved context and  |
| o02 | 1 | no | 1 | 1 | The answer accurately reflects the duration and specific times stated in the retrieved context and matches the reference |
| o03 | 1 | no | 1 | 1 | The answer correctly identifies the 'nb_' prefix and provides the exact example found in the retrieved context, matching |
| o04 | 1 | no | 1 | 1 | The answer correctly identifies the branch name format and the base branch (main) as stated in the retrieved context, ma |
| o05 | 1 | no | 1 | 1 | The answer correctly states the two-working-day timeframe and is fully supported by the retrieved context. |
| o06 | 1 | no | 1 | 1 | The answer correctly identifies the action (notify reporting team) and the deadline (5:30 AM) as stated in the retrieved |
| u01 | - | yes |  |  |  |
| u02 | - | yes |  |  |  |
| u03 | - | yes |  |  |  |
| u04 | - | yes |  |  |  |
| u05 | - | yes |  |  |  |
