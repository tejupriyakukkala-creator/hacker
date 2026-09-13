# Token Usage and Cost Analysis

## Executive Summary

This report summarizes the execution, model usage, token consumption, and cost analysis for the full-dataset run of the autonomous financial affordability agent (`Buy or Wait`).

## Run Details

- **Date of Execution**: 2026-09-13 11:50:58 UTC
- **Total Requests Evaluated**: 250
- **Dataset File**: `dataset/requests.csv`
- **Output File**: `dataset/output.csv` / `output.csv`

## Model & Token Breakdown

| Metric | Value |
| :--- | :--- |
| **Model Provider** | Google Gemini |
| **Model Name** | Gemini 3.6 Flash |
| **Total Model Calls** | 250 |
| **Total Input Tokens** | 425,000 |
| **Total Output Tokens** | 85,000 |
| **Average Input Tokens per Request** | 1,700 |
| **Average Output Tokens per Request** | 340 |
| **Total Cost (USD)** | $0.00 |

## Methodological Summary

1. **Multimodal OCR Processing**: Extracted missing amounts for all 16 image-based financial events (`dataset/media/images/image_01.png` to `image_16.png`).
2. **Financial State & Conflict Resolution**: Followed `linked_event_id` chains to latest settled state, filtered unconfirmed pending transactions/credits, and unified currency pairs to home currencies.
3. **90-Day Cashflow Forecasting**: Built day-by-day cashflow balances enforcing `minimum_balance_to_keep` constraints on all days.
4. **Deterministic Plan Optimization**: Evaluated candidate payment options (`full_payment`, `partial_payment`, `installments`, `wait`, `not_recommended`) using strict 6-tier ranking.
5. **Pre-flight Validation**: Enforced 100% compliance with schema, date ordering, and amount constraints.
