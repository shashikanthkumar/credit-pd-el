-- Model frame: typed, term-filtered, with target and vintage / maturity flags.
-- {{TERM}} and {{SNAPSHOT}} are substituted from configs/config.yaml.
-- Rows are KEPT for in-flight loans (target NULL) so PSI can use them; performance code filters is_resolved.
DROP TABLE IF EXISTS model_frame;
CREATE TABLE model_frame AS
WITH typed AS (
    SELECT
        loan_status,
        TRY_CAST(loan_amnt AS DOUBLE)       AS loan_amnt,
        TRY_CAST(funded_amnt AS DOUBLE)     AS funded_amnt,
        TRY_CAST(replace(int_rate, '%', '') AS DOUBLE) AS int_rate,
        trim(grade)                         AS grade,
        trim(sub_grade)                     AS sub_grade,
        TRY_CAST(regexp_extract(term, '(\d+)', 1) AS INTEGER) AS term_months,
        TRY_CAST(annual_inc AS DOUBLE)      AS annual_inc,
        TRY_CAST(dti AS DOUBLE)             AS dti,
        NULLIF(trim(emp_length), '')        AS emp_length_raw,
        trim(home_ownership)                AS home_ownership,
        trim(purpose)                       AS purpose,
        TRY_CAST(fico_range_low AS DOUBLE)  AS fico_range_low,
        TRY_CAST(fico_range_high AS DOUBLE) AS fico_range_high,
        TRY_CAST(delinq_2yrs AS DOUBLE)     AS delinq_2yrs,
        TRY_CAST(replace(revol_util, '%', '') AS DOUBLE) AS revol_util,
        TRY_STRPTIME(trim(issue_d), '%b-%Y') AS issue_dt,
        TRY_CAST(out_prncp AS DOUBLE)       AS out_prncp,
        TRY_CAST(recoveries AS DOUBLE)      AS recoveries,
        TRY_CAST(total_rec_prncp AS DOUBLE) AS total_rec_prncp
    FROM raw_loans
)
SELECT
    *,
    CAST(issue_dt AS DATE)                                   AS issue_date,
    EXTRACT(year FROM issue_dt)::INTEGER                     AS vintage_year,
    (fico_range_low + fico_range_high) / 2.0                 AS fico_mid,
    CASE WHEN annual_inc > 0 THEN loan_amnt / annual_inc END AS loan_to_income,
    CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1
         WHEN loan_status = 'Fully Paid'                THEN 0
         ELSE NULL END                                       AS target,
    (loan_status IN ('Charged Off', 'Default', 'Fully Paid')) AS is_resolved,
    date_diff('month', CAST(issue_dt AS DATE), DATE '{{SNAPSHOT}}')  AS months_observed,
    -- explicit date arithmetic (issue + term <= snapshot); date_diff('month') ignores the day-of-month
    (CAST(issue_dt AS DATE) + to_months(CAST(term_months AS INTEGER)) <= DATE '{{SNAPSHOT}}') AS fully_matured
FROM typed
WHERE term_months = {{TERM}}
  AND issue_dt IS NOT NULL;
