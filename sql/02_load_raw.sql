-- {{RAW_CSV}} is substituted by src/data/loader.py
CREATE TABLE raw_loans AS
SELECT
    loan_status, loan_amnt, funded_amnt, int_rate, grade, sub_grade, term,
    annual_inc, dti, emp_length, home_ownership, purpose,
    fico_range_low, fico_range_high, delinq_2yrs, revol_util, issue_d,
    out_prncp, recoveries, total_rec_prncp
FROM read_csv('{{RAW_CSV}}', header = true, all_varchar = true, ignore_errors = true, sample_size = -1);
