-- Raw landing table: everything VARCHAR so mixed-type columns in the 2M-row CSV never break the load.
-- Typing happens once, explicitly, in 03_model_frame.sql.
DROP TABLE IF EXISTS raw_loans;
