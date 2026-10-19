import pandas as pd


def solution(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    # 1. Inner merge: keep only employees present in BOTH df1 and df2
    inner = pd.merge(df1, df2, on="emp_id", how="inner")

    # 2. Left merge: attach salary from df3, preserving all records from the inner merge
    result = pd.merge(inner, df3, on="emp_id", how="left")

    # 3. Ensure expected column order and reset the index
    return (
        result[["emp_id", "name", "dept", "salary"]]
        .reset_index(drop=True)
    )