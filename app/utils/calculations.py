from typing import List

""" calculate Compound Annual Growth Rate (CAGR) """
def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0 or years <= 0:
        return 0
    return (end_value / start_value) ** (1 / years) - 1

def calculate_xirr(cash_flow: List[float], dates: List, guess: float = 0.1) -> float:
    return 0.0

