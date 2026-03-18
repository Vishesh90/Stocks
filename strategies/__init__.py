"""strategies/__init__.py"""
from strategies.standard import get_all_standard_strategies
from strategies.mathematical import get_all_mathematical_strategies

def get_all_strategies():
    return get_all_standard_strategies() + get_all_mathematical_strategies()
