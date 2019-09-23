import pandas as pd

class ContextDependentSampler:
    def __init__(self, df: pd.DataFrame, cat_vars: list):
        self.cat_prob = {}

