"""Shared state passed from ``app.py`` to every page."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AppContext:
    raw: pd.DataFrame                    # standardised input, before cleaning
    cleaned: pd.DataFrame                # cleaned, full dataset (not anonymised)
    report: dict                         # data-quality report from clean_dataset
    source_label: str                    # shown to the user
    using_sample: bool
    anonymize: bool
    scored: pd.DataFrame                 # cleaned + risk_probability + risk_category
    view: pd.DataFrame                   # filtered (and anonymised) for display
    trained: dict | None                 # training results, or None
    training_error: str | None           # why training was not possible
    model_name: str | None               # model used for risk estimates
    moderate: float
    high: float
    indicators: dict = field(default_factory=dict)
    filters_active: bool = False
    selected_filters: dict = field(default_factory=dict)

    @property
    def has_model(self) -> bool:
        return self.trained is not None

    @property
    def has_risk(self) -> bool:
        return "risk_category" in self.scored.columns
