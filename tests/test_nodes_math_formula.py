from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import math_formula  # noqa: E402


def test_math_formula_translates_knime_log_to_log10() -> None:
    translated = math_formula._translate_expression("log($Total applications$)")

    assert translated == "np.log10(df['Total applications'])"


def test_math_formula_translates_knime_ln_to_natural_log() -> None:
    translated = math_formula._translate_expression("ln($Total applications$)")

    assert translated == "np.log(df['Total applications'])"


def test_math_formula_inz_log_node_matches_log10_values() -> None:
    body = "\n".join(
        math_formula.generate_py_body(
            "1536",
            "tests/data/INZ_visa_decisions_unification/Math Formula (#1536)",
            [("1531", "1")],
            ["1"],
        )
    )
    context = {
        "1531:1": pd.DataFrame(
            {
                "Country": ["Afghanistan", "Albania"],
                "Total applications": [71, 7],
            }
        )
    }

    exec(body, {"pd": pd, "np": np, "context": context})

    out = context["1536:1"]
    assert np.allclose(out["Applications_log10"].to_numpy(), np.log10([71, 7]))
