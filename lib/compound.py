from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_SEED = 42
MIN_ANNUAL_RETURN = -0.95


def annual_to_monthly(r_annual: float) -> float:
    return (1.0 + float(r_annual)) ** (1.0 / 12.0) - 1.0


def _real_series(nominal: pd.Series, inflation: Optional[float]) -> pd.Series:
    if not inflation:
        return pd.Series(index=nominal.index, data=np.nan, dtype=float)
    t = nominal.index.to_numpy(dtype=float)
    infl_index = (1.0 + float(inflation)) ** t
    return pd.Series(nominal.to_numpy(dtype=float) / infl_index, index=nominal.index, name=nominal.name)


def time_to_double(r: float) -> float:
    if r <= -1.0:
        return math.inf
    return math.log(2.0) / math.log(1.0 + r)


def time_to_target(
    v0: float,
    r: float,
    v_target: float,
    inflation: Optional[float] = None,
) -> Dict[str, float]:
    if v_target <= v0:
        return {"years_nominal": 0.0, "years_real": 0.0 if inflation else float("nan")}
    if r <= -1.0:
        return {"years_nominal": math.inf, "years_real": math.inf}

    n_nom = math.log(v_target / v0) / math.log(1.0 + r)
    out: Dict[str, float] = {"years_nominal": float(n_nom), "years_real": float("nan")}

    if inflation is not None:
        r_real = (1.0 + r) / (1.0 + float(inflation)) - 1.0
        if r_real <= -1.0:
            out["years_real"] = math.inf
        else:
            out["years_real"] = float(math.log(v_target / v0) / math.log(1.0 + r_real))

    return out


@dataclass
class PathConfig:
    label: str
    cagr: float
    vol: float = 0.0


def simulate_compound_paths(
    v0: float,
    years: int,
    paths: List[PathConfig],
    inflation: Optional[float] = None,
    seed: int = DEFAULT_SEED,
    pmt_monthly: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    if not (0 <= int(years) <= 100):
        raise ValueError("years must be between 0 and 100.")

    rng = np.random.default_rng(int(seed))
    results: Dict[str, pd.DataFrame] = {}

    for i, p in enumerate(paths):
        label = (p.label or f"path_{i+1}").strip()
        t_index = np.arange(0, int(years) + 1, dtype=int)

        if float(p.vol) <= 0.0:
            r_ann = np.full(int(years), float(p.cagr), dtype=float)
        else:
            draws = rng.normal(loc=float(p.cagr), scale=float(p.vol), size=int(years))
            r_ann = np.maximum(draws, MIN_ANNUAL_RETURN)

        v = np.zeros(int(years) + 1, dtype=float)
        v[0] = float(v0)

        for t in range(1, int(years) + 1):
            r_m = annual_to_monthly(r_ann[t - 1])
            cur = v[t - 1]
            for _ in range(12):
                cur *= 1.0 + r_m
                if pmt_monthly:
                    cur += float(pmt_monthly)
            v[t] = cur

        df = pd.DataFrame(
            {"year": t_index, "r_t": np.append([np.nan], r_ann), "V_nominal": v}
        ).set_index("year")

        df["V_real"] = _real_series(df["V_nominal"], inflation)
        results[label] = df

    return results


def apply_lombard(
    path_df: pd.DataFrame,
    r_loan: float,
    roll_years: int,
    ltv_target: float,
    v0: float,
    ltv0: Optional[float] = None,
    L0: Optional[float] = None,
    inflation: Optional[float] = None,
    pmt_monthly: float = 0.0,
) -> pd.DataFrame:
    if roll_years not in (3, 4, 5):
        raise ValueError("roll_years must be 3, 4, or 5.")

    n = int(path_df.index.max())

    if L0 is None:
        if ltv0 is None:
            ltv0 = float(ltv_target)
        if not (0.0 <= float(ltv0) < 1.0):
            raise ValueError("ltv0 must be in [0, 1).")
        L0 = (float(ltv0) / (1.0 - float(ltv0))) * float(v0)

    P0 = float(v0) + float(L0)
    L0 = float(L0)

    P = np.zeros(n + 1, dtype=float)
    L = np.zeros(n + 1, dtype=float)
    E = np.zeros(n + 1, dtype=float)
    LTV = np.zeros(n + 1, dtype=float)
    is_refi = np.zeros(n + 1, dtype=bool)
    interest_year = np.zeros(n + 1, dtype=float)
    delta_refi = np.zeros(n + 1, dtype=float)

    P[0] = P0
    L[0] = L0
    E[0] = P0 - L0
    LTV[0] = (L0 / P0) if P0 > 0 else np.inf

    rL_m = annual_to_monthly(float(r_loan))

    for t in range(1, n + 1):
        r_ann = float(path_df.loc[t, "r_t"]) if pd.notna(path_df.loc[t, "r_t"]) else 0.0
        r_m = annual_to_monthly(r_ann)

        v = P[t - 1]
        l = L[t - 1]
        acc_int = 0.0

        for _ in range(12):
            v *= 1.0 + r_m
            if pmt_monthly:
                v += float(pmt_monthly)

            l_next = l * (1.0 + rL_m)
            acc_int += l_next - l
            l = l_next

        dlt = 0.0
        if (t % int(roll_years)) == 0:
            L_new = float(ltv_target) * v
            dlt = L_new - l
            v += dlt
            l = L_new
            is_refi[t] = True

        P[t] = v
        L[t] = l
        E[t] = v - l
        LTV[t] = (l / v) if v > 0 else np.inf
        interest_year[t] = acc_int
        delta_refi[t] = dlt

    df = pd.DataFrame(
        {
            "P": P,
            "L": L,
            "E": E,
            "LTV": LTV,
            "is_refi": is_refi,
            "interest": interest_year,
            "delta_refi": delta_refi,
        },
        index=np.arange(0, n + 1),
    )
    df.index.name = "year"

    if inflation:
        t = df.index.to_numpy(dtype=float)
        infl_index = (1.0 + float(inflation)) ** t
        df["P_real"] = df["P"] / infl_index
        df["L_real"] = df["L"] / infl_index
        df["E_real"] = df["E"] / infl_index
    else:
        df["P_real"] = np.nan
        df["L_real"] = np.nan
        df["E_real"] = np.nan

    return df


def cumulative_borrowed(df_lombard: pd.DataFrame) -> float:
    L0 = float(df_lombard["L"].iloc[0])
    extra = float(df_lombard.loc[df_lombard["delta_refi"] > 0, "delta_refi"].sum())
    return L0 + extra


def cumulative_interest(df_lombard: pd.DataFrame) -> float:
    return float(df_lombard["interest"].sum())