from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import (
    calculate_gfevd,
    calculate_gis,
    calculate_hasbrouck_proxy,
    calculate_pairwise_hasbrouck_proxy,
)


def make_demo_prices(n_obs: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(scale=0.01, size=n_obs))
    micro_noise = rng.normal(scale=[0.006, 0.008, 0.010], size=(n_obs, 3))
    log_prices = trend[:, None] + micro_noise
    return pd.DataFrame(
        np.exp(log_prices),
        columns=["BTC", "AAPL", "NVDA"],
        index=pd.date_range("2020-01-01", periods=n_obs, freq="D"),
    )


def main() -> None:
    prices = make_demo_prices()

    gis = calculate_gis(prices)
    gfevd = calculate_gfevd(prices, horizon=20)
    hasbrouck = calculate_hasbrouck_proxy(prices, max_orderings=60)
    pairwise = calculate_pairwise_hasbrouck_proxy(prices)

    print("\nGIS")
    print(gis.shares.round(4))

    print("\nGFEVD, rows receive shocks, columns are shock sources")
    print(gfevd.table.round(4))
    print("\nConnectedness")
    print(gfevd.connectedness.round(4))
    print(f"Total connectedness: {gfevd.total_connectedness:.4f}")

    print("\nHasbrouck proxy")
    print(hasbrouck.summary.round(4))

    print("\nPairwise Hasbrouck proxy")
    print(pairwise.round(4))


if __name__ == "__main__":
    main()
