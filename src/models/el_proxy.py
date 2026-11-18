"""ILLUSTRATIVE LGD / EAD proxies. Not production models, not Basel/IRB, no downturn adjustment.

EAD proxy: balance at default = funded_amnt - total_rec_prncp for charged-off loans.
  CCF_proxy = mean(EAD / funded_amnt) on train defaults (term loans are fully drawn; the factor is amortisation).
LGD proxy: exposure-weighted 1 - recoveries / EAD on train defaults, clipped to [0, 1].
Both are estimated on TRAIN defaults only. Realised-recovery data in the snapshot is not discounted for time value.
"""
import numpy as np
import pandas as pd


def fit_lgd_ead_proxy(train_defaults: pd.DataFrame) -> dict:
    d = train_defaults.copy()
    ead = (d["funded_amnt"] - d["total_rec_prncp"]).clip(lower=0)
    ok = ead > 0
    ccf = float((ead[ok] / d.loc[ok, "funded_amnt"]).clip(0, 1).mean())
    lgd = float(np.clip(1 - d.loc[ok, "recoveries"].sum() / ead[ok].sum(), 0, 1))
    return dict(lgd=lgd, ccf=ccf, n_defaults=int(ok.sum()),
                note="Illustrative proxies estimated on train defaults only.")
