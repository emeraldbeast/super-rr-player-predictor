"""Phase 4: rank robustness, pressure-cohort diagnostics and submission lists."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from superr.scoring import _posterior_rank, score_players  # noqa: E402


OUT = ROOT / "outputs"
FIG = OUT / "figures"


def _rank_variant(data: pd.DataFrame, predictions: pd.DataFrame, cfg: dict, *,
                  variant: str, expected_source: str = "selected", weighting: str = "continuous",
                  chase_only: bool = False, half_life: float | None = None,
                  min_matches: int = 10, min_balls: float = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = data.copy()
    if chase_only:
        d = d.loc[d["innings"].eq(2)].copy()
    d["pressure_weight"] = pd.to_numeric(d["pressure_weight"], errors="coerce").fillna(0)
    if weighting == "binary_high":
        d["pressure_weight"] = d["high_pressure"].fillna(False).astype(float)
    p = predictions.copy()
    for target in ("bat_runs", "bowler_conceded", "bowler_wicket"):
        col = f"expected_{target}_{expected_source}" if expected_source != "selected" else f"expected_{target}"
        p[f"expected_{target}"] = p[col]
    phase = dict(cfg)
    phase["phase3"] = dict(cfg["phase3"])
    phase["phase3"]["ranking_eligibility"] = {
        "min_pressure_matches": min_matches, "min_pressure_weighted_balls": min_balls
    }
    phase["phase3"]["candidate_eligibility"] = phase["phase3"]["ranking_eligibility"]
    bat, bowl, bm, wm, _, _, _ = score_players(d, p, phase)

    # score_players applies the exposure/cluster estimator and role shrinkage.
    # Recency weighting is applied at match level, preserving within-match clusters.
    if half_life is not None:
        bm = bm.copy(); wm = wm.copy()
        max_year = int(max(bm.season_year.max(), wm.season_year.max()))
        for frame in (bm, wm):
            factor = np.power(0.5, (max_year - frame["season_year"].astype(float)) / half_life)
            frame["weighted_residual"] *= factor
            frame["pressure_balls"] *= factor
            if "weighted_wicket_residual" in frame:
                frame["weighted_wicket_residual"] *= factor
            frame["rate"] = 100 * frame["weighted_residual"] / frame["pressure_balls"].replace(0, np.nan)
        bat = _rank_from_match(bm, cfg, min_matches, min_balls)
        bowl = _rank_from_match(wm, cfg, min_matches, min_balls)

    frames=[]
    for category, frame in (("batter", bat), ("bowler", bowl)):
        if frame.empty:
            continue
        frame=frame.copy(); frame["category"]=category; frame["variant"]=variant
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            pd.concat([bm.assign(category="batter"), wm.assign(category="bowler")], ignore_index=True))


def _rank_from_match(match_rates: pd.DataFrame, cfg: dict, min_matches: int, min_balls: float) -> pd.DataFrame:
    rows=[]
    match_rates=match_rates.rename(columns={"striker_id":"player_id","bowler_id":"player_id"})
    for pool, frame in match_rates.groupby("competition_pool", sort=False):
        result = _posterior_rank(frame, numerator_col="weighted_residual", minimum_matches=min_matches,
            minimum_balls=min_balls, lower_quantile=float(cfg["phase3"]["lower_posterior_quantile"]),
            minimum_prior_variance=float(cfg["phase3"]["minimum_prior_variance"]))
        if not result.empty:
            result.insert(0, "competition_pool", pool); rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
    common = a[["player_id", "rank"]].merge(b[["player_id", "rank"]], on="player_id", suffixes=("_a", "_b"))
    if len(common) < 3:
        return float("nan")
    return float(common["rank_a"].corr(common["rank_b"], method="spearman"))


def _calibration(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data=pred.copy()
    rows=[]
    for (pool, season, target, high), g in data.groupby(["competition_pool","season_year","target_name","high_pressure"], dropna=False):
        err=g["predicted"]-g["actual"]
        row={"pool":pool,"season":int(season),"target":target,"cohort":"high_pressure" if high else "ordinary","n":len(g),"bias_pred_minus_actual":float(err.mean())}
        if target == "bowler_wicket":
            row.update(brier=float(np.mean(err**2)), actual_rate=float(g.actual.mean()), predicted_rate=float(g.predicted.mean()))
        else:
            row.update(mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err**2))))
        rows.append(row)
    by_cohort=pd.DataFrame(rows)
    reliability=[]
    wicket=data.loc[data.target_name.eq("bowler_wicket")].copy()
    for (pool,season,high),g in wicket.groupby(["competition_pool","season_year","high_pressure"]):
        g["probability_bin"]=pd.qcut(g["predicted"].rank(method="first"), q=10, labels=False, duplicates="drop")
        for bin_id,b in g.groupby("probability_bin"):
            reliability.append({"pool":pool,"season":int(season),"cohort":"high_pressure" if high else "ordinary",
                "probability_bin":int(bin_id),"n":len(b),"mean_predicted_probability":float(b.predicted.mean()),
                "observed_wicket_rate":float(b.actual.mean())})
    return by_cohort, pd.DataFrame(reliability)


def main() -> None:
    cfg=yaml.safe_load((ROOT/"configs"/"config.yaml").read_text(encoding="utf-8"))
    data=pd.read_parquet(ROOT/"data"/"processed"/"model_table.parquet")
    preds=pd.read_parquet(ROOT/"data"/"processed"/"expected_outcomes_oof.parquet")
    validation=pd.read_parquet(ROOT/"data"/"processed"/"chronological_validation_predictions.parquet")
    # Exclude unavailable-pressure rows and post-match DLS/super-over rows in all variants.
    data=data.loc[data.pressure_available & ~data.is_super_over & ~data.is_dls].copy()
    variants=[
        ("primary_selected_continuous", "selected", "continuous", False, None, 10, 100),
        ("baseline_only_continuous", "baseline", "continuous", False, None, 10, 100),
        ("binary_high_pressure", "selected", "binary_high", False, None, 10, 100),
        ("chase_only", "selected", "continuous", True, None, 10, 100),
        ("recency_half_life_3y", "selected", "continuous", False, 3.0, 10, 100),
        ("eligibility_relaxed_3m_30b", "selected", "continuous", False, None, 3, 30),
        ("eligibility_strict_15m_200b", "selected", "continuous", False, None, 15, 200),
    ]
    ranked=[]; match_tables=[]
    for args in variants:
        table, matches=_rank_variant(data,preds,cfg,variant=args[0],expected_source=args[1],weighting=args[2],chase_only=args[3],half_life=args[4],min_matches=args[5],min_balls=args[6])
        ranked.append(table); match_tables.append(matches)
        print(f"Scored sensitivity variant: {args[0]}",flush=True)
    all_ranks=pd.concat(ranked,ignore_index=True)
    # Format-standardized corroboration view: compare each format against its
    # own role prior, then average z-scores only for players observed in both.
    base=all_ranks.loc[all_ranks.variant.eq("primary_selected_continuous")].copy()
    base["role_z"]=(base["posterior_rate"]-base["role_mean"])/np.sqrt(base["role_variance"].clip(lower=1e-9))
    cross_rows=[]
    for category, frame in base.groupby("category"):
        # Provider player IDs are pool-specific in some records. Join exact
        # source player names here, retaining the IPL ID solely as the output key.
        wide=frame.pivot_table(index="player_name",columns="competition_pool",values="role_z",aggfunc="first").dropna(subset=["IPL","T20I"])
        if wide.empty: continue
        meta=frame.loc[frame.competition_pool.eq("IPL")].set_index("player_name")
        for player_name,row in wide.iterrows():
            record=meta.loc[player_name].to_dict(); record.update({"competition_pool":"CROSS_FORMAT_COMMON","category":category,
                "variant":"cross_format_standardized","cross_format_role_z":float((row["IPL"]+row["T20I"])/2)})
            cross_rows.append(record)
    if cross_rows:
        cross=pd.DataFrame(cross_rows).sort_values("cross_format_role_z",ascending=False)
        cross["rank"]=cross.groupby("category").cumcount()+1
        all_ranks=pd.concat([all_ranks,cross],ignore_index=True,sort=False)
    all_ranks.to_csv(OUT/"phase4_variant_rankings.csv",index=False)
    primary=all_ranks.loc[all_ranks.variant.eq("primary_selected_continuous")]
    stability=[]
    for variant in all_ranks.variant.unique():
        for category in ("batter","bowler"):
            for pool in ("IPL","T20I","CROSS_FORMAT_COMMON"):
                reference_pool="IPL" if pool=="CROSS_FORMAT_COMMON" else pool
                ref=primary.loc[primary.category.eq(category)&primary.competition_pool.eq(reference_pool)]
                cur=all_ranks.loc[all_ranks.variant.eq(variant)&all_ranks.category.eq(category)&all_ranks.competition_pool.eq(pool)]
                if ref.empty or cur.empty: continue
                top_ref=set(ref.nsmallest(10,"rank").player_id)
                top_cur=set(cur.nsmallest(10,"rank").player_id)
                stability.append({"variant":variant,"category":category,"pool":pool,
                    "players_in_common":len(ref.merge(cur,on="player_id")),"spearman_rank_correlation":_spearman(ref,cur),
                    "top10_overlap":len(top_ref&top_cur),"top10_jaccard":len(top_ref&top_cur)/max(len(top_ref|top_cur),1)})
    stability=pd.DataFrame(stability); stability.to_csv(OUT/"phase4_rank_stability.csv",index=False)
    # Primary lists keep the current empirical-Bayes lower confidence score, including uncertainty.
    for category, source, filename in (("batter","phase3_batting_rankings_all_pools.csv","top10_batters.csv"),("bowler","phase3_bowling_rankings_all_pools.csv","top10_bowlers.csv")):
        frame=pd.read_csv(OUT/source)
        top=frame.loc[frame.competition_pool.eq("IPL")].sort_values("rank").head(10).copy()
        if len(top)!=10 or top.player_name.nunique()!=10: raise RuntimeError(f"Expected exactly 10 unique IPL {category}s")
        # Submission-specific lists are rank/name only; evidence stays in full ranking tables.
        top[["rank","player_name"]].rename(columns={"player_name":"player_name"}).to_csv(OUT/filename,index=False)
    # Copy validation metrics to a granular pressure-cohort diagnostic and a wicket reliability table.
    by_cohort,reliability=_calibration(validation)
    by_cohort.to_csv(OUT/"phase4_validation_by_pressure_cohort.csv",index=False)
    reliability.to_csv(OUT/"phase4_wicket_reliability.csv",index=False)
    # The validation predictions are CatBoost fold predictions; the paired summary table is the model comparison.
    metrics=pd.read_csv(OUT/"phase3_validation_metrics.csv")
    recent=metrics.loc[metrics.test_season.ge(metrics.test_season.max()-2)]
    avg=recent.groupby(["pool","target","model"],as_index=False).mean(numeric_only=True)
    avg.to_csv(OUT/"phase4_recent_model_comparison.csv",index=False)
    # Concise machine-readable roll-up.
    summary={"variants":len(variants),"rank_rows":len(all_ranks),"validation_rows":len(validation),
        "high_pressure_validation_rows":int(validation.high_pressure.sum()),
        "ipl_top10_batters":pd.read_csv(OUT/"top10_batters.csv").player_name.tolist(),
        "ipl_top10_bowlers":pd.read_csv(OUT/"top10_bowlers.csv").player_name.tolist(),
        "primary_eligibility":{"min_pressure_matches":10,"min_pressure_weighted_balls":100},
        "auction_pool_available":False,"rr_roster_available":False}
    (OUT/"phase4_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.mkdir(parents=True,exist_ok=True)
        for category in ("batter","bowler"):
            ss=stability.loc[stability.category.eq(category)&stability.pool.eq("IPL")]
            fig,ax=plt.subplots(figsize=(9,4.5))
            ax.bar(ss.variant,ss.top10_overlap,color="#db6b3c")
            ax.set_ylim(0,10); ax.set_ylabel("Players retained in top 10"); ax.set_xlabel(""); ax.tick_params(axis="x",rotation=25)
            ax.grid(axis="y",alpha=.25); ax.set_axisbelow(True)
            ax.set_title(f"IPL {category.title()} top-10 sensitivity")
            fig.tight_layout(); fig.savefig(FIG/f"phase4_{category}_rank_stability.png",dpi=180); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        rel=reliability.loc[reliability.cohort.eq("high_pressure")]
        if not rel.empty:
            ax.plot(rel.mean_predicted_probability,rel.observed_wicket_rate,"o",alpha=.5)
            lo=min(rel.mean_predicted_probability.min(),rel.observed_wicket_rate.min()); hi=max(rel.mean_predicted_probability.max(),rel.observed_wicket_rate.max())
            ax.plot([lo,hi],[lo,hi],"--",color="grey")
        ax.set_xlabel("Mean predicted wicket probability"); ax.set_ylabel("Observed wicket rate"); ax.set_title("High-pressure wicket calibration (held-out seasons)")
        fig.tight_layout(); fig.savefig(FIG/"phase4_high_pressure_wicket_calibration.png",dpi=180); plt.close(fig)
    except ImportError:
        print("matplotlib/seaborn unavailable; CSV diagnostics retained",flush=True)
    print("Phase 4 outputs written",flush=True)


if __name__ == "__main__":
    main()
