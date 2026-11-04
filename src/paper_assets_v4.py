"""Builds LaTeX macros and tables for the revised MLWA manuscript (v4) from the CSV/JSON outputs of
MLWA_experiments_v4.ipynb.  Usage: build(src_dir, out_dir) -> out_dir/results_v4.tex, out_dir/tables/*.tex
The same source is embedded in the notebook (Section 26a)."""
import os, json
import numpy as np
import pandas as pd

GATE = {"MobileNetV3Small": "MobileNetV3-S", "MobileNetV3Large": "MobileNetV3-L",
        "EfficientNetB0": "EfficientNet-B0", "ResNet50": "ResNet50", "HOG": "HOG", "ImgStats": "Image statistics"}
HANDCRAFTED = ("HOG", "ImgStats")
STRAT = {"LR": "LR", "LR-weighted": "Weighted LR", "LR-SMOTE": "LR + SMOTE", "Focal-weighted": "Weighted focal",
         "OneClass-Mahalanobis": "One-class Mahalanobis", "OneClass-kNN": "One-class $k$NN"}
SETTING = {"CPU": "CPU", "hybrid": "Hyb", "GPU": "GPU"}
SETTING_TXT = {"CPU": "CPU (gate + HED on CPU)", "hybrid": "Hybrid (gate on GPU, HED on CPU)", "GPU": "GPU (gate + HED on GPU)"}
NA = r"\tbd{n/a}"
MANUSCRIPT_MACROS = ['ApparentCPURecall', 'ApparentCPUS', 'BestCPHigh', 'BestCPLow', 'BestCalibAFPR', 'BestCalibARecall', 'BestCalibBRecall', 'BestCalibCFPR', 'BestCalibCRecall', 'BestDetected', 'BestFPR', 'BestFPRHalf', 'BestFPRHi', 'BestFPRLo', 'BestFrameFPR', 'BestFrameS', 'BestGroupFPR', 'BestGroupS', 'BestHead', 'BestMissHalf', 'BestMixedFPRR', 'BestMixedPRAUC', 'BestName', 'BestPRAUC', 'BestPRAUCFold', 'BestPRAUCFoldSD', 'BestQ', 'BestRecall', 'BestRecallHalf', 'BestRecallLB', 'BestRecallLegacy', 'BestRhoCPU', 'BestSCPU', 'BestSCPUHi', 'BestSCPULo', 'BestSGPU', 'BestStationFPR', 'BestStationMissed', 'BestStationRecall', 'CPUCores', 'CPUModel', 'CVVersion', 'EdgeCannyF', 'EdgeHEDHalfF', 'EdgeHEDHalfS', 'EdgeHEDTQF', 'EdgeHEDTQS', 'EdgeSobelF', 'EdgeTol', 'EffFullCPUms', 'EffFullGFLOPs', 'EtoEBestCPUFPS', 'EtoEBestCPUS', 'EtoEBestCPUSD', 'EtoEBestCPUms', 'EtoEBestGPUFPS', 'EtoEBestGPUS', 'EtoEBestGPUms', 'EtoEBestHybS', 'EtoEBestHybms', 'EtoEGBestHybS', 'EtoEGBestHybSpred', 'EtoEMaxErr', 'EtoEMeanAbsErr', 'EtoERefCPUS', 'EtoERefHybS', 'FLOPsLatCorr', 'FPSHedCPU', 'FastHead', 'FastName', 'FastRecall', 'FastSCPU', 'FocalAlpha', 'FocalGamma', 'FocalLtwo', 'GBestCPUms', 'GBestFPR', 'GBestHead', 'GBestName', 'GBestPRAUC', 'GBestQ', 'GBestRecall', 'GBestRhoGPU', 'GBestSCPU', 'GBestSGPU', 'GFLOPsRange', 'GPUModel', 'GPUmsMax', 'GPUmsMin', 'GatingIdenticalDiff', 'HED', 'HOGCPUms', 'HOGFPR', 'HOGRecall', 'HOGSCPU', 'HedGFLOPs', 'HedGPUSpeedup', 'HedOverResFLOPs', 'HedOverResLat', 'HedPortDiff', 'ImgStatsCPUms', 'ImgStatsFPR', 'ImgStatsRecall', 'ImgStatsSCPU', 'JackBestMax', 'JackBestMed', 'JackBestMissedFull', 'JackBestMissedWorst', 'JackBestN', 'JackBestShare', 'JointRate', 'LHedCPUms', 'LHedCPUsd', 'LHedGPUms', 'LRC', 'LegacyFixedTau', 'LegacyFixedValFP', 'LegacyFixedValQ', 'LegacyStarTau', 'LogoBestAugRecall', 'LogoBestCleanRecall', 'LogoRefAugRecall', 'LogoRefCleanRecall', 'MNSHalfPRAUC', 'MirFPROn', 'MirFPRPlainOn', 'MixedFrames', 'MixedJoints', 'NAugCopies', 'NBackground', 'NBoot', 'NCalibSplits', 'NCandidates', 'NConfigs', 'NEtoERuns', 'NFolds', 'NFoldsTotal', 'NFrames', 'NHeadTests', 'NInner', 'NJoints', 'NJointsOnlyJointStations', 'NNestedInner', 'NNestedSeeds', 'NPerturb', 'NRecJoints', 'NRecordings', 'NRepeats', 'NStations', 'NStratFullRecall', 'NStratSup', 'NSup', 'NSupFullRecall', 'NestedCPUFPR', 'NestedCPUMissedTotal', 'NestedCPUNDistinct', 'NestedCPUNFolds', 'NestedCPURecall', 'NestedCPURecallMin', 'NestedCPUS', 'NestedCPUTop', 'OneClassBestFPR', 'OneClassBestName', 'OneClassBestPRAUC', 'OneClassBestRecall', 'PertBestAugMeanFPR', 'PertBestAugMeanRecall', 'PertBestAugWorstName', 'PertBestAugWorstRecall', 'PertBestCleanGlareFPR', 'PertBestCleanGlareRecall', 'PertBestCleanMaxFPR', 'PertBestCleanMeanFPR', 'PertBestCleanMeanRecall', 'PertBestCleanN', 'PertBestCleanNFull', 'PertBestCleanWorstName', 'PertBestCleanWorstRecall', 'PrevBestCPUHigh', 'PrevBestCPULow', 'PrevHigh', 'PrevLow', 'RefCPUms', 'RefCalibARecall', 'RefCalibCRecall', 'RefFPR', 'RefFrameFPR', 'RefFrameS', 'RefGroupFPR', 'RefGroupS', 'RefHead', 'RefName', 'RefRecall', 'RefRhoCPU', 'RefSCPU', 'RefSGPU', 'RefStationFPR', 'RefStationMissed', 'ResHalfCPUms', 'ResHalfGFLOPs', 'RhoCPUMax', 'RhoCPUMin', 'RhoFullMax', 'RhoFullMin', 'RhoGPUMaxPct', 'RuleBestConfBound', 'RuleBestConfRecall', 'RuleBestFixedRecall', 'RuleBestNoMarginRecall', 'RuleBestRninetyfiveRecall', 'RuleBestRninetyfiveSCPU', 'SmoteK', 'StOnlyMixedFPRR', 'StOnlyMixedPRAUC', 'StationFPRMax', 'StationFPRMin', 'StationsNoJoints', 'StationsOnlyJoints', 'StratRecallMin', 'StratSMax', 'StratSMin', 'StreamBestCPUMaxCam', 'SupMinPRAUCBestGate', 'TFVersion', 'TauBestMax', 'TauBestMean', 'TauBestMin', 'TauBestSD', 'Platform']


def _cfg(g, s):
    return GATE.get(g, g) if g in HANDCRAFTED else f"{GATE.get(g, g)} ({float(s):g}$\\times$)"


def _cfg_str(c):
    """'EfficientNetB0@0.5 LR-weighted' -> ('EfficientNet-B0 (0.5x)', 'Weighted LR')"""
    gs, st = c.split(" ", 1)
    g, s = gs.split("@")
    return _cfg(g, s), STRAT.get(st, st)


def _pm(x):
    """'0.123 ± 0.004' -> (0.123, 0.004)"""
    if isinstance(x, (float, int, np.floating, np.integer)) and not isinstance(x, bool):
        return float(x), 0.0
    if not isinstance(x, str) or "±" not in x:
        return np.nan, np.nan
    a, b = x.split("±")
    return float(a), float(b)


def _num(v, nd):
    s = f"{v:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s          # never print "-0"


def _fmt_pm(x, scale=1.0, nd=1):
    """mean +- SD, SD always shown (Reviewer 1, comment 8)"""
    m, s = _pm(x)
    if np.isnan(m):
        return "--"
    return f"{_num(m * scale, nd)} $\\pm$ {_num(abs(s) * scale, nd)}"


def _f(x, nd=1, scale=1.0):
    try:
        if x is None or np.isnan(float(x)):
            return "--"
    except (TypeError, ValueError):
        return str(x)
    return _num(float(x) * scale, nd)


def _err(x):
    v = float(f"{x:.0f}")
    return "0" if v == 0 else f"{v:+.0f}"


def _write(path, body):
    with open(path, "w") as f:
        f.write(body)


def _tabular(cols, header, rows, note=None):
    lines = [r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}" + cols + r"@{}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines += [r"\par\vspace{2pt}{\footnotesize " + note + r"\par}"]
    return "\n".join(lines) + "\n"


def build(src, out):
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    rd = lambda n: (pd.read_csv(os.path.join(src, n), dtype={"station": str}) if os.path.exists(os.path.join(src, n)) else None)
    T = lambda name, body: _write(os.path.join(out, "tables", name + ".tex"), body)
    cfg = json.load(open(os.path.join(src, "config.json")))
    env = json.load(open(os.path.join(src, "environment.json")))
    sel = json.load(open(os.path.join(src, "selection.json")))
    best, ref = tuple(sel["best"]), tuple(sel["reference"])
    gbest = tuple(sel["gpu_best"]) if sel.get("gpu_best") else None
    key = lambda c: f"{c[0]}@{c[1]} {c[2]}"
    best_s, ref_s, gbest_s = key(best), key(ref), (key(gbest) if gbest else None)
    acc = tuple(sel["most_accurate"]) if sel.get("most_accurate") else None
    acc_s = key(acc) if acc else None
    ROLE = {"best": best, "gpu_best": gbest, "reference": ref}
    TAG = {"best": "Best", "gpu_best": "GBest", "reference": "Ref"}
    ds, st = rd("dataset_summary.csv").iloc[0], rd("frames_per_station.csv")
    cx, lat, cvm, cvs = rd("complexity.csv"), rd("latency_cpu_gpu.csv"), rd("cv_grouped_mean.csv"), rd("cv_grouped_sd.csv")
    bpg, strat, rob = rd("best_per_gate.csv"), rd("strategies_on_best_gate.csv"), rd("robustness_protocols.csv")
    mir, leg, pst, conf = rd("mirror_stress_test.csv"), rd("legacy_gate_threshold.csv"), rd("per_station_logo.csv"), rd("confound_check.csv")
    gpu = "GPU" in set(lat.device)
    M = {}
    N_hand = len([h for h in cfg.get("HANDCRAFTED", []) if h in set(cvm.gate)])

    # ---------------- dataset, protocol, environment ----------------
    pi = ds.joints / ds.frames
    n_cnn = len(cfg["BACKBONES"]) * len(cfg["SCALES"])
    M.update({"NFrames": int(ds.frames), "NJoints": int(ds.joints), "NBackground": int(ds.background),
              "JointRate": f"{100 * pi:.1f}", "NRecordings": int(ds.recordings), "NRecJoints": int(ds.recordings_with_joints),
              "NStations": int(ds.stations), "NFolds": cfg["N_FOLDS"], "NInner": cfg["INNER_FOLDS"],
              "NRepeats": len(cfg["CV_SEEDS"]), "NGates": n_cnn, "NStrategies": len(cfg["STRATEGIES"]),
              "NConfigs": n_cnn * len(cfg["STRATEGIES"]), "NHand": N_hand,
              "NCandidates": (n_cnn + N_hand) * len(cfg["STRATEGIES"]), "NFoldsTotal": cfg["N_FOLDS"] * len(cfg["CV_SEEDS"]),
              "NBoot": cfg.get("BOOTSTRAP", 0), "NNestedInner": cfg.get("NESTED_INNER_FOLDS", 4),
              "NNestedSeeds": len(cfg.get("NESTED_SEEDS", [])), "NCalibSplits": cfg.get("CALIB_SPLITS", 0),
              "NAugCopies": cfg.get("AUG_COPIES", 0), "NPerturb": len(cfg.get("PERTURBATIONS", [])),
              "EdgeTol": cfg.get("EDGE_TOL_PX", 2), "SmoteK": cfg.get("SMOTE_K", 5), "LRC": f"{cfg['C']:g}",
              "FocalGamma": f"{cfg['FOCAL_GAMMA']:g}", "FocalAlpha": f"{cfg['FOCAL_ALPHA']:g}", "FocalLtwo": f"{cfg['FOCAL_L2']:g}",
              "FTEpochs": cfg.get("FT_EPOCHS", 0), "FTLR": f"{cfg.get('FT_LR', 0):g}", "FTBatch": cfg.get("FT_BATCH", 0),
              "CPUModel": env["cpu_model"].replace("(R)", "").replace("CPU ", "").replace("  ", " "),
              "CPUCores": env["cpu_logical_cores"], "Platform": env.get("platform") or "Google Colaboratory",
              "GPUModel": (env.get("gpu_model") or "none").split(",")[0].replace("Tesla ", ""), "TFVersion": env["tensorflow"],
              "CVVersion": env["opencv"]})
    only_joint = st[(st.joints == st.frames) & (st.frames > 0)].station.tolist()
    no_joint = st[st.joints == 0].station.tolist()
    M["StationsOnlyJoints"] = ", ".join(only_joint) if only_joint else "none"
    M["NStationsOnlyJoints"] = len(only_joint)
    M["NJointsOnlyJointStations"] = int(st[st.station.isin(only_joint)].joints.sum())
    M["StationsNoJoints"] = ", ".join(no_joint) if no_joint else "none"

    # ---------------- complexity and latency ----------------
    hed = lat[(lat.model == "HED") & (lat.device == "CPU")].iloc[0]
    hed_g = lat[lat.model.str.startswith("HED (PyTorch")]
    port = sel.get("hed_gpu_port") or {}
    hedfl = float(cx[cx.gate.str.startswith("HED")].GFLOPs.iloc[0])
    M.update({"LHedCPUms": f"{hed.mean_ms:.0f}", "LHedCPUsd": f"{hed.sd_ms:.0f}", "HedGFLOPs": f"{hedfl:.1f}",
              "FPSHedCPU": f"{1000 / hed.mean_ms:.1f}", "HedPort": "yes" if port.get("ok") else "no",
              "HedPortDiff": (f"{port['max_abs_diff']:.1e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}")
              if port.get("max_abs_diff") is not None and np.isfinite(port.get("max_abs_diff", np.inf)) else NA})
    if len(hed_g):
        M.update({"LHedGPUms": f"{hed_g.mean_ms.iloc[0]:.1f}", "FPSHedGPU": f"{1000 / hed_g.mean_ms.iloc[0]:.0f}",
                  "HedGPUSpeedup": f"{hed.mean_ms / hed_g.mean_ms.iloc[0]:.0f}"})

    def lrow(g, s, dev):
        r = lat[(lat.model == g) & (lat.scale == float(s)) & (lat.device == dev)]
        return r.iloc[0] if len(r) else None

    def crow(g, s):
        r = cx[(cx.gate == g) & (cx.scale == float(s))]
        return r.iloc[0] if len(r) else None

    for role, c in ROLE.items():
        if c is None:
            continue
        tg = TAG[role]
        M[f"{tg}Name"] = _cfg(c[0], c[1]); M[f"{tg}Head"] = STRAT[c[2]]
        cr = crow(c[0], c[1])
        M[f"{tg}GFLOPs"] = f"{cr.GFLOPs:.2f}" if cr is not None else "--"
        M[f"{tg}ParamsM"] = f"{cr.params_M:.1f}" if cr is not None else "--"
        M[f"{tg}CPUms"] = f"{lrow(c[0], c[1], 'CPU').mean_ms:.1f}"; M[f"{tg}RhoCPU"] = f"{lrow(c[0], c[1], 'CPU').rho:.3f}"
        if gpu:
            lg = lrow(c[0], c[1], "GPU")
            M[f"{tg}GPUms"] = f"{lg.mean_ms:.1f}"; M[f"{tg}RhoGPU"] = f"{lg.rho:.3f}"
            if len(hed_g):
                M[f"{tg}RhoFull"] = f"{lg.mean_ms / hed_g.mean_ms.iloc[0]:.2f}"
    gates = lat[~lat.model.str.startswith("HED") & ~lat.model.isin(HANDCRAFTED)]
    gfl = cx[~cx.gate.str.startswith("HED") & ~cx.gate.isin(HANDCRAFTED)].GFLOPs
    M["GFLOPsRange"] = f"{gfl.max() / gfl.min():.0f}"
    M["RhoCPUMin"], M["RhoCPUMax"] = f"{gates[gates.device == 'CPU'].rho.min():.3f}", f"{gates[gates.device == 'CPU'].rho.max():.3f}"
    if gpu:
        g = gates[gates.device == "GPU"]
        M["RhoGPUMin"], M["RhoGPUMax"] = f"{g.rho.min():.3f}", f"{g.rho.max():.3f}"
        M["GPUmsMin"], M["GPUmsMax"] = f"{g.mean_ms.min():.1f}", f"{g.mean_ms.max():.1f}"
        M["RhoGPUMaxPct"] = f"{100 * g.rho.max():.1f}"
        if len(hed_g):
            M["RhoFullMin"], M["RhoFullMax"] = f"{g.mean_ms.min() / hed_g.mean_ms.iloc[0]:.2f}", f"{g.mean_ms.max() / hed_g.mean_ms.iloc[0]:.2f}"
    for h in HANDCRAFTED:
        lh = lrow(h, 1.0, "CPU")
        if lh is not None:
            M[f"{h}CPUms"] = f"{lh.mean_ms:.1f}"; M[f"{h}Rho"] = f"{lh.rho:.3f}"
    try:
        rf = crow("ResNet50", 1.0)
        M["HedOverResFLOPs"] = f"{hedfl / rf.GFLOPs:.1f}"
        M["HedOverResLat"] = f"{hed.mean_ms / lrow('ResNet50', 1.0, 'CPU').mean_ms:.1f}"
    except (IndexError, AttributeError, TypeError):
        M["HedOverResFLOPs"] = M["HedOverResLat"] = NA
    cpu = gates[gates.device == "CPU"].merge(cx, left_on=["model", "scale"], right_on=["gate", "scale"])
    M["FLOPsLatCorr"] = f"{np.corrcoef(np.log(cpu.GFLOPs), np.log(cpu.mean_ms))[0, 1]:.2f}"
    for k_, g_, s_ in [("EffFull", "EfficientNetB0", 1.0), ("ResHalf", "ResNet50", 0.5)]:
        try:
            M[k_ + "GFLOPs"], M[k_ + "CPUms"] = f"{crow(g_, s_).GFLOPs:.2f}", f"{lrow(g_, s_, 'CPU').mean_ms:.1f}"
        except (IndexError, AttributeError, TypeError):
            M[k_ + "GFLOPs"] = M[k_ + "CPUms"] = NA

    rows = []
    for g_, s_ in [(g, s) for g in cfg["BACKBONES"] for s in cfg["SCALES"]]:
        c, lc, lg = crow(g_, s_), lrow(g_, s_, "CPU"), lrow(g_, s_, "GPU") if gpu else None
        r = [_cfg(g_, s_), c.input.replace("x", r"$\times$"), f"{c.params_M:.1f}", f"{c.GFLOPs:.2f}",
             f"{lc.mean_ms:.1f} $\\pm$ {lc.sd_ms:.1f}", f"{lc.rho:.3f}"]
        if gpu:
            r += [f"{lg.mean_ms:.1f} $\\pm$ {lg.sd_ms:.1f}", f"{lg.rho:.3f}"]
        rows.append(r)
    for h in HANDCRAFTED:
        lc = lrow(h, 1.0, "CPU")
        if lc is None:
            continue
        r = [GATE[h] + r"$^{b}$", "--", "--", "--", f"{lc.mean_ms:.1f} $\\pm$ {lc.sd_ms:.1f}", f"{lc.rho:.3f}"]
        if gpu:
            r += ["--", "--"]
        rows.append(r)
    hr = ["HED (stage 2)", cx[cx.gate.str.startswith("HED")].input.iloc[0].replace("x", r"$\times$"), "14.7", f"{hedfl:.1f}",
          f"{hed.mean_ms:.0f} $\\pm$ {hed.sd_ms:.0f}", "1"]
    if gpu:
        hr += ([f"{hed_g.mean_ms.iloc[0]:.1f} $\\pm$ {hed_g.sd_ms.iloc[0]:.1f}$^{{a}}$", "--"] if len(hed_g) else ["--", "--"])
    rows.append(hr)
    head = ["Gate (input scale)", "Input", "Params (M)", "GFLOPs", "CPU (ms)", r"$\rho_{\mathrm{CPU}}$"]
    if gpu:
        head += ["GPU (ms)", r"$\rho_{\mathrm{Hyb}}$"]
    note = (r"Latency per frame, batch size 1, mean $\pm$ SD. $\rho = L_{\mathrm{gate}}/L_{\mathrm{HED,CPU}}$; in the GPU column, "
            r"$\rho_{\mathrm{Hyb}}$ is the GPU gate relative to HED on the CPU (hybrid setting). 1 multiply--add = 2 FLOPs. "
            + (r"$^{a}$HED on the GPU through the PyTorch port of the Caffe model (Section~\ref{sec:setup}). " if len(hed_g) else "")
            + r"$^{b}$Hand-crafted features computed on the CPU.")
    T("tab_complexity", _tabular("l" + "r" * (len(head) - 1), head, rows, note))

    # ---------------- design space: best head per gate ----------------
    bp = bpg.copy()
    bp["S_m"] = bp.S_CPU_r100.map(lambda x: _pm(x)[0])
    rows = []
    for _, r in bp.sort_values("S_m", ascending=False).iterrows():
        rr = [_cfg(r.gate, r.scale), STRAT[r["best strategy"]], _fmt_pm(r.pr_auc, 1, 3), _fmt_pm(r.get("pr_auc_fold", np.nan), 1, 3),
              _fmt_pm(r.fpr_at_r100, 100, 1), _fmt_pm(r.recall_r100, 100, 1), _fmt_pm(r.fpr_r100, 100, 1),
              _fmt_pm(r.q_r100, 100, 1), _fmt_pm(r.S_CPU_r100, 1, 1)]
        if gpu and "S_GPU_r100" in r:
            rr.append(_fmt_pm(r.S_GPU_r100, 1, 1))
        if (r.gate, float(r.scale), r["best strategy"]) == (best[0], float(best[1]), best[2]):
            rr = [r"\textbf{" + x + "}" for x in rr]; rr[0] += r"$^{\ast}$"
        elif gbest and (r.gate, float(r.scale), r["best strategy"]) == (gbest[0], float(gbest[1]), gbest[2]):
            rr[0] += r"$^{\dagger}$"
        rows.append(rr)
    head = ["Gate", "Head", "PR-AUC (pooled)", "PR-AUC (per fold)", r"FPR@R$_{100}$ (\%)", r"Recall (\%)", r"FPR (\%)", r"$q$ (\%)", r"$S_{\mathrm{CPU}}$"]
    if gpu:
        head.append(r"$S_{\mathrm{Hyb}}$")
    note = (r"Mean $\pm$ SD over " + str(M["NRepeats"]) + r" repetitions of recording-grouped " + str(M["NFolds"]) +
            r"-fold cross-validation (per-fold PR-AUC: over all " + str(M["NFoldsTotal"]) + r" test folds). "
            r"FPR@R$_{100}$: threshold-free share of background frames scoring at least as high as the lowest joint. "
            r"Recall, FPR and $q$ at the threshold selected inside each training fold for a 100\% recall target. "
            r"$S = 1/(\rho + \bar q)$ with $\bar q$ the mean over repetitions; SD of $S$ by the delta method. "
            r"$^{\ast}$Selected for the CPU (highest $S_{\mathrm{CPU}}$ with 100\% recall in every repetition); "
            r"$^{\dagger}$selected for the GPU in the same way. Selection on these same results is optimistic; Table~\ref{tab:nested} gives the nested estimate.")
    T("tab_design_space", _tabular("ll" + "r" * (len(head) - 2), head, rows, note))

    # ---------------- imbalance remedies on the best gate ----------------
    rows = []
    for _, r in strat.iterrows():
        rr = [STRAT[r.strategy], _fmt_pm(r.pr_auc, 1, 3), _fmt_pm(r.recall_r100, 100, 1), _fmt_pm(r.fpr_r100, 100, 1),
              _fmt_pm(r.q_r100, 100, 1), _fmt_pm(r.S_CPU_r100, 1, 1),
              _fmt_pm(r.get("recall_t050", np.nan), 100, 1), _fmt_pm(r.get("recall_t035", np.nan), 100, 1)]
        if r.strategy == best[2]:
            rr = [r"\textbf{" + x + "}" for x in rr]
        rows.append(rr)
    head = ["Head", "PR-AUC", r"Recall (\%)", r"FPR (\%)", r"$q$ (\%)", r"$S_{\mathrm{CPU}}$", r"Recall, $\tau$=0.5 (\%)", r"Recall, $\tau$=0.35 (\%)"]
    fx = [100 * _pm(strat[c].iloc[i])[0] for c in ("fpr_t050", "fpr_t035") if c in strat for i in range(len(strat))
          if not np.isnan(_pm(strat[c].iloc[i])[0])]
    fx_txt = ("n/a" if not fx else f"{max(fx):.1f}\\%" if f"{min(fx):.1f}" == f"{max(fx):.1f}" else f"{min(fx):.1f}--{max(fx):.1f}\\%")
    note = (r"Gate: " + M["BestName"] + r". Mean $\pm$ SD over repetitions. Columns 3--6: threshold selected for a 100\% recall target. "
            r"Last two columns: fixed thresholds (FPR " + fx_txt + r"). One-class heads output distances, so fixed thresholds do not apply.")
    T("tab_strategies", _tabular("l" + "r" * 7, head, rows, note))
    sb = strat.set_index("strategy")
    M["BestRecallHalf"] = _f(_pm(sb.loc[best[2], "recall_t050"])[0], 1, 100)
    M["BestMissHalf"] = _f(1 - _pm(sb.loc[best[2], "recall_t050"])[0], 1, 100)
    M["BestRecallLegacy"] = _f(_pm(sb.loc[best[2], "recall_t035"])[0], 1, 100)
    M["BestFPRHalf"] = _f(_pm(sb.loc[best[2], "fpr_t050"])[0], 1, 100)
    sup = sb.loc[[s for s in sb.index if not s.startswith("OneClass")]]
    Ssup = sup.S_CPU_r100.map(lambda x: _pm(x)[0])
    M["StratSMin"], M["StratSMax"] = f"{Ssup.min():.1f}", f"{Ssup.max():.1f}"
    Rsup = sup.recall_r100.map(lambda x: _pm(x)[0])
    M["StratRecallMin"] = f"{100 * Rsup.min():.1f}"
    M["NStratFullRecall"] = int((Rsup >= 0.9999).sum()); M["NStratSup"] = len(Rsup)

    # ---------------- scalar results from the full grid ----------------
    cvm["key"] = list(zip(cvm.gate, cvm.scale.astype(float), cvm.strategy)); cg = cvm.set_index("key")
    cvs["key"] = list(zip(cvs.gate, cvs.scale.astype(float), cvs.strategy)); cs = cvs.set_index("key")
    k3 = lambda c: (c[0], float(c[1]), c[2])
    for role, c in ROLE.items():
        if c is None:
            continue
        tg, r, sd = TAG[role], cg.loc[[k3(c)]].iloc[0], cs.loc[[k3(c)]].iloc[0]
        M.update({f"{tg}PRAUC": f"{r.pr_auc:.3f}", f"{tg}PRAUCFold": f"{r.get('pr_auc_fold', np.nan):.3f}",
                  f"{tg}PRAUCFoldSD": f"{sd.get('pr_auc_fold', np.nan):.3f}", f"{tg}FPRRfull": f"{100 * r.fpr_at_r100:.1f}",
                  f"{tg}Recall": f"{100 * r.recall_r100:.1f}", f"{tg}FPR": f"{100 * r.fpr_r100:.1f}", f"{tg}FPRSD": f"{100 * sd.fpr_r100:.1f}",
                  f"{tg}Q": f"{100 * r.q_r100:.1f}", f"{tg}QSD": f"{100 * sd.q_r100:.1f}", f"{tg}SCPU": f"{r.S_CPU_r100:.1f}",
                  f"{tg}SCPUSD": f"{sd.S_CPU_r100:.1f}", f"{tg}RecallSD": f"{100 * sd.recall_r100:.1f}",
                  f"{tg}MissedTotal": str(int(round(max(0.0, 1 - r.recall_r100) * M["NJoints"] * M["NRepeats"])))})
        if gpu:
            M[f"{tg}SGPU"] = f"{r.S_GPU_r100:.1f}"
    M["GBestSame"] = "yes" if gbest and k3(gbest) == k3(best) else "no"
    if acc is not None:
        ra = cg.loc[[k3(acc)]].iloc[0]
        M.update({"AccName": _cfg(acc[0], acc[1]), "AccHead": STRAT[acc[2]], "AccPRAUC": f"{ra.pr_auc:.3f}", "AccRecall": f"{100 * ra.recall_r100:.1f}",
                  "AccFPR": f"{100 * ra.fpr_r100:.1f}", "AccQ": f"{100 * ra.q_r100:.1f}", "AccSCPU": f"{ra.S_CPU_r100:.1f}",
                  "AccSame": "yes" if k3(acc) == k3(best) else "no"})
        if gpu:
            M["AccSGPU"] = f"{ra.S_GPU_r100:.1f}"; M["AccGPUms"] = f"{lrow(acc[0], acc[1], 'GPU').mean_ms:.1f}"
            M["AccCPUms"] = f"{lrow(acc[0], acc[1], 'CPU').mean_ms:.1f}"
    strict = cvm[(cvm.recall_r100 >= 0.9999)]
    supc = cvm[~cvm.strategy.str.startswith("OneClass")]
    fast = supc.sort_values("S_CPU_r100", ascending=False).iloc[0]
    M.update({"FastName": _cfg(fast.gate, fast.scale), "FastHead": STRAT[fast.strategy], "FastSCPU": f"{fast.S_CPU_r100:.1f}",
              "FastRecall": f"{100 * fast.recall_r100:.1f}", "FastFPR": f"{100 * fast.fpr_r100:.1f}",
              "FastIsBest": "yes" if k3((fast.gate, fast.scale, fast.strategy)) == k3(best) else "no"})
    M["NStrict"] = len(strict[~strict.strategy.str.startswith("OneClass")])
    rf_ = cg.loc[[k3(ref)]].iloc[0]
    M["RefRecallHalf"] = f"{100 * rf_.recall_t050:.1f}"; M["RefFPRHalf"] = f"{100 * rf_.fpr_t050:.1f}"
    M["RefRecallLegacy"] = f"{100 * rf_.recall_t035:.1f}"; M["RefFPRLegacy"] = f"{100 * rf_.fpr_t035:.1f}"
    oc = cvm[cvm.strategy.str.startswith("OneClass")]
    boc = oc.sort_values("pr_auc", ascending=False).iloc[0]
    M.update({"OneClassBestName": f"{_cfg(boc.gate, boc.scale)}, {STRAT[boc.strategy]}", "OneClassBestPRAUC": f"{boc.pr_auc:.2f}",
              "OneClassBestRecall": f"{100 * boc.recall_r100:.1f}", "OneClassBestFPR": f"{100 * boc.fpr_r100:.1f}",
              "SupMinPRAUCBestGate": f"{sup.pr_auc.map(lambda x: _pm(x)[0]).min():.3f}"})
    supv = ~cvm.strategy.str.startswith("OneClass")
    mnh = cvm[(cvm.gate == "MobileNetV3Small") & (cvm.scale == 0.5) & supv]
    if len(mnh):
        M["MNSHalfPRAUC"] = f"{mnh.pr_auc.max():.2f}"
    sup_all = cvm[supv & ~cvm.gate.isin(HANDCRAFTED)]
    M["NSupFullRecall"] = int((sup_all.recall_r100 >= 0.9999).sum()); M["NSup"] = len(sup_all)
    for h in HANDCRAFTED:          # best head of each hand-crafted gate
        hs = cvm[(cvm.gate == h) & supv]
        if len(hs):
            hb = hs.sort_values(["recall_r100", "S_CPU_r100"], ascending=False).iloc[0]
            M.update({f"{h}PRAUC": f"{hb.pr_auc:.3f}", f"{h}Recall": f"{100 * hb.recall_r100:.1f}",
                      f"{h}FPR": f"{100 * hb.fpr_r100:.1f}", f"{h}SCPU": f"{hb.S_CPU_r100:.1f}", f"{h}Head": STRAT[hb.strategy]})

    # ---------------- end-to-end (three settings, all repetitions) ----------------
    e2e, e2s = rd("end_to_end.csv"), rd("end_to_end_per_seed.csv")
    rows = []
    for _, r in e2e.iterrows():
        g_, h_ = _cfg_str(r.config)
        rows.append([g_, h_, SETTING_TXT.get(r.setting, r.setting), f"{100 * r.q:.1f}", f"{r.rho:.3f}", f"{1000 * r.L_pipeline_s:.0f}",
                     f"{r.FPS:.1f}", f"{r.S_measured:.2f} $\\pm$ {r.S_measured_sd:.2f}", f"{r.S_predicted:.2f}", _err(r["model_error_%"]),
                     f"{r.p95_ms:.0f}"])
        for role in [t for t, c_ in [("Best", best_s), ("GBest", gbest_s), ("Acc", acc_s), ("Ref", ref_s)] if c_ == r.config]:
            tg = f"EtoE{role}{SETTING.get(r.setting, r.setting)}"
            M.update({f"{tg}S": f"{r.S_measured:.1f}", f"{tg}SD": f"{r.S_measured_sd:.1f}", f"{tg}Spred": f"{r.S_predicted:.1f}",
                      f"{tg}FPS": f"{r.FPS:.1f}", f"{tg}ms": f"{1000 * r.L_pipeline_s:.0f}", f"{tg}Err": _err(r["model_error_%"]),
                      f"{tg}Q": f"{100 * r.q:.1f}", f"{tg}Pnf": f"{r.p95_ms:.0f}"})
    M["EtoEMaxErr"] = f"{e2s['model_error_%'].abs().max():.0f}"
    M["EtoEMeanAbsErr"] = f"{e2s['model_error_%'].abs().mean():.1f}"
    M["NEtoERuns"] = len(e2s)
    head = ["Gate", "Head", "Setting", r"$q$ (\%)", r"$\rho$", "Latency (ms)", "FPS", r"$S$ measured", r"$S$ predicted", r"Error (\%)", "p95 (ms)"]
    note = (r"Complete pipelines over all " + str(M["NFrames"]) + r" frames in stream order (gate on every frame, HED on the frames routed by the "
            r"cross-validated decisions), mean over the " + str(M["NRepeats"]) + r" repetitions ($S$ measured: mean $\pm$ SD). "
            r"$S$ and $\rho$ are relative to un-gated HED on the same HED device. Error = (predicted $-$ measured)/measured, for the mean values. "
            r"p95: 95th percentile of the per-frame latency.")
    T("tab_end_to_end", _tabular("lll" + "r" * 8, head, rows, note))
    sq = rd("stream_queueing.csv")
    if sq is not None and len(sq):
        rows = []
        for _, r in sq.iterrows():
            g_, h_ = _cfg_str(r.config)
            rows.append([f"{g_}, {h_}", SETTING_TXT.get(r.setting, r.setting), str(int(r.camera_fps)), f"{100 * r.utilisation:.0f}",
                         "yes" if str(r.keeps_up) == "True" else "no", f"{r.p95_delay_ms:.0f}", f"{r.max_delay_ms:.0f}"])
        T("tab_stream", _tabular("llrrlrr", ["Configuration", "Setting", "Camera FPS", r"Utilization (\%)", "Keeps up", "p95 delay (ms)", "Max. delay (ms)"],
                                 rows, r"Stream replay of repetition 1 in recording order; delay = queueing + processing (Lindley recursion). "
                                       r"Utilization $>$100\% means the queue grows without bound at that frame rate."))
        b_cpu = sq[(sq.config == best_s) & (sq.setting == "CPU")]
        ok = b_cpu[b_cpu.keeps_up.astype(str) == "True"]
        M["StreamBestCPUMaxCam"] = str(int(ok.camera_fps.max())) if len(ok) else "none"

    # ---------------- robustness protocols ----------------
    rows = []
    pname = {"frame-level 5-fold": "frame-level CV", "group-level 5-fold": "recording-grouped CV", "leave-one-station-out": "leave-one-station-out"}
    for _, r in rob.iterrows():
        g_, h_ = _cfg_str(r.config)
        rows.append([f"{g_}, {h_}", pname.get(r.protocol, r.protocol), f"{r.pr_auc:.3f}", f"{100 * r.recall_r100:.1f}",
                     f"{100 * r.fpr_r100:.1f}", f"{100 * r.q_r100:.1f}", f"{r.S_CPU_r100:.1f}"])
        tg = "Best" if r.config == best_s else ("Ref" if r.config == ref_s else "GBest")
        k_ = {"frame-level 5-fold": "Frame", "group-level 5-fold": "Group", "leave-one-station-out": "Station"}[r.protocol]
        M.update({f"{tg}{k_}Recall": f"{100 * r.recall_r100:.1f}", f"{tg}{k_}FPR": f"{100 * r.fpr_r100:.1f}",
                  f"{tg}{k_}S": f"{r.S_CPU_r100:.1f}", f"{tg}{k_}PRAUC": f"{r.pr_auc:.3f}", f"{tg}{k_}Q": f"{100 * r.q_r100:.1f}",
                  f"{tg}{k_}Missed": str(int(round((1 - r.recall_r100) * M["NJoints"])))})
    head = ["Configuration", "Protocol", "PR-AUC", r"Recall (\%)", r"FPR (\%)", r"$q$ (\%)", r"$S_{\mathrm{CPU}}$"]
    T("tab_robustness", _tabular("ll" + "r" * 5, head, rows,
      r"Frame-level and recording-grouped CV: mean over " + str(M["NRepeats"]) + r" repetitions; leave-one-station-out: one run. "
      r"Recall, FPR and $q$ at the threshold selected for a 100\% recall target; $S_{\mathrm{CPU}} = 1/(\rho_{\mathrm{CPU}} + \bar q)$."))

    if mir is not None and len(mir):
        rows = []
        for _, r in mir.iterrows():
            g_, h_ = _cfg_str(r.config)
            fa = {"True": "yes", "False": "no", True: "yes", False: "no"}.get(r.flip_augmentation, str(r.flip_augmentation))
            rows.append([f"{g_}, {h_}", fa, f"{100 * r.recall_r100:.1f}", f"{100 * r.fpr_r100:.1f}",
                         f"{100 * r.recall_mirrored:.1f}", f"{100 * r.fpr_mirrored:.1f}"])
        T("tab_mirror", _tabular("llrrrr", ["Configuration", "Flip augmentation", r"Recall (\%)", r"FPR (\%)", r"Recall, mirrored (\%)", r"FPR, mirrored (\%)"],
                                 rows, r"Test folds mirrored horizontally; thresholds unchanged. Mean over repetitions."))
        mb_ = mir[mir.config == best_s]
        on = mb_[mb_.flip_augmentation.astype(str) == "True"]; off = mb_[mb_.flip_augmentation.astype(str) == "False"]
        if len(on) and len(off):
            M.update({"MirFPROn": f"{100 * on.fpr_mirrored.iloc[0]:.1f}", "MirFPROff": f"{100 * off.fpr_mirrored.iloc[0]:.1f}",
                      "MirRecallOn": f"{100 * on.recall_mirrored.iloc[0]:.1f}", "MirRecallOff": f"{100 * off.recall_mirrored.iloc[0]:.1f}",
                      "MirFPRPlainOff": f"{100 * off.fpr_r100.iloc[0]:.1f}", "MirFPRPlainOn": f"{100 * on.fpr_r100.iloc[0]:.1f}"})

    if leg is not None and len(leg):
        rows = []
        for th in ["fixed 0.35", "fixed 0.5", "validation-selected"]:
            v = leg[(leg.threshold == th) & (leg.split == "validation")].iloc[0]
            t = leg[(leg.threshold == th) & (leg.split == "test")].iloc[0]
            rows.append([th, f"{v.tau:.3f}", f"{100 * v.recall:.0f}", f"{int(v.FP)}", f"{100 * v.q:.1f}", f"{100 * t.recall:.0f}", f"{int(t.FP)}"])
            k_ = {"fixed 0.35": "Fixed", "fixed 0.5": "Half", "validation-selected": "Star"}[th]
            M.update({f"Legacy{k_}Tau": f"{v.tau:.3f}", f"Legacy{k_}ValFP": int(v.FP), f"Legacy{k_}ValQ": f"{100 * v.q:.1f}", f"Legacy{k_}TestFP": int(t.FP)})
        T("tab_legacy", _tabular("lrrrrrr", ["Threshold", r"$\tau$", r"Val. recall (\%)", "Val. FP", r"Val. $q$ (\%)", r"Test recall (\%)", "Test FP"], rows,
                                 r"Fine-tuned ResNet50 gate of the original design, original split (validation: 8 joints, 77 background; test: 1 joint, 10 background)."))

    if pst is not None and len(pst):
        rows = [[str(r.station), str(int(r.frames)), str(int(r.joints)), _f(r.get("recall_best"), 0, 100), _f(r.get("fpr_best"), 1, 100),
                 _f(r.get("recall_ref"), 0, 100), _f(r.get("fpr_ref"), 1, 100)] for _, r in pst.iterrows()]
        T("tab_stations", _tabular("lrrrrrr", ["Held-out station", "Frames", "Joints", r"Recall, best (\%)", r"FPR, best (\%)", r"Recall, ref. (\%)", r"FPR, ref. (\%)"],
                                   rows, r"Each station is held out in turn; ``--'' = not defined (no joints or no background frames)."))
        fb = pst.fpr_best.dropna()
        if len(fb):
            M["StationFPRMin"], M["StationFPRMax"] = f"{100 * fb.min():.0f}", f"{100 * fb.max():.0f}"

    if conf is not None and len(conf):
        rows = []
        for _, r in conf.iterrows():
            nm = "Station-only baseline" if r.model.startswith("station") else ", ".join(_cfg_str(r.model))
            rows.append([nm, r.subset, str(int(r.frames)), str(int(r.joints)), _f(r.pr_auc, 3), _f(r.fpr_at_r100, 1, 100),
                         _f(r.get("recall", np.nan), 1, 100), _f(r.get("fpr", np.nan), 1, 100)])
        T("tab_confound", _tabular("llrrrrrr", ["Model", "Frames evaluated", "Frames", "Joints", "PR-AUC", r"FPR@R$_{100}$ (\%)", r"Recall (\%)", r"FPR (\%)"], rows,
                                   r"Out-of-fold scores of recording-grouped cross-validation (repetition 1). Station-only baseline: score = joint rate of the frame's station in the training folds. "
                                   r"Recall and FPR at the selected threshold."))
        mixed_rows = conf[conf.subset == "stations with both classes"]
        if len(mixed_rows):
            M["MixedFrames"], M["MixedJoints"] = int(mixed_rows.frames.iloc[0]), int(mixed_rows.joints.iloc[0])
            bm = mixed_rows[mixed_rows.model == best_s]
            if len(bm):
                M.update({"BestMixedPRAUC": _f(bm.pr_auc.iloc[0], 3), "BestMixedFPRR": _f(bm.fpr_at_r100.iloc[0], 1, 100),
                          "BestMixedRecall": _f(bm.recall.iloc[0], 1, 100), "BestMixedFPR": _f(bm.fpr.iloc[0], 1, 100)})
            so = mixed_rows[mixed_rows.model.str.startswith("station")]
            if len(so):
                M.update({"StOnlyMixedPRAUC": _f(so.pr_auc.iloc[0], 3), "StOnlyMixedFPRR": _f(so.fpr_at_r100.iloc[0], 1, 100)})

    # ---------------- uncertainty: recall bounds, bootstrap CIs ----------------
    rb, bt = rd("recall_bounds.csv"), rd("bootstrap_ci.csv")
    if rb is not None and len(rb):
        for role, tg in TAG.items():
            d = rb[rb.role == role]
            if len(d):
                w = d.sort_values("detected").iloc[0]          # worst repetition
                M.update({f"{tg}Detected": int(w.detected), f"{tg}CPLow": f"{100 * w.cp_low:.1f}", f"{tg}CPHigh": f"{100 * w.cp_high:.1f}",
                          f"{tg}RecallLB": f"{100 * w.one_sided_95_lower:.1f}"})
    if bt is not None and len(bt):
        rows, seen = [], set()
        for role, tg in TAG.items():
            d = bt[bt.role == role].set_index("metric")
            if not len(d):
                continue
            first = d.config.iloc[0] not in seen; seen.add(d.config.iloc[0])
            g_, h_ = _cfg_str(d.config.iloc[0])
            ci = lambda m, sc, nd: (f"{_f(d.loc[m, 'point'], nd, sc)} [{_f(d.loc[m, 'ci_low'], nd, sc)}, {_f(d.loc[m, 'ci_high'], nd, sc)}]"
                                    if m in d.index else "--")
            rr = [f"{g_}, {h_}", ci("pr_auc", 1, 3), ci("recall", 100, 1), ci("fpr", 100, 1), ci("q", 100, 1), ci("S_CPU", 1, 1)]
            if gpu:
                rr.append(ci("S_GPU", 1, 1))
            w = rb[rb.role == role].sort_values("detected").iloc[0] if rb is not None and len(rb[rb.role == role]) else None
            rr.append(f"{int(w.detected)}/{int(w.joints)}: [{100 * w.cp_low:.1f}, {100 * w.cp_high:.1f}]" if w is not None else "--")
            if first:
                rows.append(rr)
            for m, sc, nd in [("pr_auc", 1, 3), ("recall", 100, 1), ("fpr", 100, 1), ("q", 100, 1), ("S_CPU", 1, 1), ("S_GPU", 1, 1)]:
                if m in d.index:
                    nm = {"pr_auc": "PRAUC", "recall": "Recall", "fpr": "FPR", "q": "Q", "S_CPU": "SCPU", "S_GPU": "SGPU"}[m]
                    M[f"{tg}{nm}Lo"], M[f"{tg}{nm}Hi"] = _f(d.loc[m, "ci_low"], nd, sc), _f(d.loc[m, "ci_high"], nd, sc)
        head = ["Configuration", "PR-AUC", r"Recall (\%)", r"FPR (\%)", r"$q$ (\%)", r"$S_{\mathrm{CPU}}$"] + ([r"$S_{\mathrm{Hyb}}$"] if gpu else []) + \
               [r"Joints found: 95\% CI (\%)"]
        T("tab_uncertainty", _tabular("l" + "r" * (len(head) - 1), head, rows,
          r"Point estimate [95\% CI] from " + str(M["NBoot"]) + r" bootstrap resamples of recordings (frames of a recording kept together), "
          r"averaged over repetitions. Last column: joints detected in the worst repetition and the exact Clopper--Pearson 95\% interval for recall; "
          r"the repetitions re-use the same " + str(M["NJoints"]) + r" joints, so the interval uses $n = " + str(M["NJoints"]) + r"$."))

    # ---------------- significance tests ----------------
    tt = rd("significance_tests.csv")
    if tt is not None and len(tt):
        rows = []
        mname = {"pr_auc": "PR-AUC", "fpr_at_r100": r"FPR@R$_{100}$", "q": r"$q$"}
        for _, r in tt.iterrows():
            a_, b_ = ", ".join(_cfg_str(r.A)), ", ".join(_cfg_str(r.B))
            if r.family.startswith("heads"):
                a_, b_ = _cfg_str(r.A)[1], _cfg_str(r.B)[1]
            sc = 1 if r.metric == "pr_auc" else 100
            rows.append([r.family.replace("heads on best backbone", "heads on " + M["BestName"]), a_, b_, mname.get(r.metric, r.metric),
                         _f(r.mean_diff, 3 if sc == 1 else 1, sc), _f(r.t, 2), _f(r.p, 3), _f(r.p_holm, 3)])
        T("tab_tests", _tabular("lllrrrrr", ["Family", "A", "B", "Metric", r"Mean diff. (A$-$B)", r"$t$", r"$p$", r"$p$ (Holm)"], rows,
          r"Corrected resampled $t$-test \citep{nadeau2003inference} on the per-fold differences over " + str(M["NFoldsTotal"]) +
          r" test folds (" + str(M["NRepeats"]) + r" repetitions $\times$ " + str(M["NFolds"]) + r" folds); Holm correction within each family. "
          r"FPR@R$_{100}$ and $q$ differences in percentage points."))
        h_ = tt[tt.family.str.startswith("heads")]
        M["NHeadTests"] = len(h_); M["NHeadTestsSig"] = int((h_.p_holm < 0.05).sum())
        M["HeadTestMinP"] = _f(h_.p_holm.min(), 2) if len(h_) else NA
        s_ = tt[tt.family.str.startswith("selected")]
        M["NSelTests"] = len(s_); M["NSelTestsSig"] = int((s_.p_holm < 0.05).sum())
        for _, r in s_.iterrows():
            pair = {frozenset([best_s, ref_s]): "BestRef", frozenset([gbest_s, ref_s]): "GBestRef", frozenset([best_s, gbest_s]): "BestGBest"}.get(frozenset([r.A, r.B]))
            if pair:
                M[f"P{pair}{ {'pr_auc': 'PRAUC', 'fpr_at_r100': 'FPRR', 'q': 'Q'}[r.metric]}"] = _f(r.p_holm, 3)
    if sel.get("friedman"):
        M["FriedmanChi"], M["FriedmanP"] = _f(sel["friedman"].get("chi2"), 2), _f(sel["friedman"].get("p"), 3)
        M["FriedmanIdentical"] = "yes" if sel["friedman"].get("identical") else "no"

    # ---------------- nested selection ----------------
    nv, nf, ns = rd("nested_vs_apparent.csv"), rd("nested_selection_frequency.csv"), rd("nested_selection_per_seed.csv")
    if nv is not None and len(nv):
        rows = []
        for _, r in nv.iterrows():
            rows.append([{"CPU": "CPU rule", "GPU": "GPU rule"}.get(r.rule, r.rule), r.estimate, _f(r.recall, 1, 100), _f(r.fpr, 1, 100),
                         _f(r.q, 1, 100), _f(r.S, 2)])
            tg = ("Nested" if r.estimate.startswith("nested") else "Apparent") + {"CPU": "CPU", "GPU": "GPU"}.get(r.rule, r.rule)
            M.update({f"{tg}Recall": _f(r.recall, 1, 100), f"{tg}FPR": _f(r.fpr, 1, 100), f"{tg}Q": _f(r.q, 1, 100), f"{tg}S": _f(r.S, 1)})
        if ns is not None:
            for rule, g in ns.groupby("rule"):
                tg = "Nested" + rule
                M.update({f"{tg}SSD": _f(g.S.std(), 1), f"{tg}RecallMin": _f(g.recall.min(), 1, 100),
                          f"{tg}MissedTotal": int(g.missed.sum()), f"{tg}MissedMax": int(g.missed.max())})
        if nf is not None and len(nf):
            for rule, g in nf.groupby("rule"):
                top = g.sort_values("times_selected", ascending=False).iloc[0]
                M[f"Nested{rule}Top"] = f"{_cfg(top.gate, top.scale)} with {STRAT[top.strategy]}"
                M[f"Nested{rule}TopN"] = int(top.times_selected); M[f"Nested{rule}NDistinct"] = len(g)
                M[f"Nested{rule}NFolds"] = int(g.times_selected.sum())
        T("tab_nested", _tabular("llrrrr", ["Selection rule", "Estimate", r"Recall (\%)", r"FPR (\%)", r"$q$ (\%)", r"$S$"], rows,
          r"Apparent: the configuration selected on the cross-validation results of Table~\ref{tab:design}, evaluated on the same results. "
          r"Nested: the complete procedure (candidate assessment by inner recording-grouped CV, selection, threshold selection) is run inside each outer "
          r"training fold and evaluated on the outer test fold; mean over " + str(M["NNestedSeeds"]) + r" repetitions \citep{cawley2010overfitting}. "
          r"CPU rule: $S$ on the CPU; GPU rule: $S$ in the hybrid setting."))
        if nf is not None and len(nf):
            rows = [[{"CPU": "CPU rule", "GPU": "GPU rule"}.get(r.rule, r.rule), _cfg(r.gate, r.scale), STRAT[r.strategy], str(int(r.times_selected))]
                    for _, r in nf.sort_values(["rule", "times_selected"], ascending=[True, False]).iterrows()]
            T("tab_nested_freq", _tabular("lllr", ["Rule", "Gate", "Head", "Times selected"], rows,
              r"Configurations selected in the " + str(int(nf.groupby("rule").times_selected.sum().iloc[0])) + r" outer folds of the nested evaluation."))

    # ---------------- thresholds ----------------
    td, tj, tr_ = rd("threshold_distribution.csv"), rd("threshold_jackknife.csv"), rd("threshold_rules.csv")
    if td is not None and len(td):
        for _, r in td.iterrows():
            tg = "Best" if r.config == best_s else ("GBest" if r.config == gbest_s else "Ref")
            M.update({f"Tau{tg}Mean": _f(r["mean"], 3), f"Tau{tg}SD": _f(r["std"], 3), f"Tau{tg}Min": _f(r["min"], 3), f"Tau{tg}Max": _f(r["max"], 3)})
    if tj is not None and len(tj):
        for _, r in tj.iterrows():
            tg = "Best" if r.role == "best" else "GBest"
            M.update({f"Jack{tg}N": int(r.n_jackknife), f"Jack{tg}Med": _f(r.median_abs_tau_change, 3), f"Jack{tg}Max": _f(r.max_abs_tau_change, 3),
                      f"Jack{tg}MissedFull": int(r.missed_with_full_tau), f"Jack{tg}MissedWorst": int(r.missed_worst_jackknife_tau),
                      f"Jack{tg}Share": _f(r.share_jack_tau_missing_any, 1, 100)})
    if tr_ is not None and len(tr_):
        rows = []
        rkey = {"inner-CV + margin (ours)": "Ours", "inner-CV, no margin": "NoMargin", "split-conformal": "Conf",
                "inner-CV, r*=95%": "Rninetyfive", "fixed 0.5": "Fixed"}
        for _, r in tr_.iterrows():
            g_, h_ = _cfg_str(r.config)
            rr = [f"{g_}, {h_}", r.rule.replace("r*=95%", r"$r^\ast$=95\%"), f"{100 * r.recall_mean:.1f}", f"{100 * r.recall_min:.1f}",
                  f"{r.missed_max:.0f}", f"{100 * r.fpr_mean:.1f}", f"{100 * r.q_mean:.1f}", f"{r.S_CPU:.1f}"]
            if gpu and "S_GPU" in r:
                rr.append(f"{r.S_GPU:.1f}")
            rows.append(rr)
            tg = ("Best" if r.role == "best" else "GBest" if r.role == "gpu_best" else "Ref") + rkey.get(r.rule, "X")
            M.update({f"Rule{tg}Recall": _f(r.recall_mean, 1, 100), f"Rule{tg}RecallMin": _f(r.recall_min, 1, 100),
                      f"Rule{tg}MissedMax": f"{r.missed_max:.0f}", f"Rule{tg}FPR": _f(r.fpr_mean, 1, 100), f"Rule{tg}SCPU": _f(r.S_CPU, 1)})
            if gpu and "S_GPU" in r:
                M[f"Rule{tg}SGPU"] = _f(r.S_GPU, 1)
            if r.rule == "split-conformal" and not np.isnan(r.conformal_bound_mean):
                M[f"Rule{tg}Bound"] = _f(r.conformal_bound_mean, 1, 100)
        head = ["Configuration", "Threshold rule", r"Recall (\%)", r"Min. recall (\%)", "Max. missed", r"FPR (\%)", r"$q$ (\%)", r"$S_{\mathrm{CPU}}$"] + \
               ([r"$S_{\mathrm{Hyb}}$"] if gpu else [])
        T("tab_threshold_rules", _tabular("ll" + "r" * (len(head) - 2), head, rows,
          r"Same folds and heads; mean over repetitions, minimum recall and maximum number of missed joints over repetitions. "
          r"Split-conformal: head fitted on two thirds of the training recordings, $\tau$ = lowest joint score on the remaining third."))

    # ---------------- station calibration ----------------
    sc_ = rd("station_calibration.csv")
    if sc_ is not None and len(sc_):
        rows = []
        vkey = {"A: source threshold": "A", "B: threshold calibrated on station": "B", "C: head + threshold updated": "C"}
        for (cf, role), g in sc_.groupby(["config", "role"], sort=False):
            g_, h_ = _cfg_str(cf)
            for stn in [s for s in g.station.unique() if s != "all"] + ["all"]:
                d = g[g.station == stn].set_index("variant")
                if not len(d):
                    continue
                rr = [f"{g_}, {h_}" if stn == [s for s in g.station.unique() if s != "all"][0] else "", "all held-out stations" if stn == "all" else stn,
                      str(int(d.test_joints.iloc[0]))]
                for v in vkey:
                    rr += [_f(d.loc[v, "recall"], 0, 100) if v in d.index else "--", _f(d.loc[v, "fpr"], 1, 100) if v in d.index else "--"]
                rows.append(rr)
                if stn == "all":
                    for v, k_ in vkey.items():
                        if v in d.index:
                            tg = ("Best" if role == "best" else "GBest" if role == "gpu_best" else "Ref") + "Calib" + k_
                            M[f"{tg}Recall"], M[f"{tg}FPR"] = _f(d.loc[v, "recall"], 1, 100), _f(d.loc[v, "fpr"], 1, 100)
        T("tab_calibration", _tabular("llr" + "r" * 6, ["Configuration", "Held-out station", "Test joints", r"Recall A (\%)", r"FPR A (\%)",
                                                        r"Recall B (\%)", r"FPR B (\%)", r"Recall C (\%)", r"FPR C (\%)"], rows,
          r"Gate trained on the other stations; the held-out station's recordings are split " + str(M["NCalibSplits"]) +
          r" times at random into calibration and test recordings, and results are pooled over the test recordings of all splits. "
          r"A: source threshold; B: threshold lowered to the lowest calibration-joint score; C: head and threshold re-fitted with the calibration frames. "
          r"FPR only where the station has background frames."))

    # ---------------- perturbations ----------------
    pr, la = rd("perturbation_robustness.csv"), rd("station_transfer_augmentation.csv")
    if pr is not None and len(pr):
        PN = {"none": "none (clean)", "mirror": "mirror", "brightness-30": r"brightness $-$30\%", "brightness+30": r"brightness +30\%",
              "gamma0.6": r"gamma 0.6", "gamma1.6": r"gamma 1.6", "contrast0.6": r"contrast $\times$0.6", "noise10": r"noise $\sigma$=10",
              "noise20": r"noise $\sigma$=20", "glare": "simulated glare", "motion_blur": "motion blur (9 px)", "defocus": r"defocus ($\sigma$=2)",
              "rotate5": r"rotation 5$^\circ$", "scale_shift": r"scale 1.1 + shift 8 px", "jpeg30": "JPEG quality 30"}
        for cf in pr.config.unique():
            d = pr[pr.config == cf]
            rows = []
            for p_ in d.perturbation.unique():
                c_ = d[(d.perturbation == p_) & (d.training == "clean")]; a_ = d[(d.perturbation == p_) & (d.training == "augmented")]
                rows.append([PN.get(p_, p_), _f(c_.recall.iloc[0], 1, 100) if len(c_) else "--", _f(c_.fpr.iloc[0], 1, 100) if len(c_) else "--",
                             _f(a_.recall.iloc[0], 1, 100) if len(a_) else "--", _f(a_.fpr.iloc[0], 1, 100) if len(a_) else "--"])
            tg = "Best" if cf == best_s else ("GBest" if cf == gbest_s else "Ref")
            g_, h_ = _cfg_str(cf)
            T(f"tab_perturb_{tg.lower()}", _tabular("lrrrr", ["Test perturbation", r"Recall, clean training (\%)", r"FPR, clean training (\%)",
                                                              r"Recall, augmented training (\%)", r"FPR, augmented training (\%)"], rows,
              f"Gate: {g_}, {h_}. " + r"Thresholds selected on clean frames for a 100\% recall target; mean over " + str(M["NRepeats"]) +
              r" repetitions of recording-grouped CV. Augmented training adds " + str(M["NAugCopies"]) +
              r" random photometric/geometric copies of each training frame."))
            dp = d[~d.perturbation.isin(["none"])]
            for train, k_ in [("clean", "Clean"), ("augmented", "Aug")]:
                e = dp[dp.training == train]
                if len(e):
                    w = e.sort_values("recall").iloc[0]
                    M.update({f"Pert{tg}{k_}WorstRecall": _f(w.recall, 1, 100), f"Pert{tg}{k_}WorstName": PN.get(w.perturbation, w.perturbation),
                              f"Pert{tg}{k_}MeanRecall": _f(e.recall.mean(), 1, 100), f"Pert{tg}{k_}MeanFPR": _f(e.fpr.mean(), 1, 100),
                              f"Pert{tg}{k_}MaxFPR": _f(e.fpr.max(), 1, 100), f"Pert{tg}{k_}NFull": int((e.recall >= 0.9999).sum()),
                              f"Pert{tg}{k_}N": len(e)})
                    gl = e[e.perturbation == "glare"]
                    if len(gl):
                        M[f"Pert{tg}{k_}GlareRecall"], M[f"Pert{tg}{k_}GlareFPR"] = _f(gl.recall.iloc[0], 1, 100), _f(gl.fpr.iloc[0], 1, 100)
                cl = d[(d.perturbation == "none") & (d.training == train)]
                if len(cl):
                    M[f"Pert{tg}{k_}CleanRecall"], M[f"Pert{tg}{k_}CleanFPR"] = _f(cl.recall.iloc[0], 1, 100), _f(cl.fpr.iloc[0], 1, 100)
    if la is not None and len(la):
        rows = []
        for _, r in la.iterrows():
            g_, h_ = _cfg_str(r.config)
            rows.append([f"{g_}, {h_}", r.training, _f(r.recall, 1, 100), str(int(r.missed)), _f(r.fpr, 1, 100), _f(r.q, 1, 100)])
            tg = "Best" if r.config == best_s else ("GBest" if r.config == gbest_s else "Ref")
            M.update({f"Logo{tg}{'Aug' if r.training == 'augmented' else 'Clean'}Recall": _f(r.recall, 1, 100),
                      f"Logo{tg}{'Aug' if r.training == 'augmented' else 'Clean'}FPR": _f(r.fpr, 1, 100),
                      f"Logo{tg}{'Aug' if r.training == 'augmented' else 'Clean'}Missed": int(r.missed)})
        T("tab_logo_aug", _tabular("llrrrr", ["Configuration", "Training", r"Recall (\%)", "Missed", r"FPR (\%)", r"$q$ (\%)"], rows,
                                   r"Leave-one-station-out, threshold selected inside the training stations."))

    # ---------------- baselines ----------------
    bl, ft = rd("baselines.csv"), rd("finetuned_gate.csv")
    if bl is not None and len(bl):
        rows = []
        if not cfg.get("FINETUNE"):
            bl = bl[~bl.gate.astype(str).str.contains("fine-tuned")]
        bl = bl.drop_duplicates(subset=["gate", "head"])
        for _, r in bl.iterrows():
            gname = r.gate
            if "@" in str(gname):
                g0, rest = gname.split("@"); s0 = rest.split(" ")[0]; gname = _cfg(g0, s0) + " (frozen)"
            elif gname in GATE:
                gname = GATE[gname]
            elif gname.startswith("MobileNetV3Small fine-tuned"):
                gname = "MobileNetV3-S (1$\\times$), fine-tuned"
            rr = [gname, STRAT.get(r["head"], r["head"]), _f(r.get("pr_auc", np.nan), 3), _f(r.recall, 1, 100), _f(r.fpr, 1, 100),
                  _f(r.q, 1, 100), _f(r.S_CPU, 1)]
            if gpu and "S_GPU" in r:
                rr.append(_f(r.S_GPU, 1))
            rows.append(rr)
        head = ["Gate", "Head", "PR-AUC", r"Recall (\%)", r"FPR (\%)", r"$q$ (\%)", r"$S_{\mathrm{CPU}}$"] + ([r"$S_{\mathrm{Hyb}}$"] if gpu else [])
        T("tab_baselines", _tabular("ll" + "r" * (len(head) - 2), head, rows,
          r"All at the threshold selected for a 100\% recall target, recording-grouped CV. Random or periodic sampling finds on average the same share of joints as of frames, so full recall requires $q = 1$. "
          r"Hand-crafted gates: best head by recall, then $S_{\mathrm{CPU}}$." + ((r" Fine-tuned gate: " + str(len(cfg.get("FT_SEEDS", [0]))) + r" repetition(s).") if cfg.get("FINETUNE") else "")))
    if ft is not None and len(ft) and cfg.get("FINETUNE"):
        q_ = ft.q.mean(); rh = lrow("MobileNetV3Small", 1.0, "CPU")
        M.update({"FTRecall": _f(ft.recall.mean(), 1, 100), "FTMissed": int(ft.missed.sum()), "FTFPR": _f(ft.fpr.mean(), 1, 100),
                  "FTPRAUC": _f(ft.pr_auc.mean(), 3), "FTQ": _f(q_, 1, 100), "FTSCPU": _f(1 / (rh.rho + q_), 1) if rh is not None else NA})

    # ---------------- edge fidelity ----------------
    ef = rd("edge_fidelity.csv")
    if ef is not None and len(ef):
        rows = []
        for _, r in ef.iterrows():
            nm = r.method
            if nm.startswith("Gated HED"):
                nm = "Gated HED (" + M["BestName"] + ", " + M["BestHead"] + ")"
            rows.append([nm.replace("%", r"\%"), _f(r.ods_F, 3), _f(r.precision, 3), _f(r.recall, 3), _f(r.latency_ms, 1), _f(r.S_vs_HED, 1)])
            k_ = {"HED at 75% input size": "HEDTQ", "HED at 50% input size": "HEDHalf", "Canny": "Canny", "Sobel magnitude": "Sobel"}.get(r.method)
            if k_:
                M.update({f"Edge{k_}F": _f(r.ods_F, 2), f"Edge{k_}S": _f(r.S_vs_HED, 1), f"Edge{k_}ms": _f(r.latency_ms, 1)})
            if r.method.startswith("Gated"):
                M["EdgeGatedF"] = _f(r.ods_F, 2)
        T("tab_edge", _tabular("lrrrrr", ["Method", r"F (ODS)", "Precision", "Recall", "Latency (ms)", r"Speed-up vs.\ HED"], rows,
          r"Agreement with full-resolution HED (reference: HED $\geq$ 0.5, thinned) on the " + str(M["NJoints"]) + r" joint frames; boundary matching "
          r"tolerance " + str(M["EdgeTol"]) + r" px; best global threshold per method. Gated HED: output identical to HED on routed frames, "
          r"so its F equals the share of joints routed; its latency and speed-up are per frame of the stream (CPU setting). "
          r"This is agreement with HED, not accuracy against human annotation."))
    if sel.get("gating_output_identical_max_abs_diff") is not None:
        M["GatingIdenticalDiff"] = _f(sel["gating_output_identical_max_abs_diff"], 1)

    # ---------------- prevalence ----------------
    pv = rd("prevalence_sensitivity.csv")
    if pv is not None and len(pv):
        rows = []
        pis = sorted(pv.prevalence.unique())
        for (cf, setting), g in pv.groupby(["config", "setting"], sort=False):
            g_, h_ = _cfg_str(cf)
            rows.append([f"{g_}, {h_}", SETTING_TXT.get(setting, setting)] + [_f(g[g.prevalence == p_].S.iloc[0], 1) for p_ in pis])
            tg = ("Best" if cf == best_s else "GBest" if cf == gbest_s else "Ref") + SETTING.get(setting, setting)
            M[f"Prev{tg}Low"], M[f"Prev{tg}High"] = _f(g[g.prevalence == pis[0]].S.iloc[0], 1), _f(g[g.prevalence == pis[-1]].S.iloc[0], 1)
        M["PrevLow"], M["PrevHigh"] = _f(pis[0], 0, 100), _f(pis[-1], 0, 100)
        T("tab_prevalence", _tabular("ll" + "r" * len(pis), ["Configuration", "Setting"] + [r"$\pi$ = " + _f(p_, 1, 100) + r"\%" for p_ in pis], rows,
          r"Predicted speed-up $S(\pi) = 1/(\rho + \pi r + (1-\pi) f)$ with the measured recall $r$ and FPR $f$ at the selected threshold."))

    # ---------------- hyperparameters ----------------
    hp = rd("hyperparameters.csv")
    if hp is not None and len(hp):
        esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace(" x ", r" $\times$ ").replace("+-", r"$\pm$")
        if not cfg.get("FINETUNE"):
            hp = hp[~hp.component.astype(str).str.startswith("Fine-tuned")]
        rows = [[esc(r.component), esc(r.setting), esc(r.value)] for _, r in hp.iterrows()]
        lines = [r"\begin{tabularx}{\linewidth}{@{}llX@{}}", r"\toprule", r"Component & Setting & Value \\", r"\midrule"]
        lines += [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabularx}"]
        T("tab_hyper", "\n".join(lines) + "\n")

    for k in MANUSCRIPT_MACROS:           # every macro used by the manuscript must exist
        M.setdefault(k, NA)
    for k in [k for k in M if k.endswith("Head")]:      # head names inside sentences: lower case ("weighted focal")
        v = str(M[k])
        if v.startswith(("Weighted", "One-class")):
            M[k] = v[0].lower() + v[1:]
    bad = [k for k in M if not k.isalpha()]
    assert not bad, f"macro names must contain letters only: {bad}"
    lines = [f"%% Generated from MLWA_experiments_v4 outputs | {env.get('timestamp_utc', '')} UTC",
             r"\providecommand{\tbd}[1]{\colorbox{yellow}{#1}}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    _write(os.path.join(out, "results_v4.tex"), "\n".join(lines) + "\n")
    return M


if __name__ == "__main__":
    import sys
    M = build(sys.argv[1], sys.argv[2])
    print(len(M), "macros")
