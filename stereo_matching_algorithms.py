"""
=============================================================================
  Utilizzo:
      python stereo_matching.py --dataset <cartella>  [opzioni]

  Opzioni:
      --window_size   lato della finestra SAD (default 11, dispari)
      --task2         abilita l'algoritmo migliorato (Task 2)
      --lr_threshold  soglia left-right check in pixel (default 1)
      --moravec_threshold  soglia operatore Moravec (default 100)
      --output        cartella di output (default <dataset>/output)
      --zncc          usa ZNCC invece di SAD per la baseline
=============================================================================
"""

import argparse, os, sys, time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  1.  LETTURA PARAMETRI
# ─────────────────────────────────────────────────────────────────────────────

def load_params(path: str) -> dict:
    """
    Legge i parametri dal file di testo del dataset.
    
    """
    params = {}
    tokens = open(path).read().split()
    for i in range(0, len(tokens) - 1, 2):
        try:
            params[tokens[i]] = int(tokens[i + 1])
        except ValueError:
            params[tokens[i]] = float(tokens[i + 1])
    for key in ("disp_min", "disp_max", "disp_scale", "ignore_border"):
        if key not in params:
            raise ValueError(f"Parametro '{key}' mancante in {path}")
    return params


# ─────────────────────────────────────────────────────────────────────────────
#  2.  TASK 1 – BASELINE SAD (Winner-Takes-All)
# ─────────────────────────────────────────────────────────────────────────────

def compute_disparity_sad(img_ref: np.ndarray,
                          img_other: np.ndarray,
                          disp_min: int,
                          disp_max: int,
                          win_size: int) -> np.ndarray:
    """
    Calcola la mappa di disparità con SAD e approccio WTA.

    Per ogni pixel (r, c) dell'immagine di riferimento (sinistra), la
    finestra di lato win_size centrata in (r, c) viene confrontata con
    tutte le finestre centrate in (r, c-d) per d in [disp_min, disp_max].
    La disparità assegnata è quella che minimizza il SAD.

    """
    if win_size % 2 == 0:
        win_size += 1
    kernel = np.ones((win_size, win_size), dtype=np.float32)

    img_l = img_ref.astype(np.float32)
    img_r = img_other.astype(np.float32)

    best_sad = np.full(img_l.shape, np.inf, dtype=np.float32)
    disparity = np.zeros(img_l.shape, dtype=np.float32)

    for d in range(disp_min, disp_max + 1):
        # Trasla img_r di d colonne verso destra:
        # il pixel (r, c) di imL corrisponde a (r, c-d) di imR
        if d > 0:
            shifted = np.roll(img_r, d, axis=1)
            shifted[:, :d] = 0           # bordo senza corrispondenza
        elif d < 0:
            shifted = np.roll(img_r, d, axis=1)
            shifted[:, d:] = 0
        else:
            shifted = img_r.copy()

        # SAD puntuale, poi somma tramite box-filter
        sad_pt = np.abs(img_l - shifted)
        sad_win = cv2.filter2D(sad_pt, ddepth=-1, kernel=kernel,
                               borderType=cv2.BORDER_CONSTANT)

        # Aggiorna WTA
        mask = sad_win < best_sad
        best_sad[mask] = sad_win[mask]
        disparity[mask] = d

    return disparity


def compute_disparity_zncc(img_ref: np.ndarray,
                           img_other: np.ndarray,
                           disp_min: int,
                           disp_max: int,
                           win_size: int) -> np.ndarray:
    """
    Calcola la mappa di disparità usando ZNCC (Zero-Mean Normalized
    Cross-Correlation) con approccio WTA.

    Restituisce una mappa di disparità (float32) con la disparità che
    massimizza la correlazione locale per ogni pixel.
    """
    if win_size % 2 == 0:
        win_size += 1

    img_l = img_ref.astype(np.float32)
    img_r = img_other.astype(np.float32)

    H, W = img_l.shape
    best_zncc = np.full((H, W), -np.inf, dtype=np.float32)
    disparity = np.zeros((H, W), dtype=np.float32)

    # box filter per somme locali (non normalizzato quando use normalize=False)
    kernel = (win_size, win_size)
    eps = 1e-6

    # medie locali
    mean_l = cv2.boxFilter(img_l, ddepth=-1, ksize=kernel, normalize=True)

    for d in range(disp_min, disp_max + 1):
        # shift img_r
        if d > 0:
            shifted = np.roll(img_r, d, axis=1)
            shifted[:, :d] = 0
        elif d < 0:
            shifted = np.roll(img_r, d, axis=1)
            shifted[:, d:] = 0
        else:
            shifted = img_r.copy()

        mean_s = cv2.boxFilter(shifted, ddepth=-1, ksize=kernel, normalize=True)

        # numerator: sum (I - meanI)*(J - meanJ)
        Izm = img_l - mean_l
        Jzm = shifted - mean_s
        num = cv2.boxFilter(Izm * Jzm, ddepth=-1, ksize=kernel, normalize=False)

        # denominator: sqrt( sum (I - meanI)^2 * sum (J - meanJ)^2 )
        sum_sq_I = cv2.boxFilter(Izm * Izm, ddepth=-1, ksize=kernel, normalize=False)
        sum_sq_J = cv2.boxFilter(Jzm * Jzm, ddepth=-1, ksize=kernel, normalize=False)
        denom = np.sqrt(np.maximum(sum_sq_I * sum_sq_J, eps))

        zncc = num / (denom + eps)

        # evita valori NaN/Inf
        zncc[np.isnan(zncc)] = -np.inf

        mask = zncc > best_zncc
        best_zncc[mask] = zncc[mask]
        disparity[mask] = d

    return disparity


# ─────────────────────────────────────────────────────────────────────────────
#  3.  TASK 2 – ALGORITMO MIGLIORATO (opzionale)
# ─────────────────────────────────────────────────────────────────────────────

def moravec_interest(img: np.ndarray, win_size: int = 5) -> np.ndarray:
    """
    Operatore di Moravec: misura la tessitura locale come il minimo della
    somma delle differenze al quadrato nelle 4 direzioni orizzontale,
    verticale e diagonali.
    Un valore basso indica una zona a bassa tessitura (piatta).

    """
    img_f = img.astype(np.float32)
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    kernel = np.ones((win_size, win_size), dtype=np.float32)
    min_e = np.full(img_f.shape, np.inf, dtype=np.float32)

    for dr, dc in shifts:
        shifted = np.roll(np.roll(img_f, dr, axis=0), dc, axis=1)
        diff_sq = (img_f - shifted) ** 2
        energy = cv2.filter2D(diff_sq, ddepth=-1, kernel=kernel,
                              borderType=cv2.BORDER_CONSTANT)
        min_e = np.minimum(min_e, energy)

    return min_e


def compute_disparity_improved(img_ref: np.ndarray,
                                img_other: np.ndarray,
                                disp_min: int,
                                disp_max: int,
                                win_size: int,
                                lr_threshold: float = 1.0,
                                moravec_threshold: float = 100.0
                                ) -> np.ndarray:
    """
    Algoritmo migliorato (Sezione 3.2). Combina:

    1. Rilevamento zone a bassa tessitura (operatore di Moravec):
       I pixel con interesse < moravec_threshold vengono scartati.

    2. Left-right check:
       Calcola la mappa L→R e la mappa R→L; scarta i pixel per cui la
       disparità è incoerente (differenza > lr_threshold pixel).

    Restituisce una mappa di disparità con NaN nei pixel scartati.

    """
    # ── Mappa L→R (riferimento = sinistra) ──────────────────────────────────
    disp_lr = compute_disparity_sad(img_ref, img_other,
                                    disp_min, disp_max, win_size)

    # ── Mappa R→L (riferimento = destra) ────────────────────────────────────
    # Per la mappa R→L le disparità sono negative (o invertite),
    # quindi cerchiamo la disparità nel senso opposto
    disp_rl = compute_disparity_sad(img_other, img_ref,
                                    disp_min, disp_max, win_size)

    H, W = disp_lr.shape
    disparity = disp_lr.copy()
    valid = np.ones((H, W), dtype=bool)

    # ── Left-right check ─────────────────────────────────────────────────────
    # Per ogni pixel (r, c) con disparità d_lr, controlla se
    # disp_rl[r, c - d_lr] ≈ d_lr
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    c_match = (cols - disp_lr.astype(int)).clip(0, W - 1)
    disp_rl_at_match = disp_rl[rows, c_match]
    lr_err = np.abs(disp_lr - disp_rl_at_match)
    valid &= lr_err <= lr_threshold

    # ── Moravec: scarta zone a bassa tessitura ────────────────────────────────
    interest = moravec_interest(img_ref, win_size=max(3, win_size // 2))
    valid &= interest >= moravec_threshold

    # Marca i pixel non validi con NaN
    disparity[~valid] = np.nan

    return disparity


# ─────────────────────────────────────────────────────────────────────────────
#  4.  VALUTAZIONE QUANTITATIVA
# ─────────────────────────────────────────────────────────────────────────────

def compute_eval_border(win_size: int, ignore_border: int) -> int:
    """
    Bordo effettivo per la valutazione: max(K, ignore_border).
    K = win_size // 2 (semi-finestra).
    Garantisce che l'Area contenga solo pixel per cui la finestra è
    completamente all'interno dell'immagine E rispetti il bordo del dataset.
    """
    K = win_size // 2
    return max(K, ignore_border)


def evaluate_baseline(disp_computed: np.ndarray,
                      disp_gt: np.ndarray,
                      disp_scale: int,
                      win_size: int,
                      ignore_border: int) -> dict:
    """
    Metrica E = Σ S(p, p_gt) / N_Area
    dove S(p, p_gt) = 1 se |p - p_gt| > 1, altrimenti 0.
    p e p_gt sono in unità di disparità (pixel).

    Area: regione centrale escluso il bordo b = max(K, ignore_border).
    N_Area = (W - 2b) × (H - 2b).
    Pixel con GT = 0 (occlusioni) esclusi da numeratore e denominatore.
    """
    H, W = disp_gt.shape
    b = compute_eval_border(win_size, ignore_border)

    # Region of interest
    comp_roi = disp_computed[b:H-b, b:W-b].astype(np.float32)
    gt_roi   = disp_gt[b:H-b, b:W-b].astype(np.float32)

    # Converti GT da valore grezzo a unità di disparità
    gt_units = gt_roi / disp_scale

    # Maschera pixel validi (GT > 0)
    valid = gt_roi > 0

    N_area  = int(valid.sum())          # denominatore di E (pixel validi)
    N_total = (H - 2*b) * (W - 2*b)    # pixel totali nell'Area

    if N_area == 0:
        print("[WARNING] Nessun pixel valido nell'Area di valutazione.")
        return {}

    # Errore assoluto in unità di disparità
    err = np.abs(comp_roi[valid] - gt_units[valid])

    # Funzione S: 1 se |p - p_gt| > 1 disparity unit
    E = float((err > 1.0).mean()) * 100.0   # in percentuale

    return {
        "E (bad pixel %, soglia=1)": E,
        "MAE [px]"  : float(err.mean()),
        "RMSE [px]" : float(np.sqrt((err**2).mean())),
        "N_Area (validi)": N_area,
        "N_Area (totale)": N_total,
        "bordo b": b,
    }


def evaluate_improved(disp_computed: np.ndarray,
                      disp_gt: np.ndarray,
                      disp_scale: int,
                      win_size: int,
                      ignore_border: int) -> dict:
    """
    Valutazione per l'algoritmo migliorato.

    - i pixel NaN (disparità non calcolata) vengono esclusi anche dal
      numeratore e denominatore di E;
    - viene calcolata la densità = frazione di pixel dell'Area con
      disparità valida.
      
    """
    H, W = disp_gt.shape
    b = compute_eval_border(win_size, ignore_border)

    comp_roi = disp_computed[b:H-b, b:W-b].astype(np.float32)
    gt_roi   = disp_gt[b:H-b, b:W-b].astype(np.float32)
    gt_units = gt_roi / disp_scale

    # Pixel con disparità calcolata (non NaN)
    has_disp = ~np.isnan(comp_roi)
    # Pixel con GT valido
    gt_valid = gt_roi > 0
    # Pixel valutabili: hanno disparità calcolata E GT noto
    evaluable = has_disp & gt_valid

    N_area  = int(evaluable.sum())
    N_total = (H - 2*b) * (W - 2*b)
    N_gt_valid = int(gt_valid.sum())

    # Densità = pixel con disparità calcolata / pixel totali nell'Area
    density = float(has_disp.sum()) / max(N_total, 1) * 100.0

    if N_area == 0:
        return {"Nessun pixel valutabile": True}

    err = np.abs(comp_roi[evaluable] - gt_units[evaluable])
    E   = float((err > 1.0).mean()) * 100.0

    return {
        "E (bad pixel %, soglia=1)": E,
        "Densità mappa [%]"        : density,
        "MAE [px]"                 : float(err.mean()),
        "RMSE [px]"                : float(np.sqrt((err**2).mean())),
        "N valutabili"             : N_area,
        "N GT validi"              : N_gt_valid,
        "N totale Area"            : N_total,
        "bordo b"                  : b,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5.  VISUALIZZAZIONE & SALVATAGGIO
# ─────────────────────────────────────────────────────────────────────────────

def colorize_disp(disp: np.ndarray, dmin: int, dmax: int) -> np.ndarray:
    """Mappa la disparità su una colormap TURBO (NaN → nero)."""
    norm = np.clip((disp - dmin) / max(dmax - dmin, 1), 0, 1)
    norm[np.isnan(disp)] = 0.0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def print_metrics(metrics: dict, title: str):
    print(f"\n{'='*52}")
    print(f"  {title}")
    print(f"{'='*52}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30s}: {v:8.3f}")
        elif isinstance(v, int):
            print(f"  {k:<30s}: {v:,}")
        else:
            print(f"  {k:<30s}: {v}")
    print(f"{'='*52}")


def save_results(data: dict, output_dir: str):
    """
    Salva su disco:
        disparity_baseline.png     – mappa SAD (colorata)
        disparity_improved.png     – mappa migliorata (se Task 2)
        error_map_baseline.png     – mappa errore baseline
        error_map_improved.png     – mappa errore improved (se Task 2)
        results_summary.png        – figura riepilogativa
        metrics_baseline.txt       – metriche numeriche baseline
        metrics_improved.txt       – metriche numeriche improved (se Task 2)
        disparity_baseline_raw.npy – dati grezzi float32
    """
    os.makedirs(output_dir, exist_ok=True)

    imL        = data["imL"]
    imR        = data["imR"]
    disp_base  = data["disp_base"]
    disp_gt    = data["disp_gt"]
    m_base     = data["metrics_base"]
    dmin       = data["disp_min"]
    dmax       = data["disp_max"]
    dscale     = data["disp_scale"]
    border     = data["border"]
    win_size   = data["win_size"]

    has_task2  = "disp_improved" in data

    H, W = disp_gt.shape
    b    = border

    # ── Mappe colorate ────────────────────────────────────────────────────────
    gt_units   = disp_gt.astype(np.float32) / dscale
    gt_color   = colorize_disp(gt_units, dmin, dmax)
    base_color = colorize_disp(disp_base, dmin, dmax)
    cv2.imwrite(os.path.join(output_dir, "disparity_baseline.png"), base_color)
    np.save(os.path.join(output_dir, "disparity_baseline_raw.npy"), disp_base)

    # ── Mappa errore baseline ─────────────────────────────────────────────────
    err_full = np.abs(disp_base - gt_units)
    err_full[:b, :] = 0; err_full[H-b:, :] = 0
    err_full[:, :b] = 0; err_full[:, W-b:] = 0
    err_mask = disp_gt == 0
    err_full[err_mask] = 0
    err_color = cv2.applyColorMap(
        np.clip(err_full / max(dmax - dmin, 1) * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_HOT)
    cv2.imwrite(os.path.join(output_dir, "error_map_baseline.png"), err_color)

    # ── Task 2 ────────────────────────────────────────────────────────────────
    if has_task2:
        disp_imp  = data["disp_improved"]
        imp_color = colorize_disp(disp_imp, dmin, dmax)
        cv2.imwrite(os.path.join(output_dir, "disparity_improved.png"), imp_color)

        err_imp = np.where(~np.isnan(disp_imp),
                           np.abs(np.nan_to_num(disp_imp) - gt_units), 0)
        err_imp[:b, :] = 0; err_imp[H-b:, :] = 0
        err_imp[:, :b] = 0; err_imp[:, W-b:] = 0
        err_imp[err_mask] = 0
        err_imp_color = cv2.applyColorMap(
            np.clip(err_imp / max(dmax - dmin, 1) * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(output_dir, "error_map_improved.png"), err_imp_color)

    # ── Figura riepilogativa ──────────────────────────────────────────────────
    ncols = 3
    nrows = 2 if not has_task2 else 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    algo = "ZNCC" if "disp_zncc" in data else "SAD"
    fig.suptitle(
        f"Stereo Matching {algo}  |  win={win_size}×{win_size}  |"
        f"  d=[{dmin},{dmax}]  |  disp_scale={dscale}",
        fontsize=13, fontweight="bold"
    )

    def show(ax, img_bgr, title):
        if img_bgr.ndim == 3:
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img_bgr, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    def show_metrics(ax, metrics, title):
        ax.axis("off")
        lines = [f"{title}\n"]
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.3f}")
            elif isinstance(v, int):
                lines.append(f"  {k}: {v:,}")
        ax.text(0.05, 0.95, "\n".join(lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#f0f4ff", edgecolor="#4466aa"))

    show(axes[0, 0], imL,       "Immagine sinistra (riferimento)")
    show(axes[0, 1], imR,       "Immagine destra")
    show(axes[0, 2], gt_color,  "Ground Truth (colorata)")
    show(axes[1, 0], base_color,"Disparità baseline (SAD-WTA)")
    show(axes[1, 1], err_color, "Mappa errore baseline")
    show_metrics(axes[1, 2], m_base, "Metriche baseline (Sez. 4.1)")

    if has_task2:
        show(axes[2, 0], imp_color,     "Disparità migliorata (Task 2)")
        show(axes[2, 1], err_imp_color, "Mappa errore migliorata")
        show_metrics(axes[2, 2], data["metrics_improved"], "Metriche improved (Sez. 4.2)")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "results_summary.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── File di testo con le metriche ─────────────────────────────────────────
    def write_metrics(metrics: dict, path: str, title: str, win: int, params: dict):
        with open(path, "w") as f:
            f.write("=" * 55 + "\n")
            f.write(f"  {title}\n")
            f.write("=" * 55 + "\n")
            f.write(f"  Window size   : {win} × {win} (K={win//2})\n")
            f.write(f"  Disp range    : [{params['dmin']}, {params['dmax']}]\n")
            f.write(f"  Disp scale    : {params['dscale']}\n")
            f.write(f"  Ignore border : {params['ib']} px\n")
            f.write("-" * 55 + "\n")
            for k, v in metrics.items():
                if isinstance(v, float):
                    f.write(f"  {k:<30s}: {v:.4f}\n")
                elif isinstance(v, int):
                    f.write(f"  {k:<30s}: {v:,}\n")
            f.write("=" * 55 + "\n")

    params_dict = dict(dmin=dmin, dmax=dmax, dscale=dscale,
                       ib=data["ignore_border"])
    write_metrics(m_base, os.path.join(output_dir, "metrics_baseline.txt"),
                  "TASK 1 – Baseline (Sez. 4.1)", win_size, params_dict)
    if has_task2:
        write_metrics(data["metrics_improved"],
                      os.path.join(output_dir, "metrics_improved.txt"),
                      "TASK 2 – Improved (Sez. 4.2)", win_size, params_dict)

    print(f"\n[OUTPUT] Risultati salvati in: {output_dir}")
    return fig_path


# ─────────────────────────────────────────────────────────────────────────────
#  6.  PIPELINE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(dataset_dir: str,
                 win_size: int = 11,
                 run_task2: bool = False,
                 lr_threshold: float = 1.0,
                 moravec_threshold: float = 100.0,
                 output_dir: str = None,
                 use_zncc: bool = False):

    if output_dir is None:
        output_dir = os.path.join(dataset_dir, "output")

    # ── Caricamento ───────────────────────────────────────────────────────────
    print("[1/5] Caricamento immagini e parametri...")
    for name in ("imL.png", "imR.png", "groundtruth.bmp", "params.txt"):
        p = os.path.join(dataset_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"File non trovato: {p}")

    imL_bgr   = cv2.imread(os.path.join(dataset_dir, "imL.png"))
    imR_bgr   = cv2.imread(os.path.join(dataset_dir, "imR.png"))
    imL_gray  = cv2.cvtColor(imL_bgr, cv2.COLOR_BGR2GRAY)
    imR_gray  = cv2.cvtColor(imR_bgr, cv2.COLOR_BGR2GRAY)
    disp_gt   = cv2.imread(os.path.join(dataset_dir, "groundtruth.bmp"),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32)
    params    = load_params(os.path.join(dataset_dir, "params.txt"))

    disp_min      = params["disp_min"]
    disp_max      = params["disp_max"]
    disp_scale    = params["disp_scale"]
    ignore_border = params["ignore_border"]
    K             = win_size // 2
    b             = compute_eval_border(win_size, ignore_border)

    H, W = imL_gray.shape
    print(f"     Immagine         : {W}×{H} px")
    print(f"     Disparità        : [{disp_min}, {disp_max}]  scale={disp_scale}")
    print(f"     Finestra         : {win_size}×{win_size}  K={K}")
    print(f"     Bordo valutazione: b=max({K},{ignore_border})={b} px")
    print(f"     N_Area           : ({W}-2·{b})×({H}-2·{b}) = {(W-2*b)*(H-2*b):,} px")

    # ── Task 1: Baseline ──────────────────────────────────────────────────────
    if use_zncc:
        print("\n[2/5] Task 1 – Calcolo mappa ZNCC (WTA)...")
        t0 = time.time()
        disp_base = compute_disparity_zncc(imL_gray, imR_gray,
                                          disp_min, disp_max, win_size)
        print(f"     Completato in {time.time()-t0:.1f} s")
    else:
        print("\n[2/5] Task 1 – Calcolo mappa SAD (WTA)...")
        t0 = time.time()
        disp_base = compute_disparity_sad(imL_gray, imR_gray,
                                          disp_min, disp_max, win_size)
        print(f"     Completato in {time.time()-t0:.1f} s")

    print("\n[3/5] Task 1 – Valutazione quantitativa (Sez. 4.1)...")
    metrics_base = evaluate_baseline(disp_base, disp_gt,
                                     disp_scale, win_size, ignore_border)
    print_metrics(metrics_base, "TASK 1 – BASELINE (Sezione 4.1)")

    # ── Task 2: Algoritmo migliorato (opzionale) ──────────────────────────────
    disp_imp     = None
    metrics_imp  = {}
    if run_task2:
        print("\n[4/5] Task 2 – Algoritmo migliorato (LR-check + Moravec)...")
        t0 = time.time()
        disp_imp = compute_disparity_improved(
            imL_gray, imR_gray,
            disp_min, disp_max, win_size,
            lr_threshold=lr_threshold,
            moravec_threshold=moravec_threshold
        )
        print(f"     Completato in {time.time()-t0:.1f} s")

        metrics_imp = evaluate_improved(disp_imp, disp_gt,
                                        disp_scale, win_size, ignore_border)
        print_metrics(metrics_imp, "TASK 2 – IMPROVED (Sezione 4.2)")
    else:
        print("\n[4/5] Task 2 – Saltato (usa --task2 per abilitarlo)")

    # ── Salvataggio ───────────────────────────────────────────────────────────
    print("\n[5/5] Salvataggio risultati...")
    data = {
        "imL": imL_bgr, "imR": imR_bgr,
        "disp_base": disp_base, "disp_gt": disp_gt,
        "metrics_base": metrics_base,
        "disp_min": disp_min, "disp_max": disp_max,
        "disp_scale": disp_scale, "ignore_border": ignore_border,
        "border": b, "win_size": win_size,
    }
    if run_task2 and disp_imp is not None:
        data["disp_improved"]    = disp_imp
        data["metrics_improved"] = metrics_imp

    save_results(data, output_dir)
    print("\n[DONE] Pipeline completata.")
    return disp_base, metrics_base


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    required=True,
                    help="Cartella con imL.png, imR.png, groundtruth.bmp, params.txt")
    ap.add_argument("--window_size", type=int, default=11,
                    help="Lato finestra SAD (default 11, dispari)")
    ap.add_argument("--task2",      action="store_true",
                    help="Abilita Task 2 (LR-check + Moravec)")
    ap.add_argument("--zncc",       action="store_true",
                    help="Usa baseline ZNCC invece di SAD")
    ap.add_argument("--lr_threshold",     type=float, default=1.0,
                    help="Soglia left-right check in pixel (default 1.0)")
    ap.add_argument("--moravec_threshold",type=float, default=100.0,
                    help="Soglia operatore Moravec (default 100)")
    ap.add_argument("--output",     default=None,
                    help="Cartella output (default <dataset>/output)")
    args = ap.parse_args()

    run_pipeline(
        dataset_dir=args.dataset,
        win_size=args.window_size,
        run_task2=args.task2,
        lr_threshold=args.lr_threshold,
        moravec_threshold=args.moravec_threshold,
        output_dir=args.output,
        use_zncc=args.zncc,
    )
