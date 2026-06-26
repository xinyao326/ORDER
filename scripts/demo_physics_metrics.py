import argparse
import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.stats import pearsonr, spearmanr

sys.path.append('..')


def detect_fiber_features(img_np: np.ndarray, min_area_frac: float = 0.001):
    h, w = img_np.shape
    min_area = h * w * min_area_frac
    max_area = h * w * 0.25

    tiled = np.tile(img_np, (3, 3))
    blurred_t = cv2.GaussianBlur(tiled, (5, 5), 0)
    edges = cv2.Canny(blurred_t, 20, 60)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    det_N = 0
    areas_valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx_c = M['m10'] / M['m00']
        cy_c = M['m01'] / M['m00']
        if w <= cx_c < 2 * w and h <= cy_c < 2 * h:
            det_N += 1
            areas_valid.append(area)

    vf_est = sum(areas_valid) / (h * w)

    blurred_orig = cv2.GaussianBlur(img_np, (5, 5), 0)
    gx = cv2.Sobel(blurred_orig.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred_orig.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    strong = mag > np.percentile(mag, 90)
    angles = np.arctan2(gy[strong], gx[strong])
    # exp(2i·θ): θ and θ+π treated identically (direction, not orientation)
    circ_var = 1.0 - abs(np.mean(np.exp(2j * angles)))

    return det_N, vf_est, circ_var


def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('L'))


def corr_row(x, y):
    r, p_r = pearsonr(x, y)
    rho, p_s = spearmanr(x, y)
    mae = np.mean(np.abs(np.array(x) - np.array(y)))
    return r, p_r, rho, p_s, mae


def validate_on_real(args):
    csv_path = os.path.join('..', 'datasets_composite', 'train_clean.csv')
    img_dir  = os.path.join('..', 'datasets_composite', 'processed')
    df = pd.read_csv(csv_path)

    rows = []
    missing = 0
    for _, row in df.iterrows():
        idx = int(row['Image index'])
        fpath = os.path.join(img_dir, f'{idx:03d}.png')
        if not os.path.exists(fpath):
            missing += 1
            continue
        img = load_gray(fpath)
        det_N, vf_est, circ_var = detect_fiber_features(img)
        rows.append({'gt_N': row['NumFibers'], 'gt_MMA': row['MMA'], 'gt_Vf': row['Vf'],
                     'det_N': det_N, 'vf_est': vf_est, 'circ_var': circ_var})

    rdf = pd.DataFrame(rows)
    print(f'\nProcessed {len(rdf)} real images  ({missing} missing)\n')

    r, p_r, rho, p_s, mae = corr_row(rdf['gt_N'], rdf['det_N'])
    print('=== Fiber Count (det_N vs gt_NumFibers) ===')
    print(f'  Pearson  r = {r:.3f}  (p={p_r:.2e})')
    print(f'  Spearman ρ = {rho:.3f}  (p={p_s:.2e})')
    print(f'  MAE        = {mae:.2f} fibers   '
          f'(systematic ~{rdf["det_N"].mean()/rdf["gt_N"].mean():.2f}× scale from tiling artefacts)')
    print(f'  gt mean = {rdf["gt_N"].mean():.1f},  detected mean = {rdf["det_N"].mean():.1f}')

    r, p_r, rho, p_s, _ = corr_row(rdf['gt_MMA'], rdf['circ_var'])
    print('\n=== MMA Proxy (circ_var vs gt_MMA)  [negative r: lower circ_var → more elongation] ===')
    print(f'  Pearson  r = {r:.3f}  (p={p_r:.2e})')
    print(f'  Spearman ρ = {rho:.3f}  (p={p_s:.2e})')
    mma0_cv = rdf.loc[rdf['gt_MMA'] == 0, 'circ_var'].mean()
    mma3_cv = rdf.loc[rdf['gt_MMA'] > 3, 'circ_var'].mean()
    print(f'  circ_var mean at MMA = 0 : {mma0_cv:.4f}')
    print(f'  circ_var mean at MMA > 3 : {mma3_cv:.4f}  (Δ = {mma3_cv - mma0_cv:.4f})')

    r, p_r, rho, p_s, _ = corr_row(rdf['gt_Vf'], rdf['vf_est'])
    print('\n=== Vf Proxy (vf_est vs gt_Vf) ===')
    print(f'  Pearson  r = {r:.3f}  (p={p_r:.2e})')
    print(f'  Spearman ρ = {rho:.3f}  (p={p_s:.2e})')

    print('\n=== Sample-level check ===')
    print(f'{"idx":>5}  {"gt_N":>4}  {"det_N":>5}  {"gt_MMA":>6}  {"circ_var":>8}  {"gt_Vf":>5}  {"vf_est":>6}')
    rdf2 = rdf.copy()
    rdf2['Image index'] = [int(df.iloc[i]['Image index']) for i in range(len(rdf2))]
    for cid in [83, 104, 89, 758, 820, 903]:
        sub = rdf2[rdf2['Image index'] == cid]
        if len(sub):
            r2 = sub.iloc[0]
            print(f'{cid:>5}  {r2["gt_N"]:>4.0f}  {r2["det_N"]:>5}  '
                  f'{r2["gt_MMA"]:>6.3f}  {r2["circ_var"]:>8.4f}  '
                  f'{r2["gt_Vf"]:>5.3f}  {r2["vf_est"]:>6.4f}')


def _collect_rows_composite(gen_dir, csv_path):
    df = pd.read_csv(csv_path)
    sample_dirs = sorted(
        [d for d in os.listdir(gen_dir) if d.isdigit()],
        key=lambda x: int(x)
    )
    rows = []
    n_gen = 0
    for dir_name in sample_dirs:
        i = int(dir_name)
        if i >= len(df):
            continue
        gt_row = df.iloc[i]
        sample_path = os.path.join(gen_dir, dir_name)
        real_path   = os.path.join(sample_path, 'real.png')
        if not os.path.exists(real_path):
            continue

        real_img = load_gray(real_path)
        r_N, r_vf, r_cv = detect_fiber_features(real_img)

        gen_files = sorted([f for f in os.listdir(sample_path)
                            if f.endswith('.png') and f != 'real.png'])
        if not gen_files:
            continue
        n_gen = len(gen_files)

        g_Ns, g_vfs, g_cvs = [], [], []
        for gf in gen_files:
            gen_img = load_gray(os.path.join(sample_path, gf))
            gN, gvf, gcv = detect_fiber_features(gen_img)
            g_Ns.append(gN); g_vfs.append(gvf); g_cvs.append(gcv)

        rows.append({
            'idx': i,
            'gt_NumFibers': gt_row['NumFibers'],
            'gt_MMA':       gt_row['MMA'],
            'gt_Vf':        gt_row['Vf'],
            'real_N':  r_N,              'gen_N':    np.mean(g_Ns),
            'real_vf': r_vf,             'gen_vf':   np.mean(g_vfs),
            'real_cv': r_cv,             'gen_cv':   np.mean(g_cvs),
            'gen_N_std':  np.std(g_Ns),
            'gen_vf_std': np.std(g_vfs),
            'gen_cv_std': np.std(g_cvs),
        })
    return rows, n_gen


def eval_generated(args):
    if not os.path.exists(args.gen_dir):
        raise RuntimeError(f'Generated image directory not found: {args.gen_dir}')
    if not os.path.exists(args.csv):
        raise RuntimeError(f'CSV not found: {args.csv}')

    rows, n_gen = _collect_rows_composite(args.gen_dir, args.csv)

    if args.gen_dir2 and args.csv2:
        if not os.path.exists(args.gen_dir2):
            raise RuntimeError(f'Generated image directory not found: {args.gen_dir2}')
        if not os.path.exists(args.csv2):
            raise RuntimeError(f'CSV not found: {args.csv2}')
        rows2, n_gen2 = _collect_rows_composite(args.gen_dir2, args.csv2)
        rows.extend(rows2)
        n_gen = n_gen or n_gen2

    rdf = pd.DataFrame(rows)
    n_gen = n_gen  # keep last observed value
    print(f'\nEvaluated {len(rdf)} test samples  '
          f'({n_gen} generated + 1 real each)\n')

    r_n, p_n, rho_n, _, mae_n = corr_row(rdf['gt_NumFibers'], rdf['real_N'])
    r_m, p_m, rho_m, _, _    = corr_row(rdf['gt_MMA'],       rdf['real_cv'])
    r_v, p_v, rho_v, _, _    = corr_row(rdf['gt_Vf'],        rdf['real_vf'])
    print('=== 1. Detector validity on real test images ===')
    print(f'  NumFibers: Pearson={r_n:.3f} (p={p_n:.1e}), Spearman={rho_n:.3f},  MAE={mae_n:.2f}')
    print(f'             gt_mean={rdf["gt_NumFibers"].mean():.1f},  '
          f'detected_mean={rdf["real_N"].mean():.1f}')
    print(f'  MMA proxy: Pearson={r_m:.3f} (p={p_m:.1e}), Spearman={rho_m:.3f}')
    mma0 = rdf.loc[rdf['gt_MMA'] == 0, 'real_cv']
    mma3 = rdf.loc[rdf['gt_MMA'] > 3, 'real_cv']
    if len(mma0): print(f'             circ_var at MMA=0  : {mma0.mean():.4f}')
    if len(mma3): print(f'             circ_var at MMA>3  : {mma3.mean():.4f}')
    print(f'  Vf proxy:  Pearson={r_v:.3f} (p={p_v:.1e}), Spearman={rho_v:.3f}')

    r_ng, _, rho_ng, _, mae_ng = corr_row(rdf['real_N'],  rdf['gen_N'])
    r_mg, _, rho_mg, _, mae_mg = corr_row(rdf['real_cv'], rdf['gen_cv'])
    r_vg, _, rho_vg, _, mae_vg = corr_row(rdf['real_vf'], rdf['gen_vf'])
    print('\n=== 2. Physics fidelity: generated vs real ===')
    print(f'  NumFibers proxy:  Pearson={r_ng:.3f}, Spearman={rho_ng:.3f},  MAE={mae_ng:.2f} (detected fibers)')
    print(f'    real_mean={rdf["real_N"].mean():.1f},  gen_mean={rdf["gen_N"].mean():.1f}  '
          f'(within-sample std={rdf["gen_N_std"].mean():.1f})')
    print(f'  MMA proxy (cv):   Pearson={r_mg:.3f}, Spearman={rho_mg:.3f},  MAE={mae_mg:.4f}')
    print(f'    real_cv_mean={rdf["real_cv"].mean():.4f},  gen_cv_mean={rdf["gen_cv"].mean():.4f}  '
          f'(within-sample std={rdf["gen_cv_std"].mean():.4f})')
    print(f'  Vf proxy:         Pearson={r_vg:.3f}, Spearman={rho_vg:.3f},  MAE={mae_vg:.4f}')
    print(f'    real_vf_mean={rdf["real_vf"].mean():.4f},  gen_vf_mean={rdf["gen_vf"].mean():.4f}')

    print('\n=== 3. Per-sample results (first 10) ===')
    hdr = f'{"i":>3}  {"gt_N":>4}  {"real_N":>6}  {"gen_N":>6}  {"gt_MMA":>6}  {"real_cv":>8}  {"gen_cv":>8}'
    print(hdr)
    for _, r in rdf.head(10).iterrows():
        print(f'{int(r["idx"]):>3}  {r["gt_NumFibers"]:>4.0f}  {r["real_N"]:>6.1f}  '
              f'{r["gen_N"]:>6.1f}  {r["gt_MMA"]:>6.3f}  {r["real_cv"]:>8.4f}  {r["gen_cv"]:>8.4f}')

    rdf['n_err'] = (rdf['real_N'] - rdf['gen_N']).abs()
    print('\n=== 4. Worst 5 samples by |gen_N − real_N| ===')
    print(hdr)
    for _, r in rdf.nlargest(5, 'n_err').iterrows():
        print(f'{int(r["idx"]):>3}  {r["gt_NumFibers"]:>4.0f}  {r["real_N"]:>6.1f}  '
              f'{r["gen_N"]:>6.1f}  {r["gt_MMA"]:>6.3f}  {r["real_cv"]:>8.4f}  {r["gen_cv"]:>8.4f}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',    type=str, default='validate',
                   choices=['validate', 'eval'])
    p.add_argument('--dataset', type=str, default='composite')
    p.add_argument('--gen_dir', type=str,
                   help='Directory produced by generate.py / test_generate.py')
    p.add_argument('--csv',     type=str,
                   help='CSV with NumFibers, MMA, Vf columns (rows = DataLoader order)')
    p.add_argument('--gen_dir2', type=str, default=None,
                   help='Optional second gen dir to pool with --gen_dir (e.g. gen-train when --gen_dir is gen-test)')
    p.add_argument('--csv2',     type=str, default=None,
                   help='CSV corresponding to --gen_dir2')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'validate':
        validate_on_real(args)
    else:
        eval_generated(args)
