#!/usr/bin/env python3
"""
streamlit_dicom_fredholm_app_complete.py

Complete, corrected Streamlit app for Fredholm inversion of MRI DICOM series.
Includes decorative images (robust), DICOM reading, time extraction, T1/T2 auto-grid,
kernel construction, solvers (FISTA/lsq/nnls), diagnostics, CSV/NPZ export, and plots.
"""

import os
import time
import tempfile
import shutil
import streamlit as st
import numpy as np
import SimpleITK as sitk
from scipy.optimize import lsq_linear, nnls, curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

# matplotlib figure backend imports for placeholder generation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# reduce matplotlib noise
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

try:
    import pydicom
except Exception:
    pydicom = None

st.set_page_config(layout="wide", page_title="Fredholm MRI DICOM Inversion — Brain")

# ------------------ Decorative images ------------------
# Remote image URLs (Unsplash). If remote fetch fails we generate placeholders.
#DECOR_IMAGES = [
    #"https://images.unsplash.com/photo-1526256262350-7da7584cf5eb?auto=format&fit=crop&w=1200&q=80",
    #"https://images.unsplash.com/photo-1582719478250-5f3b0f28d560?auto=format&fit=crop&w=1200&q=80",
    #"https://images.unsplash.com/photo-1582719478638-7b3b0b1e7b54?auto=format&fit=crop&w=1200&q=80"
#]

# Optional: local decorative images (set to None to not use local images)
LOCAL_DECOR_PATHS = ["./assets/1.png", "./assets/2.png", "./assets/image.jpg"]

def make_placeholder_image(text, w=800, h=400, bg=(40,44,52), fg=(220,220,220)):
    """Return an (H,W,3) uint8 RGB placeholder generated with matplotlib Figure."""
    fig = Figure(figsize=(w/100, h/100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor(np.array(bg)/255.0)
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=20, color=np.array(fg)/255.0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)
    # arr shape (H, W, 4) RGBA — convert to RGB
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3].copy()
    else:
        rgb = np.ones((h, w, 3), dtype=np.uint8) * int(bg[0])
    return rgb

def show_decorations(sidebar=True):
    """Display banner and thumbnails robustly: local -> remote -> placeholder."""
    # banner
    try:
        if LOCAL_DECOR_PATHS and os.path.exists(LOCAL_DECOR_PATHS[0]):
            st.image(LOCAL_DECOR_PATHS[0], caption="MRI scanner — demo", use_container_width=True)
        else:
            st.image(DECOR_IMAGES[0], caption="MRI scanner — demo", use_container_width=True)
    except Exception:
        ph = make_placeholder_image("MRI scanner — decorative banner")
        st.image(ph, caption="MRI scanner — demo", use_container_width=True)

    # sidebar thumbnails
    if sidebar:
        st.sidebar.markdown("### Visuals")
        labels = ["MRI scanner", "Colorful MRI", "Patient in MRI"]
        for i, lab in enumerate(labels):
            try:
                if LOCAL_DECOR_PATHS and i < len(LOCAL_DECOR_PATHS) and os.path.exists(LOCAL_DECOR_PATHS[i]):
                    st.sidebar.image(LOCAL_DECOR_PATHS[i], width=200)
                else:
                    st.sidebar.image(DECOR_IMAGES[i], width=200)
            except Exception:
                ph = make_placeholder_image(lab, w=320, h=180)
                st.sidebar.image(ph, width=200)

# ----------------- DICOM reading helpers -----------------

def read_dicom_series_robust(directory, choose_largest_series=True, pad_mode='constant', pad_value=0):
    files_all = [os.path.join(directory, f) for f in sorted(os.listdir(directory))]
    files_all = [f for f in files_all if os.path.isfile(f)]
    if len(files_all) == 0:
        raise RuntimeError(f"No files found in directory: {directory}")

    meta_list = []
    for f in files_all:
        meta = {'file': f, 'series_uid': None, 'inst_no': None, 'acqtime': None, 'echo': None}
        if pydicom is not None:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                meta['series_uid'] = getattr(ds, 'SeriesInstanceUID', None)
                meta['inst_no'] = getattr(ds, 'InstanceNumber', None)
                meta['acqtime'] = getattr(ds, 'AcquisitionTime', None)
                meta['echo'] = getattr(ds, 'EchoTime', None)
            except Exception:
                pass
        meta_list.append(meta)

    groups = {}
    for m in meta_list:
        uid = m['series_uid'] if (m['series_uid'] is not None) else '__no_series__'
        groups.setdefault(uid, []).append(m)

    if choose_largest_series:
        best_uid = max(groups.items(), key=lambda kv: len(kv[1]))[0]
    else:
        best_uid = next(iter(groups.keys()))
    chosen = groups[best_uid]

    def sort_key(m):
        if m.get('inst_no') is not None:
            try:
                return int(m['inst_no'])
            except Exception:
                return m['file']
        if m.get('acqtime') is not None:
            return m['acqtime']
        return m['file']

    chosen_sorted = sorted(chosen, key=sort_key)

    volumes = []
    file_list = []
    max_rows = 0
    max_cols = 0
    for m in chosen_sorted:
        f = m['file']
        try:
            img = sitk.ReadImage(f)
            arr = sitk.GetArrayFromImage(img)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            arr = np.asarray(arr)
            volumes.append(arr)
            file_list.append(f)
            if arr.ndim == 2:
                r, c = arr.shape
            elif arr.ndim == 3:
                r, c = arr.shape[-2], arr.shape[-1]
            else:
                r, c = arr.shape[-2], arr.shape[-1]
            if r > max_rows: max_rows = r
            if c > max_cols: max_cols = c
        except Exception as e:
            st.warning(f"Skipping unreadable file: {os.path.basename(f)} ({e})")
            continue

    if len(volumes) == 0:
        raise RuntimeError("No readable DICOM frames found in chosen series/group.")

    padded = []
    for arr in volumes:
        if arr.ndim == 2:
            r, c = arr.shape
            pad_r = max_rows - r
            pad_c = max_cols - c
            pr1 = pad_r // 2; pr2 = pad_r - pr1
            pc1 = pad_c // 2; pc2 = pad_c - pc1
            if pad_r < 0 or pad_c < 0:
                start_r = max(0, (r - max_rows)//2)
                start_c = max(0, (c - max_cols)//2)
                cropped = arr[start_r:start_r+max_rows, start_c:start_c+max_cols]
                padded.append(cropped)
            else:
                parr = np.pad(arr, ((pr1, pr2), (pc1, pc2)), mode=pad_mode, constant_values=pad_value)
                padded.append(parr)
        elif arr.ndim == 3:
            z, r, c = arr.shape
            pad_r = max_rows - r
            pad_c = max_cols - c
            pr1 = pad_r // 2; pr2 = pad_r - pr1
            pc1 = pad_c // 2; pc2 = pad_c - pc1
            if pad_r < 0 or pad_c < 0:
                start_r = max(0, (r - max_rows)//2)
                start_c = max(0, (c - max_cols)//2)
                cropped = arr[:, start_r:start_r+max_rows, start_c:start_c+max_cols]
                padded.append(cropped)
            else:
                parr = np.pad(arr, ((0,0), (pr1, pr2), (pc1, pc2)), mode=pad_mode, constant_values=pad_value)
                padded.append(parr)
        else:
            try:
                pad_r = max_rows - arr.shape[-2]
                pad_c = max_cols - arr.shape[-1]
                pad_width = [(0,0)]*(arr.ndim-2) + [(pad_r//2, pad_r-pad_r//2), (pad_c//2, pad_c-pad_c//2)]
                parr = np.pad(arr, pad_width, mode=pad_mode, constant_values=pad_value)
                padded.append(parr)
            except Exception:
                padded.append(arr)

    return padded, file_list, best_uid

def read_echo_times_from_files(file_paths):
    times = []
    for p in file_paths:
        et = None
        try:
            img = sitk.ReadImage(p)
            keys = img.GetMetaDataKeys()
            for k in keys:
                if 'echotime' in k.lower() or 'echo time' in k.lower():
                    try:
                        et = float(img.GetMetaData(k))
                        break
                    except Exception:
                        pass
            if et is None and '0018|0081' in keys:
                try:
                    et = float(img.GetMetaData('0018|0081'))
                except Exception:
                    et = None
        except Exception:
            et = None

        if (et is None) and (pydicom is not None):
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                if hasattr(ds, 'EchoTime') and ds.EchoTime is not None:
                    et = float(ds.EchoTime)
                elif hasattr(ds, 'InversionTime') and ds.InversionTime is not None:
                    et = float(ds.InversionTime)
                elif hasattr(ds, 'RepetitionTime') and ds.RepetitionTime is not None:
                    et = float(ds.RepetitionTime)
            except Exception:
                et = None

        if et is None:
            times.append(None)
        else:
            try:
                etf = float(et)
                times.append(etf/1000.0 if etf > 1.0 else etf)
            except Exception:
                times.append(None)
    if all(t is None for t in times):
        return None
    return np.array([np.nan if t is None else float(t) for t in times])

def compute_timeseries_from_volume_list(volumes_list, mask=None):
    S = []
    for vol in volumes_list:
        arr = np.asarray(vol)
        if mask is None:
            S.append(arr.mean())
        else:
            try:
                marr = np.asarray(mask)
                if marr.ndim == 2 and arr.ndim == 3:
                    vals = []
                    for z in range(arr.shape[0]):
                        vals.append(arr[z][marr].mean() if marr.any() else arr[z].mean())
                    S.append(np.mean(vals))
                elif marr.ndim == 3 and arr.ndim == 3 and marr.shape == arr.shape:
                    S.append(arr[marr].mean())
                elif marr.ndim == 2 and arr.ndim == 2:
                    S.append(arr[marr].mean())
                else:
                    S.append(arr.mean())
            except Exception:
                S.append(arr.mean())
    return np.array(S, dtype=float)

# ----------------- Small estimation helpers -----------------
def _aic(n, rss, k):
    if rss <= 0:
        rss = 1e-12
    return 2.0 * k + n * np.log(rss / n)

def estimate_T2_multi_start(S, tvals):
    S = np.asarray(S, dtype=float)
    t = np.asarray(tvals, dtype=float)
    if S.size < 3:
        return None, None
    initials = [0.03, 0.06, 0.08, 0.12, 0.2, 0.5]
    best = None
    best_aic = np.inf
    n = S.size
    for T0 in initials:
        try:
            p0 = [max(S[0] - S[-1], 1e-6), max(T0, 1e-4), np.min(S)]
            def fun(t, A, T, C):
                return A * np.exp(-t / T) + C
            popt, pcov = curve_fit(fun, t, S, p0=p0, bounds=([0, 1e-4, -np.inf], [np.inf, 10.0, np.inf]), maxfev=20000)
            residuals = S - fun(t, *popt)
            rss = float(np.sum(residuals**2))
            aic = _aic(n, rss, k=3)
            if aic < best_aic:
                best_aic = aic
                best = (abs(float(popt[1])), {'model': 'monoexp', 'popt': popt, 'rss': rss, 'aic': aic})
        except Exception:
            continue
    if best is None:
        try:
            mask = S > 0
            if np.sum(mask) >= 3:
                y = np.log(S[mask])
                x = t[mask]
                A = np.vstack([x, np.ones_like(x)]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                T_est = -1.0 / m if m != 0 else None
                if T_est is not None and T_est > 0:
                    return float(T_est), {'model': 'log-linear'}
        except Exception:
            pass
        return None, None
    return best

def estimate_T1_multi_start(S, tvals):
    S = np.asarray(S, dtype=float)
    t = np.asarray(tvals, dtype=float)
    if S.size < 3:
        return None, None

    model_candidates = []
    n = S.size

    # saturation recovery
    for T0 in [0.2, 0.5, 1.0, 1.5]:
        try:
            def fun_sat(t, A, T, C):
                return A * (1.0 - np.exp(-t / T)) + C
            p0 = [np.max(S) - np.min(S), max(T0, 1e-4), np.min(S)]
            popt, pcov = curve_fit(fun_sat, t, S, p0=p0, bounds=([-np.inf, 1e-4, -np.inf], [np.inf, 30.0, np.inf]), maxfev=20000)
            residuals = S - fun_sat(t, *popt)
            rss = float(np.sum(residuals**2))
            aic = _aic(n, rss, k=3)
            model_candidates.append((abs(float(popt[1])), {'model': 'sat', 'popt': popt, 'rss': rss, 'aic': aic}))
        except Exception:
            pass

    # inversion recovery
    for T0 in [0.2, 0.5, 1.0, 1.5]:
        try:
            def fun_ir(t, A, T, C):
                return A * (1.0 - 2.0 * np.exp(-t / T)) + C
            p0 = [np.max(S) - np.min(S), max(T0, 1e-4), np.min(S)]
            popt, pcov = curve_fit(fun_ir, t, S, p0=p0, bounds=([-np.inf, 1e-4, -np.inf], [np.inf, 30.0, np.inf]), maxfev=20000)
            residuals = S - fun_ir(t, *popt)
            rss = float(np.sum(residuals**2))
            aic = _aic(n, rss, k=3)
            model_candidates.append((abs(float(popt[1])), {'model': 'ir', 'popt': popt, 'rss': rss, 'aic': aic}))
        except Exception:
            pass

    if len(model_candidates) == 0:
        return None, None

    best = min(model_candidates, key=lambda it: it[1].get('aic', np.inf))
    return best

# ----------------- Kernel, solver, metrics -----------------

def build_T1_T2_grid(T1_min=0.05, T1_max=5.0, nT1=60, T2_min=0.01, T2_max=1.0, nT2=60):
    T1_values = np.logspace(np.log10(max(1e-4, T1_min)), np.log10(max(T1_min * 1.0001, T1_max)), nT1)
    T2_values = np.logspace(np.log10(max(1e-5, T2_min)), np.log10(max(T2_min * 1.0001, T2_max)), nT2)
    return T1_values, T2_values

def build_kernel_matrix(t_samples, T1_values, T2_values, model_type='T1-T2'):
    t = np.asarray(t_samples).reshape(-1)
    T1g, T2g = np.meshgrid(T1_values, T2_values, indexing='xy')
    T1_flat = T1g.ravel()
    T2_flat = T2g.ravel()
    if model_type == 'T1-T2':
        inv_sum = (1.0 / T1_flat) + (1.0 / T2_flat)
        K = np.exp(-np.outer(t, inv_sum))
    elif model_type == 'T2-only':
        K = np.exp(-np.outer(t, 1.0 / T2_flat))
    elif model_type == 'T1-only':
        K = 1 - np.exp(-np.outer(t, 1.0 / T1_flat))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return K, (T1_flat, T2_flat)

def fista_nonneg(K, S, lam=1e-3, max_iters=2000, tol=1e-6, L=None, callback=None, verbose=False):
    m, n = K.shape
    if L is None:
        try:
            s = np.linalg.svd(K, compute_uv=False)
            sigma_max = s[0]
            L = sigma_max**2 + lam
        except Exception:
            x = np.random.randn(n)
            x /= np.linalg.norm(x)
            for _ in range(20):
                x = K.T.dot(K.dot(x))
                normx = np.linalg.norm(x)
                if normx == 0:
                    break
                x = x / normx
            L = float(x.dot(K.T.dot(K.dot(x)))) + lam
    x_k = np.zeros(n, dtype=float)
    y_k = np.copy(x_k)
    t_k = 1.0
    KT = K.T
    start_time = time.time()
    rel_change = 1.0
    for k in range(1, max_iters+1):
        Ky = K.dot(y_k)
        grad = KT.dot(Ky - S) + lam * y_k
        x_new = y_k - (1.0 / L) * grad
        x_new = np.maximum(0.0, x_new)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_k * t_k)**0.5)
        y_k = x_new + ((t_k - 1.0) / t_new) * (x_new - x_k)
        dx = np.linalg.norm(x_new - x_k)
        denom = max(1.0, np.linalg.norm(x_k))
        rel_change = dx / denom
        if callback is not None and (k % max(1, max_iters//100) == 0):
            callback(k, x_new, rel_change)
        if verbose and (k % max(1, max_iters//10) == 0):
            st.write(f"FISTA iter {k}: rel_change={rel_change:.3e}")
        if rel_change < tol:
            x_k = x_new
            break
        x_k = x_new
        t_k = t_new
    elapsed = time.time() - start_time
    info = {'iterations': k, 'L': L, 'elapsed_s': elapsed, 'converged': (rel_change < tol)}
    return x_k, info

def prepare_csv_bytes(T1_vals, T2_vals, f_recovered, S, tvals):
    nT2, nT1 = f_recovered.shape
    T1_grid, T2_grid = np.meshgrid(T1_vals, T2_vals, indexing='xy')
    T1_flat = T1_grid.ravel()
    T2_flat = T2_grid.ravel()
    f_flat = f_recovered.ravel()
    header = ['T1','T2','f'] + [f'S_t{idx}' for idx in range(len(S))]
    rows = [','.join(header)]
    for i in range(len(f_flat)):
        vals = [f"{T1_flat[i]:.6g}", f"{T2_flat[i]:.6g}", f"{f_flat[i]:.12g}"]
        vals.extend([f"{sv:.12g}" for sv in S])
        rows.append(','.join(vals))
    csv_text = '\n'.join(rows)
    return csv_text.encode('utf-8')

def prepare_st_series_csv_bytes(tvals, S):
    header = 'time_s,S\n'
    lines = [header]
    for t, s in zip(tvals, S):
        lines.append(f"{t:.12g},{s:.12g}\n")
    txt = ''.join(lines)
    return txt.encode('utf-8')

def calculate_performance_metrics(S, S_pred, tvals, x_hat, inversion_time):
    mse = mean_squared_error(S, S_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(S, S_pred)
    r2 = r2_score(S, S_pred)
    s_range = np.max(S) - np.min(S)
    nrmse_range = rmse / s_range if s_range > 0 else 0
    s_mean = np.mean(S)
    nrmse_mean = rmse / s_mean if s_mean > 0 else 0
    residuals = S - S_pred
    residual_std = np.std(residuals)
    signal_std = np.std(S)
    snr = 20 * np.log10(signal_std / residual_std) if residual_std > 0 else float('inf')
    solution_norm = np.linalg.norm(x_hat) if np.linalg.norm(x_hat) > 0 else 1
    residual_norm = np.linalg.norm(residuals)
    condition_ratio = residual_norm / solution_norm
    explained_variance = max(0, 1 - np.var(residuals) / np.var(S))
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'nrmse_range': nrmse_range,
        'nrmse_mean': nrmse_mean,
        'snr_db': snr,
        'condition_ratio': condition_ratio,
        'explained_variance': explained_variance,
        'inversion_time': inversion_time,
        'signal_range': s_range,
        'signal_mean': s_mean,
        'residual_std': residual_std,
        'signal_std': signal_std
    }

# ----------------- UI controls -----------------

st.title('Fredholm Inversion — Brain MRI Relaxometry Mapping for DICOM Series')

st.sidebar.header('Input selection and options')
input_mode = st.sidebar.radio('Select input mode:', ['Local folder (path)', 'Upload files (ZIP or DICOMs)'])

# show decorations?
show_images = st.sidebar.checkbox("Show decorative MRI photos", value=True)
if show_images:
    try:
        show_decorations(sidebar=True)
    except Exception:
        pass

dicom_dir = None
tmpdir = None

if input_mode == 'Local folder (path)':
    dicom_dir = st.sidebar.text_input('Enter local folder path containing DICOM files', value='')
    if dicom_dir and not os.path.isdir(dicom_dir):
        st.sidebar.error('Folder does not exist or is not accessible from server.')
else:
    uploaded = st.sidebar.file_uploader('Upload DICOM files (select multiple) or a ZIP of DICOMs', accept_multiple_files=True, type=None)
    if uploaded:
        tmpdir = tempfile.mkdtemp(prefix='dicom_upload_')
        zips = [u for u in uploaded if u.name.lower().endswith('.zip')]
        if len(zips) > 0:
            import zipfile
            zipf = os.path.join(tmpdir, zips[0].name)
            with open(zipf, 'wb') as f:
                f.write(zips[0].getbuffer())
            try:
                with zipfile.ZipFile(zipf, 'r') as z:
                    z.extractall(tmpdir)
                os.chmod(tmpdir, 0o755)
                st.sidebar.info(f'Extracted zip into temporary folder: {tmpdir}')
                dicom_dir = tmpdir
            except Exception as e:
                st.sidebar.error(f'Failed to extract zip: {e}')
        else:
            for uf in uploaded:
                outp = os.path.join(tmpdir, uf.name)
                with open(outp, 'wb') as f:
                    f.write(uf.getbuffer())
            dicom_dir = tmpdir

st.sidebar.header('Solver & grid settings')
model_type = st.sidebar.selectbox('Relaxometry Model', ['T1-T2', 'T2-only', 'T1-only'])
solver = st.sidebar.selectbox('Solver', ['fista', 'lsq', 'nnls'])
nT1 = int(st.sidebar.number_input('nT1 (T1 bins)', min_value=10, max_value=500, value=60, step=10))
nT2 = int(st.sidebar.number_input('nT2 (T2 bins)', min_value=10, max_value=500, value=60, step=10))

auto_grid = st.sidebar.checkbox('Auto-extract T1/T2 ranges from DICOM & S(t)', value=True)
if not auto_grid:
    T1_min = st.sidebar.number_input('T1 min (s)', min_value=0.001, value=0.05, format='%f')
    T1_max = st.sidebar.number_input('T1 max (s)', min_value=0.01, value=5.0, format='%f')
    T2_min = st.sidebar.number_input('T2 min (s)', min_value=0.001, value=0.01, format='%f')
    T2_max = st.sidebar.number_input('T2 max (s)', min_value=0.001, value=1.0, format='%f')
else:
    T1_min = T1_max = T2_min = T2_max = None

lam = st.sidebar.number_input('Regularization lambda (relative for FISTA)', min_value=0.0, value=1e-4, format='%g')
max_iters = int(st.sidebar.number_input('Max iterations', min_value=10, value=2000, step=10))
tol = float(st.sidebar.number_input('Tolerance (relative change)', min_value=1e-12, value=1e-6, format='%g'))

st.sidebar.header('Signal Preprocessing')
normalize_signal = st.sidebar.checkbox('Normalize signal to [0,1] (recommended)', value=True)
smooth_signal = st.sidebar.checkbox('Apply signal smoothing', value=False)
smooth_window = int(st.sidebar.slider('Smoothing window size', min_value=3, max_value=15, value=5, step=2))

st.sidebar.header('Mask (optional)')
mask_path = st.sidebar.text_input('Local mask file path (.nii/.nii.gz) (leave blank to use full image)')

run_button = st.sidebar.button('Run inversion')

st.subheader('Selected input')
if dicom_dir:
    st.write('DICOM folder:', dicom_dir)
else:
    st.write('No input selected yet.')

# session state keys
if 'csv_grid_bytes' not in st.session_state:
    st.session_state['csv_grid_bytes'] = None
if 'csv_st_bytes' not in st.session_state:
    st.session_state['csv_st_bytes'] = None
if 'npz_path' not in st.session_state:
    st.session_state['npz_path'] = None
if 'model_performance' not in st.session_state:
    st.session_state['model_performance'] = None

# === Main execution ===
if run_button:
    if not dicom_dir:
        st.error('No DICOM input selected.')
    else:
        try:
            total_start_time = time.time()

            with st.spinner('Reading DICOMs (robust mode)...'):
                volumes_list, files, series_uid = read_dicom_series_robust(dicom_dir)
            st.write('Number of frames/series found:', len(volumes_list))
            st.write('Chosen SeriesInstanceUID:', series_uid)

            tvals = read_echo_times_from_files(files)
            if tvals is not None and np.all(np.isnan(tvals)):
                tvals = None
            if tvals is None:
                st.warning('Time values not found in DICOM metadata; using uniform times 0.01-0.1s by default.')
                tvals = np.linspace(0.01, 0.1, len(volumes_list))
            st.write('Echo/time samples (s):', np.array2string(tvals, precision=6))

            # Field strength attempt
            field_T = None
            if pydicom is not None:
                try:
                    vals = []
                    for f in files[:min(len(files), 20)]:
                        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                        if hasattr(ds, 'MagneticFieldStrength') and ds.MagneticFieldStrength is not None:
                            vals.append(float(ds.MagneticFieldStrength))
                    if len(vals) > 0:
                        field_T = float(np.median(vals))
                except Exception:
                    field_T = None
            st.write('Detected Magnetic Field Strength (T):', field_T)

            mask = None
            if mask_path:
                try:
                    mimg = sitk.ReadImage(mask_path)
                    marr = sitk.GetArrayFromImage(mimg)
                    mask = (marr > 0)
                    st.write('Loaded mask shape:', mask.shape)
                except Exception as e:
                    st.error('Failed to load mask: ' + str(e))
                    mask = None

            S = compute_timeseries_from_volume_list(volumes_list, mask=mask)
            st.write('Measured S(t) length:', len(S))

            # preprocessing choices
            if normalize_signal:
                S_original = S.copy()
                S_min, S_max = np.min(S), np.max(S)
                if S_max > S_min:
                    S = (S - S_min) / (S_max - S_min)
                st.write('Signal normalized to range [0, 1]')
            else:
                S_original = S.copy()

            if smooth_signal:
                from scipy.ndimage import uniform_filter1d
                S = uniform_filter1d(S, size=smooth_window)
                st.write(f'Signal smoothed with window size {smooth_window}')

            if np.any(tvals <= 0):
                st.error('Time values must be positive. Adjusting negative/zero values to 0.001s.')
                tvals[tvals <= 0] = 0.001

            if np.std(S) < 1e-12:
                st.error('Signal has no variation. Check the DICOM series.')
                st.stop()

            # Auto-grid estimation (brain defaults)
            def brain_default_ranges(field_T=None):
                if field_T is None: field_T = 1.5
                if field_T >= 3.0:
                    return 0.2, 3.0, 0.02, 0.3
                elif field_T >= 1.5:
                    return 0.15, 2.5, 0.02, 0.25
                else:
                    return 0.12, 2.0, 0.01, 0.2

            def_minT1, def_maxT1, def_minT2, def_maxT2 = brain_default_ranges(field_T)
            est_T2, info_T2 = estimate_T2_multi_start(S, tvals)
            est_T1, info_T1 = estimate_T1_multi_start(S, tvals)
            if est_T2 is not None and np.isfinite(est_T2):
                est_T2 = float(np.clip(est_T2, def_minT2, def_maxT2))
            else:
                est_T2 = (def_minT2 + def_maxT2) / 4.0
            if est_T1 is not None and np.isfinite(est_T1):
                est_T1 = float(np.clip(est_T1, def_minT1, def_maxT1))
            else:
                est_T1 = (def_minT1 + def_maxT1) / 3.0

            used_T1_min = max(def_minT1, est_T1 / 5.0)
            used_T1_max = min(def_maxT1, est_T1 * 5.0)
            used_T2_min = max(def_minT2, est_T2 / 5.0)
            used_T2_max = min(def_maxT2, est_T2 * 5.0)
            st.write(f"Estimated T1 ≈ {est_T1:.4g} s  (bounds used: {used_T1_min:.4g}–{used_T1_max:.4g} s)")
            st.write(f"Estimated T2 ≈ {est_T2:.4g} s  (bounds used: {used_T2_min:.4g}–{used_T2_max:.4g} s)")

            # Build grid
            if model_type == 'T2-only':
                T1_vals = np.array([1.0])
                T2_vals = np.logspace(np.log10(used_T2_min), np.log10(used_T2_max), nT2)
            elif model_type == 'T1-only':
                T1_vals = np.logspace(np.log10(used_T1_min), np.log10(used_T1_max), nT1)
                T2_vals = np.array([1.0])
            else:
                T1_vals, T2_vals = build_T1_T2_grid(used_T1_min, used_T1_max, nT1, used_T2_min, used_T2_max, nT2)

            K, (T1_flat, T2_flat) = build_kernel_matrix(tvals, T1_vals, T2_vals, model_type)
            st.write('Kernel K shape:', K.shape)

            # Display ranges: linear then log (log after T2 linear as requested)
            try:
                st.write(f"T1 range (linear): {T1_vals.min():.6g} s — {T1_vals.max():.6g} s")
                st.write(f"T2 range (linear): {T2_vals.min():.6g} s — {T2_vals.max():.6g} s")
                if np.all(T2_vals > 0):
                    st.write(f"T2 range (log10): {np.log10(T2_vals.min()):.6g} — {np.log10(T2_vals.max()):.6g}")
                if np.all(T1_vals > 0):
                    st.write(f"T1 range (log10): {np.log10(T1_vals.min()):.6g} — {np.log10(T1_vals.max()):.6g}")
            except Exception:
                pass

            # Quick solver path (with baseline & scaling for stability)
            progress_bar = st.progress(0)
            progress_text = st.empty()

            baseline_est = float(np.median(S[-max(2, int(0.1 * len(S))):]))
            S_bc = S - baseline_est
            if normalize_signal:
                S_work = S_bc.copy()
            else:
                scale_val = np.max(np.abs(S_bc)) if np.max(np.abs(S_bc)) > 0 else 1.0
                S_work = S_bc / scale_val

            K_work = K.astype(float).copy()
            col_norms = np.linalg.norm(K_work, axis=0)
            col_norms_safe = np.where(col_norms == 0, 1.0, col_norms)
            K_work = K_work / col_norms_safe

            inversion_start_time = time.time()
            if solver == 'lsq':
                res = lsq_linear(K_work, S_work, bounds=(0, np.inf), lsmr_tol='auto', max_iter=int(max_iters))
                x_work = res.x
                back_scale = 1.0 if normalize_signal else scale_val
                x_hat = (x_work / col_norms_safe) * back_scale
                info = {'method': 'lsq_linear', 'nit': res.nit, 'status': res.status}
            elif solver == 'nnls':
                xw, residual = nnls(K_work, S_work)
                back_scale = 1.0 if normalize_signal else scale_val
                x_hat = (xw / col_norms_safe) * back_scale
                info = {'method': 'nnls', 'residual': residual}
            else:
                svals = np.linalg.svd(K_work, compute_uv=False)
                L_est = float(svals[0]**2) if svals.size > 0 else 1.0
                lam_scaled = float(lam) * L_est if lam > 0 else 1e-4 * L_est
                xw, info_f = fista_nonneg(K_work, S_work, lam=lam_scaled, max_iters=int(max_iters), tol=tol, callback=None, verbose=False)
                back_scale = 1.0 if normalize_signal else scale_val
                x_hat = (xw / col_norms_safe) * back_scale
                info = info_f
                info.update({'method': 'fista_nonneg', 'lam_scaled': lam_scaled})

            inversion_time = time.time() - inversion_start_time
            info['inversion_time'] = inversion_time

            # reshape recovered distribution
            if model_type == 'T2-only':
                f_recovered = x_hat.reshape(len(T2_vals), 1)
            elif model_type == 'T1-only':
                f_recovered = x_hat.reshape(1, len(T1_vals))
            else:
                f_recovered = x_hat.reshape(len(T2_vals), len(T1_vals))

            # Predict and compute metrics
            S_pred = K @ x_hat
            performance = calculate_performance_metrics(S, S_pred, tvals, x_hat, inversion_time)
            st.session_state['model_performance'] = performance
            performance['total_processing_time'] = time.time() - total_start_time

            # Save results to npz
            out_npz = os.path.join(tempfile.gettempdir(), f'recovered_{int(time.time())}.npz')
            np.savez(out_npz, T1_values=T1_vals, T2_values=T2_vals, f_recovered=f_recovered,
                     f_vector=x_hat, S=S, S_pred=S_pred, t_values=tvals, info=info, performance=performance)
            st.session_state['npz_path'] = out_npz

            # Display metrics + plots
            st.subheader('Model Performance')
            cols = st.columns(5)
            cols[0].metric("MSE", f"{performance['mse']:.4e}")
            cols[1].metric("RMSE", f"{performance['rmse']:.4e}")
            cols[2].metric("MAE", f"{performance['mae']:.4e}")
            cols[3].metric("R²", f"{performance['r2']:.4f}")
            cols[4].metric("Explained Var", f"{performance['explained_variance']:.4f}")

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes[0, 0].plot(tvals, S, 'o-', label='Measured', linewidth=2, markersize=6)
            axes[0, 0].plot(tvals, S_pred, 's--', label='Predicted', linewidth=2, markersize=4)
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Signal Intensity')
            axes[0, 0].set_title('Measured vs Predicted Signal')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            residuals = S - S_pred
            axes[0, 1].plot(tvals, residuals, 'o-', linewidth=2, markersize=6)
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals (Measured - Predicted)')
            axes[0, 1].grid(True, alpha=0.3)

            if model_type == 'T2-only':
                axes[0, 2].plot(T2_vals, f_recovered.flatten(), 'o-', linewidth=2)
                axes[0, 2].set_xlabel('T2 (s)')
                axes[0, 2].set_xscale('log')
                axes[0, 2].set_title('Recovered T2 Distribution')
            elif model_type == 'T1-only':
                axes[0, 2].plot(T1_vals, f_recovered.flatten(), 'o-', linewidth=2)
                axes[0, 2].set_xlabel('T1 (s)')
                axes[0, 2].set_xscale('log')
                axes[0, 2].set_title('Recovered T1 Distribution')
            else:
                im = axes[0, 2].imshow(np.log10(f_recovered + 1e-12), aspect='auto',
                                      extent=(np.log10(T1_vals[0]), np.log10(T1_vals[-1]),
                                              np.log10(T2_vals[-1]), np.log10(T2_vals[0])))
                axes[0, 2].set_xlabel('log10($T_{1}$)')
                axes[0, 2].set_ylabel('log10($T_{2}$)')
                axes[0, 2].set_title('Recovered log10 f($T_{1}$,$T_{2}$)')
                fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

            axes[1, 0].hist(S, bins=20, alpha=0.7)
            axes[1, 0].set_xlabel('Signal Intensity')
            axes[1, 0].set_title('Signal Distribution')

            axes[1, 1].hist(residuals, bins=20, alpha=0.7)
            axes[1, 1].set_xlabel('Residual Value')
            axes[1, 1].set_title('Residual Distribution')

            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=axes[1, 2])
            axes[1, 2].set_title('Q-Q Plot of Residuals')

            plt.tight_layout()
            st.pyplot(fig)

            # decorative gallery under plots if enabled
            if show_images:
                try:
                    if LOCAL_DECOR_PATHS:
                        imgs = [p for p in LOCAL_DECOR_PATHS if os.path.exists(p)]
                        if imgs:
                            st.image(imgs, width=300)
                    else:
                        st.image(DECOR_IMAGES, width=300, caption=["MRI scanner", "Colorful MRI", "Patient in MRI"])
                except Exception:
                    placeholders = [make_placeholder_image("MRI 1", 300, 200),
                                    make_placeholder_image("MRI 2", 300, 200),
                                    make_placeholder_image("MRI 3", 300, 200)]
                    st.image(placeholders, width=300)

            # Prepare downloads
            try:
                csv_grid = prepare_csv_bytes(T1_vals, T2_vals, f_recovered, S, tvals)
                csv_st = prepare_st_series_csv_bytes(tvals, S)
                st.session_state['csv_grid_bytes'] = csv_grid
                st.session_state['csv_st_bytes'] = csv_st
                st.success('CSV prepared successfully.')
            except Exception as e:
                st.warning(f'Failed to prepare CSV: {e}')
                st.session_state['csv_grid_bytes'] = None
                st.session_state['csv_st_bytes'] = None

        except Exception as e:
            st.error(f'Error during processing: {e}')
        finally:
            if tmpdir and os.path.isdir(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    pass

# Downloads / Exports UI
st.markdown('---')
st.header('Downloads / Exports (available after successful inversion)')

if st.session_state.get('csv_grid_bytes') is not None:
    st.download_button('Download CSV ($T_{1}$, $T_{2}$, f, S_t...)', st.session_state['csv_grid_bytes'],
                       file_name=f'recovered_grid_{int(time.time())}.csv', mime='text/csv', key='dl_grid')
else:
    st.info('CSV grid not yet available. Run inversion to produce it.')

if st.session_state.get('csv_st_bytes') is not None:
    st.download_button('Download S(t) CSV', st.session_state['csv_st_bytes'],
                       file_name=f'S_timeseries_{int(time.time())}.csv', mime='text/csv', key='dl_st')
else:
    st.info('S(t) CSV not yet available. Run inversion to produce it.')

if st.session_state.get('npz_path') is not None and os.path.exists(st.session_state['npz_path']):
    with open(st.session_state['npz_path'], 'rb') as fh:
        data = fh.read()
    st.download_button('Download recovered .npz', data=data, file_name=os.path.basename(st.session_state['npz_path']), mime='application/octet-stream', key='dl_npz')
else:
    st.info('.npz not yet available. Run inversion to create it.')

st.sidebar.markdown('---')
st.sidebar.write('Diagnostics:')
st.sidebar.write('pydicom installed:', bool(pydicom))
st.sidebar.write('CSV prepared in session:', st.session_state.get('csv_grid_bytes') is not None)

if st.session_state.get('model_performance'):
    st.sidebar.markdown('---')
    st.sidebar.subheader('Last Run Performance')
    perf = st.session_state['model_performance']
    st.sidebar.write(f"R²: {perf['r2']:.4f}")
    st.sidebar.write(f"RMSE: {perf['rmse']:.4e}")
    st.sidebar.write(f"MAE: {perf['mae']:.4e}")
    st.sidebar.write(f"Time: {perf['total_processing_time']:.2f} s")
