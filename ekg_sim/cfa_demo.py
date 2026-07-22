# """
# Cardiac Field Artifact (CFA) removal — synthetic-data + block-wiring demonstrator
# =================================================================================

# Single channel, to keep the wiring legible. Two learners, two jobs:

#   Stage A  Repetitive / ILC template   -> learns the repeatable per-beat WAVEFORM a_bar(phi)
#   Stage B  Hammerstein basis + RLS      -> learns the nonlinear MAP f(.) of the reference,
#                                            tracking beat-to-beat variation the template misses

# Generative (ground-truth) model:
#       d(t)          synthetic ECG electrical source  (quasi-periodic, RR jitter + slow drift)
#       f(.)          static electrode/tissue nonlinearity  (THE thing Stage B must recover)
#       h[.]          volume-conduction FIR (linear, channel-specific)
#       a(t) = h * f(d)      = the CFA (Hammerstein: static NL then linear filter)
#       s(t)          neural signal to PRESERVE (pink noise + 10 Hz alpha)
#       y(t) = s + a + noise        measured EEG
#       r(t) = d + ref_noise        measured ECG reference  (a good, but noisy, view of d)
# """

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
fs = 250.0                      # Hz
dur = 90.0                      # s
N = int(fs * dur)
t = np.arange(N) / fs

# ----------------------------------------------------------------------------
# 1. SYNTHETIC ECG SOURCE  d(t)  -- R-anchored PQRST, with HRV + slow amp drift
# ----------------------------------------------------------------------------
def beat_template(fs):
    """One PQRST complex, R anchored at t'=0, on [-0.30, +0.50] s."""
    tt = np.arange(-0.30, 0.50, 1.0 / fs)
    g = lambda c, a, w: a * np.exp(-0.5 * ((tt - c) / w) ** 2)
    wave = (g(-0.20, 0.10, 0.025)   # P
            + g(-0.025, -0.15, 0.008)  # Q
            + g(0.0, 1.00, 0.010)      # R
            + g(0.025, -0.25, 0.010)   # S
            + g(0.18, 0.30, 0.040))    # T
    return tt, wave

tt_b, wave_b = beat_template(fs)
r0 = int(0.30 * fs)              # index of R within the template

# R-peak times: mean HR 70 bpm, HRV = respiratory sinus + jitter
rr_mean = 60.0 / 70.0
rpeaks = []
tk = 1.0
while tk < dur - 1.0:
    rpeaks.append(tk)
    rr = rr_mean * (1 + 0.04 * np.sin(2 * np.pi * 0.25 * tk)) + rng.normal(0, 0.015)
    tk += rr
rpeaks = np.array(rpeaks)
rpeak_idx = np.round(rpeaks * fs).astype(int)

# slow amplitude drift (posture / electrode contact) -- this is what breaks a
# fixed template and forces the adaptive stage to matter
amp_drift = 1.0 + 0.25 * np.sin(2 * np.pi * 0.03 * t) + 0.05 * np.sin(2 * np.pi * 0.11 * t)

d = np.zeros(N)
for k, idx in enumerate(rpeak_idx):
    lo = idx - r0
    seg_lo = max(lo, 0)
    a0 = seg_lo - lo
    seg_hi = min(lo + len(wave_b), N)
    a1 = a0 + (seg_hi - seg_lo)
    d[seg_lo:seg_hi] += amp_drift[idx if idx < N else -1] * wave_b[a0:a1]

# ----------------------------------------------------------------------------
# 2. STATIC NONLINEARITY f(.)   (ground truth to be recovered by Stage B)
#    soft saturation (odd) + rectifying quadratic (even). The even term is what
#    a purely linear filter CANNOT represent -> motivates the basis expansion.
# ----------------------------------------------------------------------------
def f_true(x):
    return np.tanh(1.6 * x) + 0.18 * x ** 2

# ----------------------------------------------------------------------------
# 3. VOLUME-CONDUCTION FIR h[.]  (linear, low-pass + delay + gain)
# ----------------------------------------------------------------------------
h = np.array([0.05, 0.20, 0.35, 0.25, 0.10, 0.05])   # smoothing/low-pass
h = h / h.sum()
cfa = signal.lfilter(h, 1.0, f_true(d))
cfa *= 3.0                                            # scale artifact up (near-ear: large)

# ----------------------------------------------------------------------------
# 4. NEURAL SIGNAL s(t) to preserve: pink noise + amplitude-modulated alpha
# ----------------------------------------------------------------------------
white = rng.normal(0, 1, N)
b_pink = np.array([0.049922, -0.095993, 0.050612, -0.004408])
a_pink = np.array([1, -2.494956, 2.017265, -0.522189])
pink = signal.lfilter(b_pink, a_pink, white)
pink /= np.std(pink)
alpha_env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t)
alpha = alpha_env * np.sin(2 * np.pi * 10.0 * t)
s = 0.7 * pink + 0.6 * alpha

# ----------------------------------------------------------------------------
# 5. MEASURED SIGNALS
# ----------------------------------------------------------------------------
eeg_noise = rng.normal(0, 0.10, N)
y = s + cfa + eeg_noise                     # contaminated EEG
r = d + rng.normal(0, 0.03, N)              # measured ECG reference (clean-ish view of d)

eval_mask = t >= 40.0                        # steady-state window (drift fully exercised)
def snr_db(clean, est, m=eval_mask):         # artifact SNR: power(clean)/power(residual-error)
    return 10 * np.log10(np.sum(clean[m] ** 2) / np.sum((clean[m] - est[m]) ** 2))

snr_in = snr_db(s, y)                        # how far y is from clean s

# ============================================================================
#                               PIPELINE
# ============================================================================

# ----- R-peak detection on the reference (Pan-Tompkins-lite) ----------------
sos = signal.butter(2, [5, 20], btype="band", fs=fs, output="sos")
rf = signal.sosfilt(sos, r)
integ = np.convolve(np.diff(rf, prepend=rf[0]) ** 2, np.ones(int(0.10 * fs)), "same")
det_idx, _ = signal.find_peaks(integ, distance=int(0.4 * fs),
                               height=0.3 * np.max(integ))
# refine each detection to the true local max of r nearby
det_r = []
for i in det_idx:
    w0, w1 = max(i - int(0.05 * fs), 0), min(i + int(0.05 * fs), N)
    det_r.append(w0 + int(np.argmax(r[w0:w1])))
det_r = np.array(det_r)

# ----- STAGE A: repetitive / ILC template ----------------------------------
# window each beat [-pre,+post] around detected R, learn the repeatable waveform
pre, post = int(0.30 * fs), int(0.50 * fs)
L = pre + post
template = np.zeros(L)
mu_ilc = 0.15                                # ILC learning gain (leaky integrator)
n_learn = 30                                 # freeze template after it converges
tmpl_hist = []
e1 = y.copy()                                # residual after template subtraction
for bi, i in enumerate(det_r):
    lo, hi = i - pre, i + post
    if lo < 0 or hi > N:
        continue
    beat = y[lo:hi]
    if bi < n_learn:                         # learn, then freeze (repetitive control converges)
        template = template + mu_ilc * (beat - template)
        tmpl_hist.append(template.copy())
    e1[lo:hi] = y[lo:hi] - template          # subtract converged template
tmpl_hist = np.array(tmpl_hist)
# convergence = distance of running template to the converged (final) one
tmpl_conv = np.linalg.norm(tmpl_hist - tmpl_hist[-1], axis=1)

# ----- STAGE B: Hammerstein basis expansion + RLS on the residual -----------
# regressor = Chebyshev basis of the reference, each through an M-tap delay line
P_ORDER = 4                                  # Chebyshev order (0..4)
M = 8                                         # taps per basis channel (>= true h length)
rn = r / (np.max(np.abs(r)) + 1e-9)          # normalize to [-1,1] for Chebyshev
Phi = np.vstack([np.polynomial.chebyshev.Chebyshev.basis(k)(rn)
                 for k in range(P_ORDER + 1)])          # (P+1, N)

def delay_stack(x, M):
    return np.vstack([np.r_[np.zeros(m), x[:N - m]] for m in range(M)])  # (M,N)

X = np.vstack([delay_stack(Phi[k], M) for k in range(P_ORDER + 1)])      # ((P+1)*M, N)
D = X.shape[0]

lam = 0.999                                  # RLS forgetting (tracks the slow drift)
w = np.zeros(D)
Pmat = np.eye(D) * 1.0                        # small init -> mild ridge regularization
clean = e1.copy()
for n in range(N):
    xn = X[:, n]
    yhat = w @ xn
    err = e1[n] - yhat
    Px = Pmat @ xn
    g = Px / (lam + xn @ Px)
    w += g * err
    Pmat = (Pmat - np.outer(g, Px)) / lam
    clean[n] = err                           # residual after removing reference-driven part

snr_A = snr_db(s, e1)
snr_B = snr_db(s, clean)

# ----- recover the static nonlinearity via rank-1 Hammerstein identification
# A Hammerstein weight matrix factorizes as W = c (X) h  (nonlinearity coeffs
# outer-product filter taps), so it is RANK-1. Identify W by mapping the basis
# delay-lines of the reference to the measured EEG (neural is uncorrelated noise
# to the least-squares solve), then take the top singular vector for c.
gamma = 1e-2 * np.trace(X @ X.T) / D                     # ridge
W_id = np.linalg.solve(X @ X.T + gamma * np.eye(D), X @ y)
Wmat = W_id.reshape(P_ORDER + 1, M)
U, S, Vt = np.linalg.svd(Wmat)
c_hat = U[:, 0] * S[0]                                    # nonlinearity basis coefficients
xgrid = np.linspace(-1, 1, 200)
f_hat = sum(c_hat[k] * np.polynomial.chebyshev.Chebyshev.basis(k)(xgrid)
            for k in range(P_ORDER + 1))
f_ref = f_true(xgrid * np.max(np.abs(r)))
# resolve Hammerstein scale/sign ambiguity by aligning to the truth for display
A = np.vstack([f_hat, np.ones_like(f_hat)]).T
coef, *_ = np.linalg.lstsq(A, f_ref, rcond=None)
f_hat_aligned = A @ coef

print(f"Input artifact SNR (s vs y)     : {snr_in:6.2f} dB")
print(f"After Stage A (ILC template)    : {snr_A:6.2f} dB   (+{snr_A-snr_in:.2f})")
print(f"After Stage B (Hammerstein RLS) : {snr_B:6.2f} dB   (+{snr_B-snr_A:.2f})")
print(f"R-peaks: true={len(rpeak_idx)}  detected={len(det_r)}")

# ============================================================================
#                               FIGURES
# ============================================================================
w0, w1 = int(20 * fs), int(25 * fs)          # a 5 s window for time plots

fig, ax = plt.subplots(3, 2, figsize=(13, 10))
fig.suptitle("CFA removal: synthetic data + ILC + Hammerstein-RLS", fontsize=14, weight="bold")

ax[0, 0].plot(t[w0:w1], d[w0:w1], lw=0.8)
ax[0, 0].plot(det_r[(det_r > w0) & (det_r < w1)] / fs,
              r[det_r[(det_r > w0) & (det_r < w1)]], "rx", ms=6, label="detected R")
ax[0, 0].set_title("1. Synthetic ECG source d(t) + R-peaks"); ax[0, 0].legend(fontsize=8)

ax[0, 1].plot(t[w0:w1], cfa[w0:w1], lw=0.8, color="C3")
ax[0, 1].set_title("2. CFA = h * f(d)  (nonlinear, then volume-conduction)")

ax[1, 0].plot(t[w0:w1], y[w0:w1], lw=0.7, label="contaminated y")
ax[1, 0].plot(t[w0:w1], s[w0:w1], lw=0.9, label="true neural s")
ax[1, 0].set_title("3. Measured EEG vs true neural"); ax[1, 0].legend(fontsize=8)

ax[1, 1].plot(tmpl_conv, lw=1.4, color="C2", marker="o", ms=3)
ax[1, 1].set_title("4. Stage A: template convergence  ||T_k - T_final||")
ax[1, 1].set_xlabel("beat index (learning phase)")

ax[2, 0].plot(t[w0:w1], s[w0:w1], lw=1.0, label="true neural s")
ax[2, 0].plot(t[w0:w1], clean[w0:w1], lw=0.8, label="cleaned", alpha=0.8)
ax[2, 0].set_title(f"5. Cleaned vs truth  (SNR {snr_in:.1f}->{snr_B:.1f} dB)")
ax[2, 0].legend(fontsize=8); ax[2, 0].set_xlabel("time (s)")

ax[2, 1].plot(xgrid * np.max(np.abs(r)), f_ref, lw=2, label="true f(.)")
ax[2, 1].plot(xgrid * np.max(np.abs(r)), f_hat_aligned, "--", lw=2, label="recovered (Stage B)")
ax[2, 1].set_title("6. Recovered static nonlinearity (shape, up to scale)")
ax[2, 1].legend(fontsize=8); ax[2, 1].set_xlabel("reference amplitude")

for a in ax.flat:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/cfa_time.png", dpi=110)
plt.close()

# spectra: cardiac peak reduced, alpha preserved
f_ax, Pyy = signal.welch(y, fs, nperseg=2048)
_, Pcl = signal.welch(clean, fs, nperseg=2048)
_, Pss = signal.welch(s, fs, nperseg=2048)
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.semilogy(f_ax, Pyy, label="contaminated y", alpha=0.8)
ax2.semilogy(f_ax, Pcl, label="cleaned", lw=2)
ax2.semilogy(f_ax, Pss, label="true neural s", ls=":", lw=2)
ax2.axvspan(9, 11, color="green", alpha=0.12, label="alpha (preserve)")
ax2.set_xlim(0, 45); ax2.set_xlabel("Hz"); ax2.set_ylabel("PSD")
ax2.set_title("Spectra: cardiac harmonics removed, 10 Hz alpha preserved")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("/home/claude/cfa_spectrum.png", dpi=110)
plt.close()
print("figures written")
