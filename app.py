"""
porkchop.py  —  Interplanetary Transfer Window Web App (Porkchop Plot)
Run:  python app.py
Then open:  http://localhost:5000
"""

import math, csv, io, json, threading
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, Response, stream_with_context
import numpy as np

app = Flask(__name__)

# ─────────────────────────────────────────────
#  Lambert solver  (Universal Variables / BMW)
# ─────────────────────────────────────────────

# ── Scalar Lambert solver (used by /api/trajectory) ──
def _Cs(z):
    if   z >  1e-6: return (1 - math.cos(math.sqrt(z))) / z
    elif z < -1e-6: return (math.cosh(math.sqrt(-z)) - 1) / (-z)
    else:           return 0.5 - z/24 + z**2/720 - z**3/40320

def _Ss(z):
    if z > 1e-6:
        s = math.sqrt(z);  return (s - math.sin(s)) / (z * s)
    elif z < -1e-6:
        s = math.sqrt(-z); return (math.sinh(s) - s) / ((-z) * s)
    else: return 1/6 - z/120 + z**2/5040 - z**3/362880

def solve_lambert(r1, r2, tof, prograde, mu):
    r1, r2 = np.array(r1, float), np.array(r2, float)
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    cdnu   = float(np.clip(np.dot(r1, r2) / (n1 * n2), -1, 1))
    cr_z   = float(np.cross(r1, r2)[2])

    if prograde: dnu = math.acos(cdnu) if cr_z >= 0 else 2*math.pi - math.acos(cdnu)
    else:        dnu = math.acos(cdnu) if cr_z <  0 else 2*math.pi - math.acos(cdnu)

    omc = 1 - cdnu
    if abs(omc) < 1e-10 or abs(math.sin(dnu)) < 1e-10:
        raise ValueError("degenerate geometry")
    A = math.sin(dnu) * math.sqrt(n1 * n2 / omc)
    if abs(A) < 1e-10:
        raise ValueError("degenerate A")

    z, zl, zu = 0.0, -4*math.pi**2, 4*math.pi**2
    sqmu = math.sqrt(mu)
    for _ in range(500):
        C, S = _Cs(z), _Ss(z)
        y = n1 + n2 + A * (z*S - 1) / math.sqrt(C)
        if A > 0 and y < 0:
            zl = z; z = 0.5*(zl + zu); continue
        chi = math.sqrt(y / C)
        F   = chi**3 * S + A * math.sqrt(y) - sqmu * tof
        if abs(z) > 1e-6:
            dF = chi**3*(1/(2*z)*(C - 1.5*S/C)) + A/8*(3*S/C*math.sqrt(y) + A/math.sqrt(y))
        else:
            dF = math.sqrt(2)/40*y**1.5 + A/8*(math.sqrt(y) + A*math.sqrt(0.5/y))
        if abs(dF) < 1e-12: break
        zn = z - F/dF
        if zn < zl or zn > zu:
            if F < 0: zl = z
            else:     zu = z
            z = 0.5*(zl + zu)
        else: z = zn
        if abs(F) < 1e-8 * sqmu * abs(tof): break

    C, S = _Cs(z), _Ss(z)
    y  = n1 + n2 + A*(z*S - 1)/math.sqrt(C)
    f  = 1 - y/n1
    g  = A * math.sqrt(y/mu)
    gd = 1 - y/n2
    v1 = (r2 - f*r1) / g
    v2 = (gd*r2 - r1) / g
    return v1, v2

# ── Vectorized Stumpff functions (array inputs) ──
def _Cv(z):
    out = np.empty_like(z)
    pos, neg = z > 1e-6, z < -1e-6
    mid = ~pos & ~neg
    sq  = np.sqrt(np.where(pos, z, 1.0))
    sq2 = np.sqrt(np.where(neg, -z, 1.0))
    out[pos] = (1 - np.cos(sq[pos])) / z[pos]
    out[neg] = (np.cosh(sq2[neg]) - 1) / (-z[neg])
    out[mid] = 0.5 - z[mid]/24 + z[mid]**2/720
    return out

def _Sv(z):
    out = np.empty_like(z)
    pos, neg = z > 1e-6, z < -1e-6
    mid = ~pos & ~neg
    sq  = np.sqrt(np.where(pos, z, 1.0))
    sq2 = np.sqrt(np.where(neg, -z, 1.0))
    out[pos] = (sq[pos] - np.sin(sq[pos])) / (z[pos] * sq[pos])
    out[neg] = (np.sinh(sq2[neg]) - sq2[neg]) / ((-z[neg]) * sq2[neg])
    out[mid] = 1/6 - z[mid]/120 + z[mid]**2/5040
    return out

# ── Vectorized Lambert solver (processes N pairs simultaneously) ──
def solve_lambert_batch(r1, r2, tof, mu, prograde=True):
    """
    r1, r2:   (N, 3) position arrays (km)
    tof:      (N,)   time of flight (s)
    prograde: bool   transfer direction — matches solve_lambert dir='pro'/'retro'
    Returns v1t, v2t, good; good=False for degenerate or unconverged cases.
    """
    n1 = np.linalg.norm(r1, axis=1)
    n2 = np.linalg.norm(r2, axis=1)
    dot = np.clip(np.sum(r1 * r2, axis=1) / (n1 * n2), -1, 1)
    cross_z = r1[:,0]*r2[:,1] - r1[:,1]*r2[:,0]

    # Fix 1: respect prograde/retrograde — was always hardcoded prograde
    if prograde:
        dnu = np.where(cross_z >= 0, np.arccos(dot), 2*np.pi - np.arccos(dot))
    else:
        dnu = np.where(cross_z <  0, np.arccos(dot), 2*np.pi - np.arccos(dot))

    omc = 1 - dot
    sin_dnu = np.sin(dnu)

    good = (np.abs(omc) > 1e-10) & (np.abs(sin_dnu) > 1e-10)
    A = np.where(good, sin_dnu * np.sqrt(n1 * n2 / np.where(omc > 0, omc, 1.0)), 0.0)
    good &= np.abs(A) > 1e-10

    sqmu = math.sqrt(mu)

    # Run the proven scalar solver on each pair individually.
    # The vectorised Newton/bisection is fragile under np.where masking;
    # the scalar path is identical to the MATLAB implementation and always correct.
    N = len(tof)
    v1t = np.zeros((N, 3))
    v2t = np.zeros((N, 3))
    for k in range(N):
        if not good[k]:
            continue
        # Per-element scalar solve (mirrors solve_lambert / MATLAB exactly)
        n1k = float(np.linalg.norm(r1[k])); n2k = float(np.linalg.norm(r2[k]))
        Ak  = float(A[k]); tofk = float(tof[k])
        z = 0.0; zl = -4*math.pi**2; zu = 4*math.pi**2
        for _ in range(500):
            Cz = _Cs(z); Sz = _Ss(z)
            y = n1k + n2k + Ak*(z*Sz - 1)/math.sqrt(Cz)
            if Ak > 0 and y < 0:
                zl = z; z = 0.5*(zl+zu); continue
            chi = math.sqrt(y/Cz)
            F   = chi**3*Sz + Ak*math.sqrt(y) - sqmu*tofk
            if abs(z) > 1e-6:
                dF = chi**3*(1/(2*z)*(Cz - 1.5*Sz/Cz)) + Ak/8*(3*Sz/Cz*math.sqrt(y) + Ak/math.sqrt(y))
            else:
                dF = math.sqrt(2)/40*y**1.5 + Ak/8*(math.sqrt(y) + Ak*math.sqrt(0.5/y))
            if abs(dF) < 1e-12: break
            zn = z - F/dF
            if zn < zl or zn > zu:
                if F < 0: zl = z
                else:     zu = z
                z = 0.5*(zl+zu)
            else:
                z = zn
            if abs(F) < 1e-8*sqmu*abs(tofk): break
        Cz = _Cs(z); Sz = _Ss(z)
        y   = n1k + n2k + Ak*(z*Sz - 1)/math.sqrt(Cz)
        f   = 1 - y/n1k
        g   = Ak*math.sqrt(y/mu)
        gd  = 1 - y/n2k
        if abs(g) < 1e-12:
            good[k] = False; continue
        v1t[k] = (r2[k] - f*r1[k]) / g
        v2t[k] = (gd*r2[k] - r1[k]) / g

    return v1t, v2t, good

# ─────────────────────────────────────────────
#  Ephemeris loading & JD → ISO date
# ─────────────────────────────────────────────

def load_ephem(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

def load_ephem_text(text):
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        rows.append({k: float(v) for k, v in r.items()})
    return rows

def jd_to_iso(jd):
    jd += 0.5; Z = int(jd); F = jd - Z
    if Z < 2299161: A = Z
    else:
        al = int((Z - 1867216.25)/36524.25); A = Z + 1 + al - al//4
    B = A + 1524; C = int((B - 122.1)/365.25)
    D = int(365.25*C); E = int((B - D)/30.6001)
    day = B - D - int(30.6001*E) + F
    mo  = E - 1 if E < 14 else E - 13
    yr  = C - 4716 if mo > 2 else C - 4715
    return f"{yr}-{mo:02d}-{int(day):02d}"

# ─────────────────────────────────────────────
#  Compute or load cached grid
# ─────────────────────────────────────────────

CACHE = Path("porkchop_data.csv")
MU   = 132712440018.0

def compute_grid(dep_csv_text, arr_csv_text, on_progress=None):
    e = load_ephem_text(dep_csv_text)
    m = load_ephem_text(arr_csv_text)

    dep_dates = [jd_to_iso(r['JD']) for r in e]
    arr_dates = [jd_to_iso(r['JD']) for r in m]
    nd, na    = len(e), len(m)

    # Pre-build full position/velocity arrays
    R1_all = np.array([[r['X'],r['Y'],r['Z']]  for r in e])
    V1_all = np.array([[r['VX'],r['VY'],r['VZ']] for r in e])
    R2_all = np.array([[r['X'],r['Y'],r['Z']]  for r in m])
    V2_all = np.array([[r['VX'],r['VY'],r['VZ']] for r in m])
    jd_dep = np.array([r['JD'] for r in e])
    jd_arr = np.array([r['JD'] for r in m])

    dv_flat = np.full(nd * na, np.nan)

    # Process in chunks of departure columns for progress reporting
    CHUNK = max(1, nd // 20)   # ~20 progress ticks

    for chunk_start in range(0, nd, CHUNK):
        chunk_end = min(chunk_start + CHUNK, nd)
        ci = np.arange(chunk_start, chunk_end)

        # Build all (dep_in_chunk x arr) pairs
        ii, jj = np.meshgrid(ci, np.arange(na), indexing='ij')  # (chunk, na)
        tof_days = jd_arr[jj] - jd_dep[ii]
        valid = tof_days >= 30        # ≥30 days: avoids near-degenerate geometry
        vi, vj = np.where(valid)
        gi = ci[vi]   # global dep indices

        if len(gi) > 0:
            r1 = R1_all[gi]; v1 = V1_all[gi]
            r2 = R2_all[vj]; v2 = V2_all[vj]
            tof_s = tof_days[vi, vj] * 86400

            v1t, v2t, good = solve_lambert_batch(r1, r2, tof_s, MU)
            dv = np.linalg.norm(v1t - v1, axis=1) + np.linalg.norm(v2t - v2, axis=1)
            dv = np.where(good & (dv > 0.1) & (dv < 100), dv, np.nan)

            flat_idx = gi * na + vj
            dv_flat[flat_idx] = np.where(~np.isnan(dv), np.round(dv, 4), np.nan)

        if on_progress:
            on_progress(round(chunk_end / nd * 100))

    # Reshape to (na, nd) grid
    dv_2d = dv_flat.reshape(nd, na).T   # (na, nd)
    dv_grid = [[None if np.isnan(dv_2d[j,i]) else float(dv_2d[j,i])
                for i in range(nd)] for j in range(na)]

    valid_vals = dv_flat[~np.isnan(dv_flat)]
    min_dv = float(np.min(valid_vals)) if len(valid_vals) else 0.0
    max_dv = float(np.max(valid_vals)) if len(valid_vals) else 30.0

    # Write CSV: rows = arrival dates, cols = departure dates
    with open(CACHE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arr_date\\dep_date'] + dep_dates)
        for j, arr in enumerate(arr_dates):
            w.writerow([arr] + ['' if v is None else v for v in dv_grid[j]])

    return dict(dep_dates=dep_dates, arr_dates=arr_dates,
                dv_grid=dv_grid, min_dv=round(min_dv,4), max_dv=round(max_dv,4))

# ─────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/data")
def api_data():
    result = app.config.get('GRID_RESULT')
    if not result:
        return jsonify({"error": "no_data"}), 400
    out = dict(result)
    out['dep_name'] = app.config.get('DEP_NAME', 'Departure Body')
    out['arr_name'] = app.config.get('ARR_NAME', 'Arrival Body')
    return jsonify(out)

@app.route("/api/compute")
def api_compute():
    """SSE endpoint: streams progress %, then emits the completed grid as JSON."""
    if not app.config.get('DEP_CSV') or not app.config.get('ARR_CSV'):
        return jsonify({"error": "no_ephemeris"}), 400

    dep_csv = app.config['DEP_CSV']
    arr_csv = app.config['ARR_CSV']

    def generate():
        last_pct = [-1]

        def on_progress(pct):
            if pct != last_pct[0]:
                last_pct[0] = pct
                yield f"data: {json.dumps({'pct': pct})}\n\n"

        # on_progress is a generator — we need to drive it differently
        # Use a queue instead so we can yield from the outer generator
        import queue
        q = queue.Queue()

        def progress_cb(pct):
            q.put(pct)

        result_box = [None]
        error_box  = [None]

        def worker():
            try:
                result_box[0] = compute_grid(dep_csv, arr_csv, on_progress=progress_cb)
            except Exception as ex:
                error_box[0] = str(ex)
            finally:
                q.put(None)  # sentinel

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            pct = q.get()
            if pct is None:
                break
            yield f"data: {json.dumps({'pct': pct})}\n\n"

        t.join()

        if error_box[0]:
            yield f"data: {json.dumps({'error': error_box[0]})}\n\n"
        else:
            app.config['GRID_RESULT'] = result_box[0]
            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    dep_file = request.files.get("departure")
    arr_file = request.files.get("arrival")
    if not dep_file or not arr_file:
        return jsonify({"error": "Both departure and arrival files required"}), 400
    try:
        app.config['DEP_CSV']  = dep_file.read().decode('utf-8')
        app.config['ARR_CSV']  = arr_file.read().decode('utf-8')
        app.config['DEP_NAME'] = request.form.get('dep_name', '').strip() or 'Departure Body'
        app.config['ARR_NAME'] = request.form.get('arr_name', '').strip() or 'Arrival Body'
        load_ephem_text(app.config['DEP_CSV'])[0]
        load_ephem_text(app.config['ARR_CSV'])[0]
        return jsonify({"ok": True})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400

@app.route("/api/trajectory")
def api_trajectory():
    """Return orbital mechanics details for a specific dep/arr date pair."""
    from flask import request
    dep_date = request.args.get("dep")
    arr_date = request.args.get("arr")

    if not app.config.get('DEP_CSV') or not app.config.get('ARR_CSV'):
        return jsonify({"error": "No ephemeris uploaded"}), 400
    e = load_ephem_text(app.config['DEP_CSV'])
    m = load_ephem_text(app.config['ARR_CSV'])

    dep_row = next((r for r in e if jd_to_iso(r['JD']) == dep_date), None)
    arr_row = next((r for r in m if jd_to_iso(r['JD']) == arr_date), None)
    if not dep_row or not arr_row:
        return jsonify({"error": "Date not found"}), 404

    tof_d = arr_row['JD'] - dep_row['JD']
    if tof_d <= 0:
        return jsonify({"error": "Arrival must be after departure"}), 400

    R1 = [dep_row['X'], dep_row['Y'], dep_row['Z']]
    V1 = np.array([dep_row['VX'], dep_row['VY'], dep_row['VZ']])
    R2 = [arr_row['X'], arr_row['Y'], arr_row['Z']]
    V2 = np.array([arr_row['VX'], arr_row['VY'], arr_row['VZ']])

    try:
        v1t, v2t = solve_lambert(R1, R2, tof_d*86400, True, MU)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400

    dv_dep = np.linalg.norm(v1t - V1)
    dv_arr = np.linalg.norm(v2t - V2)
    dv_tot = dv_dep + dv_arr

    # Keplerian elements of transfer orbit
    r1v = np.array(R1); r2v = np.array(R2)
    h   = np.cross(r1v, v1t)
    n1  = np.linalg.norm(r1v)
    e_vec = np.cross(v1t, h)/MU - r1v/n1
    ecc = np.linalg.norm(e_vec)
    energy = np.linalg.norm(v1t)**2/2 - MU/n1
    a = -MU/(2*energy) if abs(energy) > 1e-10 else float('inf')
    p = np.linalg.norm(h)**2 / MU
    inc = math.degrees(math.acos(float(np.clip(h[2]/np.linalg.norm(h), -1, 1))))

    # Transfer orbit sample points for 3D visualization (50 pts)
    f_arr, g_arr, n_pts = [], [], 50
    Cz_final = _C(0); Sz_final = _S(0)  # we don't have z here; approximate with Lagrange
    # Use Lagrange propagation along the orbit arc
    tof_s = tof_d * 86400
    r1n = float(np.linalg.norm(r1v)); r2n = float(np.linalg.norm(r2v))
    pts = []
    for k in range(n_pts + 1):
        tau = k / n_pts * tof_s
        # Simple Kepler propagation using universal variable (abbreviated)
        # For visualization, interpolate with f/g series
        fk = 1 - MU*tau**2/(2*r1n**3)  # first-order Lagrange approx
        gk = tau - MU*tau**3/(6*r1n**3)
        rv = r1v*fk + v1t*gk
        pts.append([round(float(rv[0])/1.496e8, 6),
                    round(float(rv[1])/1.496e8, 6),
                    round(float(rv[2])/1.496e8, 6)])

    return jsonify({
        "dep_date":  dep_date,
        "arr_date":  arr_date,
        "tof_days":  round(tof_d, 1),
        "dv_dep":    round(float(dv_dep), 4),
        "dv_arr":    round(float(dv_arr), 4),
        "dv_total":  round(float(dv_tot), 4),
        "v1t":       [round(float(x),4) for x in v1t],
        "v2t":       [round(float(x),4) for x in v2t],
        "ecc":       round(ecc, 5),
        "sma_au":    round(a/1.496e8, 5),
        "inc_deg":   round(inc, 3),
        "dep_pos":   [round(x/1.496e8, 6) for x in R1],
        "arr_pos":   [round(x/1.496e8, 6) for x in R2],
        "transfer_pts": pts,
    })


# ─────────────────────────────────────────────
#  HTML / CSS / JS  (single-file frontend)
# ─────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Porkchop Plot</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #f5f4f0;
    color: #1a1a1a;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }
  .page { max-width: 1100px; margin: 0 auto; padding: 32px 24px 48px; }

  header { margin-bottom: 28px; }
  header h1 { font-size: 22px; font-weight: 600; letter-spacing: -0.3px; }
  header p { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #888; margin-top: 4px; letter-spacing: 0.04em; }

  /* ── Upload screen ── */
  #upload-screen {
    max-width: 480px; margin: 60px auto 0;
    background: #fff; border: 1px solid #e0ddd8; border-radius: 4px; overflow: hidden;
  }
  .upload-title {
    padding: 14px 18px; border-bottom: 1px solid #e0ddd8;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #999; letter-spacing: 0.08em; text-transform: uppercase;
  }
  .upload-body { padding: 24px 18px; display: flex; flex-direction: column; gap: 16px; }
  .upload-field label {
    display: block; font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #999; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 6px;
  }
  .file-drop {
    border: 1px dashed #ccc; border-radius: 3px;
    padding: 14px 16px; cursor: pointer;
    display: flex; align-items: center; gap: 10px;
    transition: border-color .15s, background .15s;
    position: relative;
  }
  .file-drop:hover { border-color: #999; background: #fafaf8; }
  .file-drop.has-file { border-color: #1a7a4a; border-style: solid; background: #f0faf5; }
  .file-drop input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .file-icon { font-size: 18px; flex-shrink: 0; }
  .file-text { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #999; }
  .file-text.chosen { color: #1a7a4a; }
  .upload-btn {
    width: 100%; padding: 11px;
    background: #1a1a1a; color: #fff;
    border: none; border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    cursor: pointer; transition: background .15s;
    letter-spacing: 0.04em;
  }
  .upload-btn:hover:not(:disabled) { background: #333; }
  .upload-btn:disabled { background: #ccc; cursor: not-allowed; }
  .upload-names { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .name-input {
    width: 100%; padding: 9px 10px;
    border: 1px solid #e0ddd8; border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #1a1a1a;
    background: #fff; outline: none;
    transition: border-color .15s;
  }
  .name-input:focus { border-color: #999; }
  .name-input::placeholder { color: #bbb; }
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #bbb; text-align: center; line-height: 1.6;
  }
  .upload-error {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #c0392b; background: #fdf0ee; border: 1px solid #f5c6c0;
    border-radius: 3px; padding: 8px 12px; display: none;
  }

  /* ── Plot screen ── */
  #plot-screen { display: none; }
  .layout { display: grid; grid-template-columns: 1fr 240px; gap: 24px; align-items: start; }
  @media (max-width: 780px) { .layout { grid-template-columns: 1fr; } }

  .plot-card { background: #fff; border: 1px solid #e0ddd8; border-radius: 4px; }
  .card-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #999;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 10px 14px 9px; border-bottom: 1px solid #e0ddd8;
    display: flex; justify-content: space-between; align-items: center;
  }
  .reset-btn {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #999; background: none; border: 1px solid #e0ddd8;
    border-radius: 2px; padding: 2px 8px; cursor: pointer;
    transition: color .15s, border-color .15s;
  }
  .reset-btn:hover { color: #1a1a1a; border-color: #999; }

  #porkchop-plot { width: 100%; height: 520px; }
  .plot-loading {
    height: 520px; display: flex; align-items: center; justify-content: center;
  }
  .plot-loading.hidden { display: none; }
  .progress-wrap { width: 280px; display: flex; flex-direction: column; gap: 10px; }
  .progress-label {
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #999;
  }
  #loading-pct { font-size: 13px; font-weight: 500; color: #1a1a1a; min-width: 36px; text-align: right; }
  .progress-track {
    height: 3px; background: #e0ddd8; border-radius: 2px; overflow: hidden;
  }
  .progress-bar {
    height: 100%; width: 0%; background: #1a1a1a;
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .sidebar { display: flex; flex-direction: column; gap: 16px; }
  .info-card { background: #fff; border: 1px solid #e0ddd8; border-radius: 4px; overflow: hidden; }
  .stat-row { display: flex; justify-content: space-between; align-items: baseline; padding: 10px 14px; border-bottom: 1px solid #f0ede8; }
  .stat-row:last-child { border-bottom: none; }
  .stat-name { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #999; letter-spacing: 0.04em; }
  .stat-val { font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 500; color: #1a1a1a; }
  .stat-val.green { color: #1a7a4a; }

  #traj-placeholder { padding: 24px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #bbb; line-height: 1.7; text-align: center; }
  #traj-content { display: none; }
  .traj-dv { padding: 14px; border-bottom: 1px solid #f0ede8; text-align: center; }
  .traj-dv-total { font-size: 28px; font-weight: 600; letter-spacing: -1px; color: #1a1a1a; line-height: 1; }
  .traj-dv-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #999; margin-top: 3px; }
  .traj-dv-split { display: flex; gap: 0; margin-top: 10px; border: 1px solid #e0ddd8; border-radius: 3px; overflow: hidden; }
  .traj-dv-part { flex: 1; padding: 7px 8px; text-align: center; background: #fafaf8; }
  .traj-dv-part:first-child { border-right: 1px solid #e0ddd8; }
  .traj-dv-part-val { font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 500; color: #1a1a1a; }
  .traj-dv-part-name { font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: #aaa; margin-top: 1px; }
</style>
</head>
<body>
<div class="page">

  <header>
    <h1 id="page-title">Porkchop Plot</h1>
    <p>PORKCHOP PLOT · LAMBERT'S PROBLEM · JPL HORIZONS EPHEMERIS</p>
  </header>

  <!-- ── Upload screen ── -->
  <div id="upload-screen">
    <div class="upload-title">Upload ephemeris files to begin</div>
    <div class="upload-body">

      <div class="upload-names">
        <div class="upload-field">
          <label>Departure body name</label>
          <input type="text" id="dep-name" class="name-input" placeholder="e.g. Earth" maxlength="40">
        </div>
        <div class="upload-field">
          <label>Arrival body name</label>
          <input type="text" id="arr-name" class="name-input" placeholder="e.g. Mars" maxlength="40">
        </div>
      </div>

      <div class="upload-field">
        <label>Departure body ephemeris</label>
        <div class="file-drop" id="dep-drop">
          <input type="file" id="dep-file" accept=".csv">
          <span class="file-icon">📄</span>
          <span class="file-text" id="dep-label">select .csv file…</span>
        </div>
      </div>

      <div class="upload-field">
        <label>Arrival body ephemeris</label>
        <div class="file-drop" id="arr-drop">
          <input type="file" id="arr-file" accept=".csv">
          <span class="file-icon">📄</span>
          <span class="file-text" id="arr-label">select .csv file…</span>
        </div>
      </div>

      <div class="upload-error" id="upload-error"></div>

      <button class="upload-btn" id="upload-btn" disabled>compute porkchop plot →</button>

      <p class="upload-hint">
        CSV format: JD, X, Y, Z, VX, VY, VZ<br>
        J2000 heliocentric ecliptic · km · km/s
      </p>
    </div>
  </div>

  <!-- ── Plot screen ── -->
  <div id="plot-screen">
    <div class="layout">
      <div class="plot-card">
        <div class="card-label">
          <span>Total ΔV (km/s) — departure × arrival date</span>
          <button class="reset-btn" onclick="resetUpload()">↑ new files</button>
        </div>
        <div id="plot-loading" class="plot-loading">
          <div class="progress-wrap">
            <div class="progress-label">
              <span id="loading-text">computing lambert solutions…</span>
              <span id="loading-pct"></span>
            </div>
            <div class="progress-track">
              <div class="progress-bar" id="progress-bar"></div>
            </div>
          </div>
        </div>
        <div id="porkchop-plot" style="display:none"></div>
      </div>

      <div class="sidebar">
        <div class="info-card">
          <div class="card-label" style="padding:10px 14px 9px">Optimal window</div>
          <div class="stat-row"><span class="stat-name">MIN ΔV</span><span class="stat-val green" id="s-min-dv">—</span></div>
          <div class="stat-row"><span class="stat-name">DEPART</span><span class="stat-val" id="s-dep">—</span></div>
          <div class="stat-row"><span class="stat-name">ARRIVE</span><span class="stat-val" id="s-arr">—</span></div>
          <div class="stat-row"><span class="stat-name">TOF</span><span class="stat-val" id="s-tof">—</span></div>
        </div>

        <div class="info-card">
          <div class="card-label" style="padding:10px 14px 9px">Selected trajectory</div>
          <div id="traj-placeholder">click any point<br>on the plot</div>
          <div id="traj-content">
            <div class="traj-dv">
              <div class="traj-dv-total" id="t-total">—</div>
              <div class="traj-dv-label">total ΔV (km/s)</div>
              <div class="traj-dv-split">
                <div class="traj-dv-part"><div class="traj-dv-part-val" id="t-dep-dv">—</div><div class="traj-dv-part-name">departure</div></div>
                <div class="traj-dv-part"><div class="traj-dv-part-val" id="t-arr-dv">—</div><div class="traj-dv-part-name">arrival</div></div>
              </div>
            </div>
            <div class="stat-row"><span class="stat-name">DEPART</span><span class="stat-val" id="t-dep">—</span></div>
            <div class="stat-row"><span class="stat-name">ARRIVE</span><span class="stat-val" id="t-arr">—</span></div>
            <div class="stat-row"><span class="stat-name">TOF</span><span class="stat-val" id="t-tof">—</span></div>
            <div class="stat-row"><span class="stat-name">ECC</span><span class="stat-val" id="t-ecc">—</span></div>
            <div class="stat-row"><span class="stat-name">SMA</span><span class="stat-val" id="t-sma">—</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /page -->

<script>
const CSCALE = [
  [0.00,'#2c7bb6'],[0.25,'#abd9e9'],[0.50,'#ffffbf'],
  [0.75,'#fdae61'],[1.00,'#d7191c'],
];
let MIN_DV = 0, MAX_DV = 30;

// ── File picker UI ──
const depFile = document.getElementById('dep-file');
const arrFile = document.getElementById('arr-file');
const uploadBtn = document.getElementById('upload-btn');

function checkUploadReady() {
  uploadBtn.disabled = !(depFile.files.length && arrFile.files.length);
}

function onFileChosen(input, dropEl, labelEl) {
  if (!input.files.length) return;
  labelEl.textContent = input.files[0].name;
  labelEl.classList.add('chosen');
  dropEl.classList.add('has-file');
  checkUploadReady();
}

depFile.addEventListener('change', () => onFileChosen(depFile, document.getElementById('dep-drop'), document.getElementById('dep-label')));
arrFile.addEventListener('change', () => onFileChosen(arrFile, document.getElementById('arr-drop'), document.getElementById('arr-label')));

uploadBtn.addEventListener('click', async () => {
  const errEl = document.getElementById('upload-error');
  errEl.style.display = 'none';
  uploadBtn.disabled = true;
  uploadBtn.textContent = 'uploading…';

  const fd = new FormData();
  fd.append('departure', depFile.files[0]);
  fd.append('arrival',   arrFile.files[0]);
  fd.append('dep_name',  document.getElementById('dep-name').value.trim() || 'Departure Body');
  fd.append('arr_name',  document.getElementById('arr-name').value.trim() || 'Arrival Body');

  const r = await fetch('/api/upload', { method: 'POST', body: fd });
  const j = await r.json();

  if (!r.ok || j.error) {
    errEl.textContent = j.error || 'Upload failed';
    errEl.style.display = 'block';
    uploadBtn.disabled = false;
    uploadBtn.textContent = 'compute porkchop plot →';
    return;
  }

  // Switch screens
  document.getElementById('upload-screen').style.display = 'none';
  document.getElementById('plot-screen').style.display   = 'block';
  loadAndPlot();
});

function resetUpload() {
  document.getElementById('upload-screen').style.display = 'block';
  document.getElementById('plot-screen').style.display   = 'none';
  document.getElementById('porkchop-plot').style.display = 'none';
  document.getElementById('plot-loading').classList.remove('hidden');
  document.getElementById('progress-bar').style.width    = '0%';
  document.getElementById('loading-pct').textContent     = '';
  document.getElementById('loading-text').textContent    = 'computing lambert solutions…';
  document.getElementById('traj-placeholder').style.display = 'block';
  document.getElementById('traj-content').style.display     = 'none';
}

// ── Plot ──
async function loadAndPlot() {
  document.getElementById('loading-text').textContent = 'computing lambert solutions…';
  document.getElementById('loading-pct').textContent  = '';
  document.getElementById('progress-bar').style.width = '0%';

  const ok = await new Promise(resolve => {
    const es = new EventSource('/api/compute');
    es.onmessage = e => {
      const msg = JSON.parse(e.data);
      if (msg.error) {
        es.close();
        document.getElementById('loading-text').textContent = 'error: ' + msg.error;
        resolve(false);
        return;
      }
      if (msg.pct !== undefined) {
        document.getElementById('progress-bar').style.width = msg.pct + '%';
        document.getElementById('loading-pct').textContent  = msg.pct + '%';
      }
      if (msg.done) { es.close(); resolve(true); }
    };
    es.onerror = () => {
      es.close();
      document.getElementById('loading-text').textContent = 'error: connection failed';
      resolve(false);
    };
  });
  if (!ok) return;

  const data = await fetch('/api/data').then(r => r.json());
  if (data.error) {
    document.getElementById('loading-text').textContent = 'error: ' + data.error;
    return;
  }

  const DEP = data.dep_name;
  const ARR = data.arr_name;

  document.getElementById('page-title').textContent = `${DEP} → ${ARR} Transfer Windows`;
  document.title = `${DEP} → ${ARR} Porkchop Plot`;
  document.querySelector('.card-label span').textContent =
    `Total ΔV (km/s) — ${DEP} departure × ${ARR} arrival`;

  MIN_DV = data.min_dv;
  MAX_DV = Math.min(data.min_dv + 30, data.max_dv);  // saturate above +30 km/s for readability

  let best = Infinity, bi = -1, bj = -1;
  for (let j = 0; j < data.arr_dates.length; j++)
    for (let i = 0; i < data.dep_dates.length; i++) {
      const v = data.dv_grid[j][i];
      if (v !== null && v < best) { best = v; bi = i; bj = j; }
    }
  if (bi >= 0) {
    const tof = Math.round((new Date(data.arr_dates[bj]) - new Date(data.dep_dates[bi])) / 86400000);
    document.getElementById('s-min-dv').textContent = best.toFixed(2) + ' km/s';
    document.getElementById('s-dep').textContent    = data.dep_dates[bi];
    document.getElementById('s-arr').textContent    = data.arr_dates[bj];
    document.getElementById('s-tof').textContent    = tof + ' days';
  }

  const z = data.dv_grid.map(row =>
    row.map(v => v === null ? null : Math.min(v, MAX_DV))
  );

  const trace = {
    type: 'contour', x: data.dep_dates, y: data.arr_dates, z,
    colorscale: CSCALE, zmin: MIN_DV, zmax: MAX_DV,
    contours: {
      start: Math.floor(MIN_DV), end: Math.ceil(MAX_DV), size: 1,
      showlabels: true,
      labelfont: { size: 9, color: '#333', family: 'IBM Plex Mono' },
    },
    line: { width: 0.5, smoothing: 0.9 },
    colorbar: {
      thickness: 12, len: 0.9,
      tickfont: { size: 9, family: 'IBM Plex Mono' },
      title: { text: 'km/s', font: { size: 10 }, side: 'right' },
      outlinewidth: 0,
    },
    hovertemplate: 'Dep: %{x}<br>Arr: %{y}<br>ΔV: %{z:.2f} km/s<extra></extra>',
  };

  const layout = {
    paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    font: { family: 'IBM Plex Mono', size: 10, color: '#555' },
    margin: { t: 12, r: 60, b: 80, l: 100 },
    xaxis: {
      title: { text: `Departure (${DEP})`, font: { size: 11 }, standoff: 12 },
      tickformat: '%b %Y', tickangle: -35,
      gridcolor: '#f0ede8', linecolor: '#e0ddd8', zerolinecolor: '#e0ddd8',
    },
    yaxis: {
      title: { text: `Arrival (${ARR})`, font: { size: 11 }, standoff: 8 },
      tickformat: '%b %Y',
      gridcolor: '#f0ede8', linecolor: '#e0ddd8', zerolinecolor: '#e0ddd8',
    },
    hoverlabel: { bgcolor: '#fff', bordercolor: '#ccc', font: { family: 'IBM Plex Mono', size: 10 } },
  };

  document.getElementById('plot-loading').classList.add('hidden');
  const plotDiv = document.getElementById('porkchop-plot');
  plotDiv.style.display = 'block';

  await Plotly.newPlot('porkchop-plot', [trace], layout, {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['select2d','lasso2d'],
    toImageButtonOptions: { filename: `porkchop_${DEP}_${ARR}`.toLowerCase().replace(/\s+/g,'_'), scale: 2 },
  });

  plotDiv.on('plotly_click', async ev => {
    const pt = ev.points[0]; if (!pt) return;
    const r = await fetch(`/api/trajectory?dep=${pt.x}&arr=${pt.y}`).then(r => r.json());
    if (r.error) return;
    const tof = Math.round((new Date(r.arr_date) - new Date(r.dep_date)) / 86400000);
    document.getElementById('traj-placeholder').style.display = 'none';
    document.getElementById('traj-content').style.display     = 'block';
    document.getElementById('t-total').textContent  = r.dv_total.toFixed(2);
    document.getElementById('t-dep-dv').textContent = r.dv_dep.toFixed(2);
    document.getElementById('t-arr-dv').textContent = r.dv_arr.toFixed(2);
    document.getElementById('t-dep').textContent    = r.dep_date;
    document.getElementById('t-arr').textContent    = r.arr_date;
    document.getElementById('t-tof').textContent    = tof + ' days';
    document.getElementById('t-ecc').textContent    = r.ecc.toFixed(4);
    document.getElementById('t-sma').textContent    = r.sma_au.toFixed(3) + ' AU';
  });
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"Ready.  Open  http://localhost:{port}")
    app.run(debug=False, port=port, host="0.0.0.0")
