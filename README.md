# Porkchop Plot — Interplanetary Transfer Window Tool

An interactive web app for computing and visualising porkchop plots for any
two solar-system bodies using Lambert's problem (Universal Variables method,
Bate–Mueller–White 1971).

Upload JPL Horizons ephemeris CSV files for a departure and arrival body, enter
their names, and the app computes the full delta-V grid and renders a contour
plot. Click any point on the plot to see detailed trajectory parameters for
that departure–arrival pair.

---

## Live Demo

Deployed on Render: **https://orbital-transfers.up.railway.app**

> The free tier sleeps after 15 minutes of inactivity. The first request after
> a sleep takes ~30 seconds to wake up.

---

## Ephemeris Format

Download state vectors from the
[JPL Horizons web interface](https://ssd.jpl.nasa.gov/horizons/) with these
settings:

| Setting | Value |
|---|---|
| Ephemeris Type | Vector Table |
| Reference Frame | ICRF / Ecliptic |
| Output Units | km and km/s |
| CSV Format | Yes |

The uploaded CSV must have columns: `JD, X, Y, Z, VX, VY, VZ`
(J2000 heliocentric ecliptic, km and km/s).

---

## Running Locally

```bash
git clone https://github.com/your-username/porkchop-plot.git
cd porkchop-plot
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## Deploying to Render (free)

1. Fork or push this repo to your GitHub account.
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repo.
4. Set these values:

   | Field | Value |
   |---|---|
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app:app --timeout 120 --workers 1 --threads 4` |
   | **Instance Type** | Free |

5. Click **Deploy**. Render installs dependencies and starts the server.
   Your app will be live at `https://your-app-name.onrender.com`.

Render redeploys automatically on every push to `main`.

---

## Deploying to Railway

1. Push this repo to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**.
3. Select your repo. Railway auto-detects Python and uses the `Procfile`.
4. Done — your URL appears in the dashboard within ~2 minutes.

---

## Algorithm

The Lambert solver uses the Universal Variables parameterisation (Bate,
Mueller & White, 1971). The universal variable $z$ encodes orbit shape
continuously across all conic types ($z < 0$ hyperbolic, $z = 0$ parabolic,
$z > 0$ elliptic). Stumpff functions $C(z)$ and $S(z)$ replace trigonometric
branching with unified series expressions. Newton–Raphson with bisection
fallback drives the time-of-flight residual $F(z) \to 0$ in up to 500
iterations per pair.

The porkchop grid evaluates one Lambert solve per (departure date, arrival
date) pair for all combinations with TOF ≥ 30 days. Results with total ΔV
between 0.1 and 100 km/s are retained; the rest are masked.

---

## References

1. Bate, R.R., Mueller, D.D., White, J.E. *Fundamentals of Astrodynamics*. Dover, 1971.
2. Vallado, D.A. *Fundamentals of Astrodynamics and Applications*, 4th ed. Microcosm Press, 2013.
3. NASA/JPL Horizons System. <https://ssd.jpl.nasa.gov/horizons/>
# orbital_transfers
