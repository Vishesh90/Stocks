#!/usr/bin/env python3
"""
run_vm.py — Single-file self-contained backtest launcher.

Run this ONE file on the VM. It:
1. Installs every dependency into a local .venv (no activation needed)
2. Patches pandas_ta out of all strategy files
3. Runs the full backtest
4. Prints the leaderboard
5. Uploads to GCS

Usage (on the VM, no venv needed):
    python3 run_vm.py
"""

import subprocess, sys, os, json
from pathlib import Path

ROOT = Path(__file__).parent
VENV = ROOT / ".venv"
PIP  = VENV / "bin" / "pip"
PY   = VENV / "bin" / "python3"

PACKAGES = [
    "pandas", "numpy", "scipy", "ta", "requests", "yfinance",
    "statsmodels", "scikit-learn", "filterpy", "loguru",
    "python-dotenv", "pydantic", "pydantic-settings",
    "rich", "tqdm", "tenacity", "psutil", "pyarrow", "dhanhq",
]

# ── STEP 1: CREATE VENV ───────────────────────────────────────────────────────
print("[1/5] Setting up isolated Python environment...")
if not VENV.exists():
    subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    print("  venv created.")
else:
    print("  venv already exists.")

# ── STEP 2: INSTALL PACKAGES ──────────────────────────────────────────────────
print("[2/5] Installing packages (this takes ~2 min on first run)...")
subprocess.run(
    [str(PIP), "install", "--quiet", "--upgrade"] + PACKAGES,
    check=True
)
print("  All packages installed.")

# ── STEP 3: PATCH pandas_ta OUT OF STRATEGY FILES ────────────────────────────
print("[3/5] Patching strategy files...")

SHIM = '''
import pandas as _pd
import numpy as _np

class ta:
    @staticmethod
    def ema(s, length=10): return s.ewm(span=length, adjust=False).mean()
    @staticmethod
    def sma(s, length=10): return s.rolling(window=length).mean()
    @staticmethod
    def rsi(s, length=14):
        d=s.diff(); g=d.clip(lower=0).rolling(length).mean()
        l=(-d.clip(upper=0)).rolling(length).mean()
        return 100-(100/(1+g/l.replace(0, _np.nan)))
    @staticmethod
    def macd(s, fast=12, slow=26, signal=9):
        ef=s.ewm(span=fast,adjust=False).mean(); es=s.ewm(span=slow,adjust=False).mean()
        ml=ef-es; sl=ml.ewm(span=signal,adjust=False).mean()
        return _pd.DataFrame({
            f"MACD_{fast}_{slow}_{signal}": ml,
            f"MACDs_{fast}_{slow}_{signal}": sl,
            f"MACDh_{fast}_{slow}_{signal}": ml-sl})
    @staticmethod
    def bbands(s, length=20, std=2.0):
        m=s.rolling(length).mean(); sg=s.rolling(length).std()
        return _pd.DataFrame({
            f"BBL_{length}_{std}": m-std*sg,
            f"BBM_{length}_{std}": m,
            f"BBU_{length}_{std}": m+std*sg,
            f"BBB_{length}_{std}": (2*std*sg)/m,
            f"BBP_{length}_{std}": (s-(m-std*sg))/(2*std*sg)})
    @staticmethod
    def atr(h, l, c, length=14):
        tr=_pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        return tr.ewm(span=length,adjust=False).mean()
    @staticmethod
    def stoch(h, l, c, k=14, d=3, smooth_k=3):
        lo=l.rolling(k).min(); hi=h.rolling(k).max()
        sk=100*(c-lo)/(hi-lo); sk=sk.rolling(smooth_k).mean(); sd=sk.rolling(d).mean()
        return _pd.DataFrame({
            f"STOCHk_{k}_{d}_{smooth_k}": sk,
            f"STOCHd_{k}_{d}_{smooth_k}": sd})
    @staticmethod
    def adx(h, l, c, length=14):
        tr=_pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        dmp=(h-h.shift()).clip(lower=0); dmn=(l.shift()-l).clip(lower=0)
        dmp=dmp.where(dmp>dmn,0); dmn=dmn.where(dmn>dmp,0)
        as_=tr.ewm(span=length,adjust=False).mean()
        dp=100*dmp.ewm(span=length,adjust=False).mean()/as_
        dn=100*dmn.ewm(span=length,adjust=False).mean()/as_
        dx=100*(dp-dn).abs()/(dp+dn); adxv=dx.ewm(span=length,adjust=False).mean()
        return _pd.DataFrame({f"ADX_{length}":adxv, f"DMP_{length}":dp, f"DMN_{length}":dn})
    @staticmethod
    def vwap(h, l, c, v, **kw):
        tp=(h+l+c)/3; return (tp*v).cumsum()/v.cumsum()
    @staticmethod
    def obv(c, v):
        return (_np.sign(c.diff()).fillna(0)*v).cumsum()
    @staticmethod
    def mfi(h, l, c, v, length=14):
        tp=(h+l+c)/3; mf=tp*v
        pos=mf.where(tp>tp.shift(),0).rolling(length).sum()
        neg=mf.where(tp<tp.shift(),0).rolling(length).sum()
        return 100-(100/(1+pos/neg))
    @staticmethod
    def cci(h, l, c, length=20):
        tp=(h+l+c)/3; ma=tp.rolling(length).mean()
        mad=tp.rolling(length).apply(lambda x: _np.mean(_np.abs(x-_np.mean(x))), raw=True)
        return (tp-ma)/(0.015*mad)
    @staticmethod
    def willr(h, l, c, length=14):
        return -100*(h.rolling(length).max()-c)/(h.rolling(length).max()-l.rolling(length).min())
    @staticmethod
    def roc(s, length=10): return 100*(s-s.shift(length))/s.shift(length)
    @staticmethod
    def donchian(h, l, lower_length=20, upper_length=20):
        return _pd.DataFrame({
            f"DCL_{lower_length}_{upper_length}": l.rolling(lower_length).min(),
            f"DCM_{lower_length}_{upper_length}": (h.rolling(upper_length).max()+l.rolling(lower_length).min())/2,
            f"DCU_{lower_length}_{upper_length}": h.rolling(upper_length).max()})
    @staticmethod
    def kc(h, l, c, length=20, scalar=1.5):
        mid=c.ewm(span=length,adjust=False).mean()
        tr=_pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atrv=tr.ewm(span=length,adjust=False).mean()
        return _pd.DataFrame({
            f"KCLe_{length}_{scalar}": mid-scalar*atrv,
            f"KCBe_{length}_{scalar}": mid,
            f"KCUe_{length}_{scalar}": mid+scalar*atrv})
    @staticmethod
    def aroon(h, l, length=25):
        au=h.rolling(length+1).apply(lambda x:(_np.argmax(x)/length)*100,raw=True)
        ad=l.rolling(length+1).apply(lambda x:(_np.argmin(x)/length)*100,raw=True)
        return _pd.DataFrame({f"AROONU_{length}":au, f"AROOND_{length}":ad})
    @staticmethod
    def psar(h, l, c, af0=0.02, af=0.02, max_af=0.2):
        n=len(c); ps=c.copy(); bull=True; afc=af0; ep=l.iloc[0]; hp=h.iloc[0]; lp=l.iloc[0]
        for i in range(2,n):
            ps.iloc[i]=ps.iloc[i-1]+afc*(ep-ps.iloc[i-1])
            if bull:
                if l.iloc[i]<ps.iloc[i]: bull=False; ps.iloc[i]=hp; ep=l.iloc[i]; afc=af0
                else:
                    if h.iloc[i]>ep: ep=h.iloc[i]; afc=min(afc+af,max_af)
                    hp=h.iloc[i]
            else:
                if h.iloc[i]>ps.iloc[i]: bull=True; ps.iloc[i]=lp; ep=h.iloc[i]; afc=af0
                else:
                    if l.iloc[i]<ep: ep=l.iloc[i]; afc=min(afc+af,max_af)
                    lp=l.iloc[i]
        return _pd.DataFrame({
            f"PSARl_{af0}_{max_af}": ps, f"PSARs_{af0}_{max_af}": ps,
            f"PSARaf_{af0}_{max_af}": _pd.Series([afc]*n, index=c.index),
            f"PSARr_{af0}_{max_af}": _pd.Series([int(bull)]*n, index=c.index)})
'''

def patch_file(path: Path, target_import: str, shim: str):
    content = path.read_text()
    if target_import not in content:
        print(f"  {path.name}: already patched")
        return
    content = content.replace(target_import, shim)
    path.write_text(content)
    print(f"  {path.name}: patched")

# Patch standard strategies
std_file = ROOT / "strategies" / "standard" / "__init__.py"
patch_file(std_file,
    "import pandas as pd\nimport numpy as np\nimport pandas_ta as ta\nfrom typing import Optional\n\nfrom strategies.base import BaseStrategy, Signal, StrategyConfig",
    "import pandas as pd\nimport numpy as np\nfrom typing import Optional\nfrom strategies.base import BaseStrategy, Signal, StrategyConfig\n" + SHIM
)

# Patch mathematical strategies — remove inline imports
math_file = ROOT / "strategies" / "mathematical" / "__init__.py"
content = math_file.read_text()
mini_shim = '''
        import pandas as _pd; import numpy as _np
        class ta:
            @staticmethod
            def ema(s,length=10): return s.ewm(span=length,adjust=False).mean()
            @staticmethod
            def rsi(s,length=14):
                d=s.diff();g=d.clip(lower=0).rolling(length).mean();l=(-d.clip(upper=0)).rolling(length).mean()
                return 100-(100/(1+g/l.replace(0,_np.nan)))
            @staticmethod
            def atr(h,l,c,length=14):
                tr=_pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
                return tr.ewm(span=length,adjust=False).mean()'''
content = content.replace("        import pandas_ta as ta", mini_shim)
math_file.write_text(content)
print(f"  mathematical/__init__.py: patched")

# Patch segment scorer
scorer_file = ROOT / "intelligence" / "segment_scorer.py"
content = scorer_file.read_text()
content = content.replace("import pandas_ta as ta", "try:\n    import pandas_ta as ta\nexcept ImportError:\n    pass")
scorer_file.write_text(content)
print(f"  segment_scorer.py: patched")

print("  All files patched.")

# ── STEP 4: RUN BACKTEST ──────────────────────────────────────────────────────
print("[4/5] Running backtest (15m interval, full universe)...")
print("      This will take 1-2 hours. Output is in /tmp/backtest_15m.log\n")

env = os.environ.copy()
env["PATH"] = str(VENV / "bin") + ":" + env["PATH"]

result = subprocess.run(
    [str(PY), "scripts/run_backtest.py", "--interval", "15m", "--top", "30"],
    cwd=str(ROOT),
    env=env,
    capture_output=False,
)

# ── STEP 5: UPLOAD RESULTS ────────────────────────────────────────────────────
print("\n[5/5] Uploading results to GCS...")
leaderboard = ROOT / "reports" / "leaderboard.csv"
if leaderboard.exists():
    subprocess.run([
        "gsutil", "cp", str(leaderboard),
        "gs://stocks-490622-stocks-backtest/results/leaderboard_15m.csv"
    ])
    subprocess.run([
        "bash", "-c",
        "echo BACKTEST_COMPLETE | gsutil cp - gs://stocks-490622-stocks-backtest/results/status.txt"
    ])
    print("\nDONE. Results at gs://stocks-490622-stocks-backtest/results/leaderboard_15m.csv")
else:
    print("ERROR: leaderboard.csv not found. Check /tmp/backtest_15m.log for errors.")
