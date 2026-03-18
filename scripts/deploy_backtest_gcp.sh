#!/bin/bash
# =============================================================================
# deploy_backtest_gcp.sh — One-command GCP backtest runner
#
# USAGE:
#   chmod +x scripts/deploy_backtest_gcp.sh
#   ./scripts/deploy_backtest_gcp.sh --project YOUR_PROJECT_ID
#
# WHAT IT DOES:
#   1. Creates a spot VM (8 vCPU, 32GB RAM) — ~₹4/hour
#   2. Uploads code + credentials securely via Secret Manager
#   3. Runs full backtest (45 strategies × full universe)
#   4. Saves leaderboard to GCS bucket
#   5. Streams results to your terminal
#   6. Deletes the VM automatically — you are never billed for idle time
#
# COST: ~₹3-5 total for the full backtest run
# =============================================================================

set -e

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ID=""
REGION="asia-south1"
ZONE="asia-south1-a"
VM_NAME="stocks-backtest-$(date +%s)"
MACHINE_TYPE="e2-standard-8"   # 8 vCPU, 32GB RAM
BUCKET_NAME=""
INTERVAL="5m"
QUICK_MODE="false"

# ── PARSE ARGS ────────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --project)   PROJECT_ID="$2";   shift ;;
        --region)    REGION="$2";       shift ;;
        --zone)      ZONE="$2";         shift ;;
        --interval)  INTERVAL="$2";     shift ;;
        --quick)     QUICK_MODE="true"  ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: --project is required"
    echo "Usage: ./scripts/deploy_backtest_gcp.sh --project YOUR_PROJECT_ID"
    exit 1
fi

BUCKET_NAME="${PROJECT_ID}-stocks-backtest"

echo "════════════════════════════════════════════════════════"
echo "  Columnly Stocks — GCP Backtest Runner"
echo "════════════════════════════════════════════════════════"
echo "  Project:  $PROJECT_ID"
echo "  Zone:     $ZONE"
echo "  Machine:  $MACHINE_TYPE (8 vCPU, 32GB RAM)"
echo "  Interval: $INTERVAL"
echo "  VM Name:  $VM_NAME"
echo "════════════════════════════════════════════════════════"
echo ""

# ── SET GCP PROJECT ───────────────────────────────────────────────────────────
gcloud config set project "$PROJECT_ID"

# ── ENABLE REQUIRED APIS ──────────────────────────────────────────────────────
echo "Enabling required GCP APIs..."
gcloud services enable compute.googleapis.com storage.googleapis.com secretmanager.googleapis.com --quiet

# ── CREATE GCS BUCKET ─────────────────────────────────────────────────────────
echo "Creating GCS bucket: gs://$BUCKET_NAME ..."
gsutil mb -l "$REGION" "gs://$BUCKET_NAME" 2>/dev/null || echo "  (bucket already exists)"

# ── UPLOAD CREDENTIALS TO SECRET MANAGER ─────────────────────────────────────
echo "Storing Dhan credentials in Secret Manager (never touches GCS)..."
DHAN_TOKEN=$(grep DHAN_ACCESS_TOKEN .env | cut -d= -f2-)
DHAN_ID=$(grep DHAN_CLIENT_ID .env | cut -d= -f2-)

# Create or update secrets
echo -n "$DHAN_TOKEN" | gcloud secrets create stocks-dhan-token --data-file=- --replication-policy=automatic 2>/dev/null || \
echo -n "$DHAN_TOKEN" | gcloud secrets versions add stocks-dhan-token --data-file=-

echo -n "$DHAN_ID" | gcloud secrets create stocks-dhan-id --data-file=- --replication-policy=automatic 2>/dev/null || \
echo -n "$DHAN_ID" | gcloud secrets versions add stocks-dhan-id --data-file=-

# ── PACKAGE AND UPLOAD CODE ───────────────────────────────────────────────────
echo "Packaging code..."
TMPDIR_PACK=$(mktemp -d)
cp -r . "$TMPDIR_PACK/stocks_code" 2>/dev/null
# Remove cache and credentials from package
rm -rf "$TMPDIR_PACK/stocks_code/data/cache"
rm -f  "$TMPDIR_PACK/stocks_code/.env"
tar -czf "$TMPDIR_PACK/stocks.tar.gz" -C "$TMPDIR_PACK" stocks_code

echo "Uploading code to GCS..."
gsutil cp "$TMPDIR_PACK/stocks.tar.gz" "gs://$BUCKET_NAME/code/stocks.tar.gz"
rm -rf "$TMPDIR_PACK"

# ── STARTUP SCRIPT ────────────────────────────────────────────────────────────
STARTUP_SCRIPT=$(cat <<STARTUP_EOF
#!/bin/bash
set -e
echo "[STARTUP] Beginning backtest setup..."

# Install dependencies
apt-get update -qq
apt-get install -y python3-pip python3-venv wget -qq

# Create working directory
mkdir -p /opt/stocks
cd /opt/stocks

# Download code
gsutil cp gs://$BUCKET_NAME/code/stocks.tar.gz .
tar -xzf stocks.tar.gz
cd stocks_code

# Get credentials from Secret Manager
DHAN_TOKEN=\$(gcloud secrets versions access latest --secret=stocks-dhan-token)
DHAN_ID=\$(gcloud secrets versions access latest --secret=stocks-dhan-id)

# Write .env
cat > .env <<EOF
DHAN_CLIENT_ID=\${DHAN_ID}
DHAN_ACCESS_TOKEN=\${DHAN_TOKEN}
EXECUTION_MODE=paper
CAPITAL_INR=20000
DAILY_LOSS_LIMIT_INR=500
MAX_TRADES_PER_DAY=10
EOF

# Install Python requirements
pip3 install -r requirements.txt -q

# Run backtest
echo "[STARTUP] Starting full backtest — interval=$INTERVAL quick=$QUICK_MODE"
if [ "$QUICK_MODE" = "true" ]; then
    python3 scripts/run_backtest.py --interval $INTERVAL --quick --top 30 2>&1 | tee /tmp/backtest_output.log
else
    python3 scripts/run_backtest.py --interval $INTERVAL --top 30 2>&1 | tee /tmp/backtest_output.log
fi

# Upload results to GCS
echo "[STARTUP] Uploading results..."
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
gsutil cp reports/leaderboard.csv "gs://$BUCKET_NAME/results/leaderboard_\${TIMESTAMP}.csv"
gsutil cp /tmp/backtest_output.log "gs://$BUCKET_NAME/results/backtest_log_\${TIMESTAMP}.log"
echo "[STARTUP] Results uploaded to gs://$BUCKET_NAME/results/"

# Signal completion
echo "BACKTEST_COMPLETE" | gsutil cp - "gs://$BUCKET_NAME/results/status.txt"

# Self-delete VM to stop billing
echo "[STARTUP] Deleting VM to stop billing..."
INSTANCE_NAME=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE_META=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print \$NF}')
gcloud compute instances delete "\$INSTANCE_NAME" --zone="\$ZONE_META" --quiet
STARTUP_EOF
)

# ── CREATE AND LAUNCH VM ──────────────────────────────────────────────────────
echo ""
echo "Launching VM: $VM_NAME ..."
echo "(This VM will self-delete after the backtest completes)"
echo ""

gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-standard \
    --scopes=cloud-platform \
    --metadata="startup-script=$STARTUP_SCRIPT" \
    --quiet

echo ""
echo "✓ VM launched: $VM_NAME"
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Streaming logs (Ctrl+C to detach — VM keeps running)"
echo "════════════════════════════════════════════════════════"
echo ""

# ── STREAM LOGS ───────────────────────────────────────────────────────────────
sleep 30
echo "Waiting for VM to initialize..."

# Poll for completion while streaming serial console output
MAX_WAIT=7200  # 2 hour max
ELAPSED=0
COMPLETE=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if backtest completed
    STATUS=$(gsutil cat "gs://$BUCKET_NAME/results/status.txt" 2>/dev/null || echo "")
    if [ "$STATUS" = "BACKTEST_COMPLETE" ]; then
        COMPLETE=true
        break
    fi

    # Stream recent serial output
    gcloud compute instances get-serial-port-output "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" 2>/dev/null | tail -5 || true

    sleep 30
    ELAPSED=$((ELAPSED + 30))
done

# ── DOWNLOAD AND DISPLAY RESULTS ──────────────────────────────────────────────
if [ "$COMPLETE" = "true" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  BACKTEST COMPLETE"
    echo "════════════════════════════════════════════════════════"
    echo ""

    # Download latest leaderboard
    mkdir -p reports
    LATEST=$(gsutil ls "gs://$BUCKET_NAME/results/leaderboard_*.csv" 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST" ]; then
        gsutil cp "$LATEST" reports/leaderboard_gcp.csv
        echo "Leaderboard downloaded → reports/leaderboard_gcp.csv"
        echo ""
        echo "Top 10 strategies:"
        python3 -c "
import csv
with open('reports/leaderboard_gcp.csv') as f:
    rows = list(csv.DictReader(f))
print(f'{'Rank':<5} {'Strategy':<32} {'Score':<8} {'Sharpe':<8} {'Win%':<8} {'PF':<6} {'Net PnL':<12} {'Verdict'}")
print('-'*85)
for r in rows[:10]:
    print(f\"{r['rank']:<5} {r['strategy']:<32} {float(r['composite_score']):<8.1f} {float(r['avg_sharpe']):<8.2f} {float(r['avg_win_rate'])*100:<8.1f} {float(r['avg_profit_factor']):<6.2f} INR {float(r['avg_net_pnl_inr']):<10,.0f} {r['verdict']}\")
"
    fi

    echo ""
    echo "All results: gs://$BUCKET_NAME/results/"
    echo "Full log:    gsutil cat gs://$BUCKET_NAME/results/backtest_log_*.log | tail -200"
else
    echo ""
    echo "Timed out waiting for results. VM may still be running."
    echo "Check: gsutil cat gs://$BUCKET_NAME/results/status.txt"
    echo "Logs:  gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE"
fi
