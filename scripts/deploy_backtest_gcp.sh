#!/bin/bash
# =============================================================================
# deploy_backtest_gcp.sh — One-command GCP backtest runner
#
# USAGE:
#   chmod +x scripts/deploy_backtest_gcp.sh
#   ./scripts/deploy_backtest_gcp.sh \
#       --project stocks-490622 \
#       --dhan-token "YOUR_JWT_TOKEN" \
#       --dhan-id   "1110911380"
#
# WHAT IT DOES:
#   1. Creates a standard VM (8 vCPU, 32GB RAM) — self-deletes when done
#   2. Stores credentials in Secret Manager (never in code or GCS)
#   3. VM clones the repo from GitHub directly
#   4. Runs full backtest (45 strategies × full universe)
#   5. Saves leaderboard to GCS bucket
#   6. VM self-deletes — you are never billed for idle time
#
# COST: ~₹5-8 total for the full backtest run (~2-3 hours)
# =============================================================================

set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ID=""
REGION="asia-south1"
ZONE="asia-south1-a"
VM_NAME="stocks-backtest-$(date +%s)"
MACHINE_TYPE="e2-standard-8"   # 8 vCPU, 32GB RAM
INTERVAL="5m"
DHAN_TOKEN=""
DHAN_ID=""
GITHUB_REPO="https://github.com/Vishesh90/Stocks.git"

# ── PARSE ARGS ────────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --project)     PROJECT_ID="$2";   shift ;;
        --zone)        ZONE="$2";         shift ;;
        --interval)    INTERVAL="$2";     shift ;;
        --dhan-token)  DHAN_TOKEN="$2";   shift ;;
        --dhan-id)     DHAN_ID="$2";      shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── VALIDATE ──────────────────────────────────────────────────────────────────
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: --project is required"
    echo "Usage: ./scripts/deploy_backtest_gcp.sh --project stocks-490622 --dhan-token YOUR_TOKEN --dhan-id 1110911380"
    exit 1
fi

if [ -z "$DHAN_TOKEN" ] || [ -z "$DHAN_ID" ]; then
    echo "ERROR: --dhan-token and --dhan-id are required"
    echo "Get your token from: https://dhanhq.co/docs/v2/"
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
gcloud config set project "$PROJECT_ID" --quiet

# ── ENABLE REQUIRED APIS ──────────────────────────────────────────────────────
echo "Enabling GCP APIs (may take 30s on first run)..."
gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    --quiet

# ── CREATE GCS BUCKET ─────────────────────────────────────────────────────────
echo "Creating GCS bucket: gs://$BUCKET_NAME ..."
gsutil mb -l "$REGION" "gs://$BUCKET_NAME" 2>/dev/null || true
echo "  Bucket ready."

# ── STORE CREDENTIALS IN SECRET MANAGER ──────────────────────────────────────
# BUG FIX 1: Don't use set -e around secret creation — use explicit if/else
# BUG FIX 3: Credentials are passed as CLI args, not read from .env file
echo "Storing Dhan credentials in Secret Manager..."

# Token secret
if gcloud secrets describe stocks-dhan-token --project="$PROJECT_ID" &>/dev/null; then
    echo -n "$DHAN_TOKEN" | gcloud secrets versions add stocks-dhan-token --data-file=- --quiet
    echo "  Token secret updated."
else
    echo -n "$DHAN_TOKEN" | gcloud secrets create stocks-dhan-token \
        --data-file=- --replication-policy=automatic --quiet
    echo "  Token secret created."
fi

# Client ID secret
if gcloud secrets describe stocks-dhan-id --project="$PROJECT_ID" &>/dev/null; then
    echo -n "$DHAN_ID" | gcloud secrets versions add stocks-dhan-id --data-file=- --quiet
    echo "  Client ID secret updated."
else
    echo -n "$DHAN_ID" | gcloud secrets create stocks-dhan-id \
        --data-file=- --replication-policy=automatic --quiet
    echo "  Client ID secret created."
fi

# ── WRITE STARTUP SCRIPT TO A TEMP FILE ──────────────────────────────────────
# BUG FIX 2: Write startup script to a file instead of inline heredoc
# This avoids the EOF-inside-STARTUP_EOF conflict that broke the VM script
# BUG FIX 4: VM clones from GitHub directly — no packaging needed

STARTUP_FILE=$(mktemp /tmp/startup_XXXXXX.sh)
cat > "$STARTUP_FILE" << 'STARTUP_END'
#!/bin/bash
set -e
exec > /var/log/backtest_startup.log 2>&1

echo "[$(date)] Starting backtest VM setup..."

# Install system dependencies
apt-get update -qq
apt-get install -y python3-pip python3-venv git wget -qq
echo "[$(date)] System packages installed."

# Clone repo from GitHub
mkdir -p /opt/stocks
cd /opt/stocks
git clone https://github.com/Vishesh90/Stocks.git stocks_code
cd stocks_code
echo "[$(date)] Repo cloned."

# Get credentials from Secret Manager
DHAN_TOKEN=$(gcloud secrets versions access latest --secret=stocks-dhan-token)
DHAN_ID=$(gcloud secrets versions access latest --secret=stocks-dhan-id)

# Write .env file
cat > .env << ENVEOF
DHAN_CLIENT_ID=${DHAN_ID}
DHAN_ACCESS_TOKEN=${DHAN_TOKEN}
EXECUTION_MODE=paper
CAPITAL_INR=20000
DAILY_LOSS_LIMIT_INR=500
MAX_TRADES_PER_DAY=10
ENVEOF
echo "[$(date)] Credentials written."

# Install Python packages
pip3 install -r requirements.txt -q
echo "[$(date)] Python packages installed."

# Run full backtest
echo "[$(date)] Starting full backtest..."
python3 scripts/run_backtest.py --interval INTERVAL_PLACEHOLDER --top 30 \
    2>&1 | tee /tmp/backtest_output.log
echo "[$(date)] Backtest complete."

# Upload results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUCKET_NAME_PLACEHOLDER="BUCKET_PLACEHOLDER"
gsutil cp reports/leaderboard.csv "gs://${BUCKET_NAME_PLACEHOLDER}/results/leaderboard_${TIMESTAMP}.csv"
gsutil cp /tmp/backtest_output.log "gs://${BUCKET_NAME_PLACEHOLDER}/results/backtest_log_${TIMESTAMP}.log"
echo "BACKTEST_COMPLETE" | gsutil cp - "gs://${BUCKET_NAME_PLACEHOLDER}/results/status.txt"
echo "[$(date)] Results uploaded."

# Self-delete VM
INSTANCE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE_META=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
echo "[$(date)] Deleting VM ${INSTANCE} in ${ZONE_META}..."
gcloud compute instances delete "${INSTANCE}" --zone="${ZONE_META}" --quiet
STARTUP_END

# Substitute actual values into the startup script
sed -i "s|INTERVAL_PLACEHOLDER|${INTERVAL}|g" "$STARTUP_FILE"
sed -i "s|BUCKET_PLACEHOLDER|${BUCKET_NAME}|g" "$STARTUP_FILE"

echo "Startup script written to $STARTUP_FILE"

# ── CREATE AND LAUNCH VM ──────────────────────────────────────────────────────
echo ""
echo "Launching VM: $VM_NAME ..."

gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-standard \
    --scopes=cloud-platform \
    --metadata-from-file="startup-script=${STARTUP_FILE}" \
    --quiet

rm -f "$STARTUP_FILE"

echo ""
echo "  VM launched: $VM_NAME"
echo "  The VM will run independently — you can close this terminal."
echo ""
echo "════════════════════════════════════════════════════════"
echo "  TO CHECK PROGRESS (run anytime):"
echo "  gsutil cat gs://$BUCKET_NAME/results/status.txt"
echo ""
echo "  TO STREAM VM LOGS:"
echo "  gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE --project=$PROJECT_ID 2>/dev/null | tail -30"
echo ""
echo "  WHEN DONE — download results:"
echo "  gsutil cp \$(gsutil ls gs://$BUCKET_NAME/results/leaderboard_*.csv | tail -1) ~/leaderboard_full.csv && cat ~/leaderboard_full.csv"
echo "════════════════════════════════════════════════════════"
