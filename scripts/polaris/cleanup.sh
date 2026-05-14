#!/bin/bash
#
# Polaris cleanup: tear down a prior Scion install at $SCION_ROOT.
#
# Preserves cache/ and home/ by default — that's where Boltz-2 model
# weights (~5 GB) and HF Hub downloads live. Pass --wipe-cache to also
# remove them (you'll re-download on the next install).
#
# Usage:
#   SCION_ROOT=/lus/eagle/projects/<PROJECT>/scion bash cleanup.sh
#   SCION_ROOT=$HOME/scion bash cleanup.sh --wipe-cache
#
set -euo pipefail

: "${SCION_ROOT:?Set SCION_ROOT to the install root you want to clean up}"

WIPE_CACHE=0
for arg in "$@"; do
    case "$arg" in
        --wipe-cache) WIPE_CACHE=1 ;;
        -h|--help)
            grep -E '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

if [ ! -d "$SCION_ROOT" ]; then
    echo "Nothing to clean up: $SCION_ROOT does not exist."
    exit 0
fi

echo "About to clean up Scion install at: $SCION_ROOT"
echo "  Will remove: envs/, environments/, .python/, manifest.json, cluster.toml"
if [ "$WIPE_CACHE" = "1" ]; then
    echo "  Will ALSO remove: cache/, home/   (--wipe-cache)"
else
    echo "  Will KEEP: cache/, home/   (use --wipe-cache to remove)"
fi
read -rp "Proceed? [y/N] " ok
case "$ok" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
esac

# A successful install creates these directly under $SCION_ROOT.
# Use -- so a stray leading dash in $SCION_ROOT can't masquerade as a flag.
rm -rf -- \
    "$SCION_ROOT/envs" \
    "$SCION_ROOT/environments" \
    "$SCION_ROOT/.python" \
    "$SCION_ROOT/manifest.json" \
    "$SCION_ROOT/cluster.toml"

if [ "$WIPE_CACHE" = "1" ]; then
    rm -rf -- "$SCION_ROOT/cache" "$SCION_ROOT/home"
    echo "Removed cache/ and home/."
fi

# Drop empty $SCION_ROOT (only if it's now empty) so a fresh install
# starts cleanly. Best-effort; ignore if the user has other files there.
rmdir "$SCION_ROOT" 2>/dev/null || true

echo "Cleanup complete."
