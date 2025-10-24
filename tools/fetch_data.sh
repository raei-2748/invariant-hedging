#!/usr/bin/env bash
set -euo pipefail

USER_DATA_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      shift
      USER_DATA_DIR=${1:-}
      ;;
    --data-dir=*)
      USER_DATA_DIR=${1#*=}
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: tools/fetch_data.sh [--data-dir <path>]

Mirrors the bundled SPY sample dataset into the requested directory.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift || true
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
if [[ -n "${USER_DATA_DIR}" ]]; then
  DATA_DIR="${USER_DATA_DIR}"
fi
DATA_ROOT=${DATA_DIR:-${REPO_ROOT}/data}
if [[ "${DATA_ROOT}" != /* ]]; then
  DATA_ROOT="${REPO_ROOT}/${DATA_ROOT}"
fi
EXTERNAL_DIR="${DATA_ROOT}/external"
RAW_DIR="${DATA_ROOT}/raw"
DATA_NAME="spy_options_synthetic.csv"
SEED_FILE="${REPO_ROOT}/tools/data_seed/${DATA_NAME}"
EXPECTED_SHA="296298b443bed9325dbdf4b0148a663668855ac4328594aa461bc1dbbe8a0f67"

mkdir -p "${EXTERNAL_DIR}" "${RAW_DIR}"

EXTERNAL_TARGET="${EXTERNAL_DIR}/${DATA_NAME}"
RAW_OUTPUT="${RAW_DIR}/${DATA_NAME}"

log() {
  printf '[fetch_data] %s\n' "$*"
}

checksum() {
  sha256sum "$1" | awk '{print $1}'
}

write_checksum_file() {
  local file_path="$1"
  local digest="$2"
  local checksum_path="${file_path}.sha256"
  printf '%s  %s\n' "${digest}" "$(basename "${file_path}")" > "${checksum_path}"
}

if [[ ! -f "${EXTERNAL_TARGET}" ]]; then
  if [[ -f "${SEED_FILE}" ]]; then
    log "Seeding ${DATA_NAME} from repository copy."
    cp "${SEED_FILE}" "${EXTERNAL_TARGET}"
  else
    cat <<MSG >&2
[fetch_data] Missing ${DATA_NAME} in the data cache and no seed copy was bundled.
[fetch_data] Please follow the instructions in data/DATA_LICENSE.md to obtain the file and place it at:
[fetch_data]   ${DATA_ROOT}/external/${DATA_NAME}
MSG
    exit 1
  fi
else
  log "Found cached dataset at ${EXTERNAL_TARGET}."
fi

EXTERNAL_SHA=$(checksum "${EXTERNAL_TARGET}")
if [[ "${EXTERNAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  cat <<MSG >&2
[fetch_data] Checksum mismatch for ${EXTERNAL_TARGET}.
[fetch_data]   expected: ${EXPECTED_SHA}
[fetch_data]   observed: ${EXTERNAL_SHA}
[fetch_data] Delete the file and re-run the fetch script or refresh from the repository copy.
MSG
  exit 1
fi
write_checksum_file "${EXTERNAL_TARGET}" "${EXTERNAL_SHA}"

TMP_RAW="${RAW_OUTPUT}.tmp"
log "Copying dataset into ${RAW_OUTPUT}."
cp "${EXTERNAL_TARGET}" "${TMP_RAW}"
RAW_SHA=$(checksum "${TMP_RAW}")
if [[ "${RAW_SHA}" != "${EXPECTED_SHA}" ]]; then
  cat <<MSG >&2
[fetch_data] Unexpected contents produced when copying ${EXTERNAL_TARGET}.
[fetch_data]   expected: ${EXPECTED_SHA}
[fetch_data]   observed: ${RAW_SHA}
[fetch_data] The source may be corrupt; delete it and try again.
MSG
  rm -f "${TMP_RAW}"
  exit 1
fi
mv "${TMP_RAW}" "${RAW_OUTPUT}"
write_checksum_file "${RAW_OUTPUT}" "${RAW_SHA}"

log "Dataset verified (${EXTERNAL_SHA})."
log "Raw dataset ready at ${RAW_OUTPUT} (sha256=${RAW_SHA})."
