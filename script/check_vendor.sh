#!/usr/bin/env bash
# Check that the vendored browser dependencies are the versions this repository
# pins, and that the WebAssembly binaries match the JavaScript loader.
#
# Only the loader is committed (`web/assets/vendor/ort/ort.min.js`); the two
# 11 MB `.wasm` files are fetched by `fetch_vendor.sh` and stay out of git. So a
# checkout that has been updated with `git pull` carries the new loader beside
# whatever wasm was fetched last time, and onnxruntime-web then fails in the
# browser with a message that never mentions a version. Run this after pulling,
# and before the rsync that deploys -- the trap is entirely in the local tree.
#
#   script/check_vendor.sh
#
# Exits non-zero when anything is missing or stale.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor="${root}/web/assets/vendor"

# One source of truth for the pin: the fetch script's own variable, read
# rather than executed.
ort_version="$(sed -n 's/^ORT_VERSION="\(.*\)"$/\1/p' "${root}/script/fetch_vendor.sh")"
papa_version="$(sed -n 's/^PAPAPARSE_VERSION="\(.*\)"$/\1/p' "${root}/script/fetch_vendor.sh")"
if [ -z "${ort_version}" ] || [ -z "${papa_version}" ]; then
    echo "cannot read the pinned versions out of script/fetch_vendor.sh" >&2
    exit 1
fi

fail=0

check() {
    local path="$1" version="$2" name="${1#"${root}/"}"
    if [ ! -f "${path}" ]; then
        echo "MISSING  ${name} -- run script/fetch_vendor.sh" >&2
        fail=1
    elif ! grep -qa "${version}" "${path}"; then
        echo "STALE    ${name} does not contain ${version} -- run script/fetch_vendor.sh" >&2
        fail=1
    else
        echo "ok       ${name} is ${version}"
    fi
}

# The loader and both wasm builds have to be the same release; each carries its
# version string, so each is checked against the pin.
check "${vendor}/ort/ort.min.js" "${ort_version}"
check "${vendor}/ort/ort-wasm-simd.wasm" "${ort_version}"
check "${vendor}/ort/ort-wasm-simd-threaded.wasm" "${ort_version}"
check "${vendor}/papaparse/papaparse.min.js" "${papa_version}"

if [ "${fail}" -ne 0 ]; then
    echo >&2
    echo "The vendored browser dependencies are not the pinned versions." >&2
    echo "The page will load a runtime the model was not verified against," >&2
    echo "or fail to start at all. Fix with: script/fetch_vendor.sh" >&2
    exit 1
fi

echo "Vendored dependencies match the pins in script/fetch_vendor.sh."
