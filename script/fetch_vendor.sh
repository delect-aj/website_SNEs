#!/usr/bin/env bash
# Fetch the pinned browser dependencies into web/assets/vendor/.
#
# The JavaScript is committed; only the ONNX Runtime WebAssembly binaries are
# not, because they are 10 MB each and git stores them badly. Run this once
# after cloning, and again only when a version below changes.
#
# Versions are pinned rather than ranged: the wasm binary and the JavaScript
# loader have to come from the same release, and the ONNX export was verified
# against this runtime.

set -euo pipefail

ORT_VERSION="1.17.3"
PAPAPARSE_VERSION="5.5.3"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor="${root}/web/assets/vendor"
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT

cd "${scratch}"

echo "Fetching onnxruntime-web ${ORT_VERSION}"
mkdir -p ort
npm pack "onnxruntime-web@${ORT_VERSION}" >/dev/null
tar -xzf onnxruntime-web-*.tgz -C ort --strip-components=1
mkdir -p "${vendor}/ort"
cp ort/dist/ort.min.js "${vendor}/ort/"
cp ort/dist/ort-wasm-simd-threaded.wasm "${vendor}/ort/"
cp ort/dist/ort-wasm-simd.wasm "${vendor}/ort/"

echo "Fetching papaparse ${PAPAPARSE_VERSION}"
mkdir -p papaparse
npm pack "papaparse@${PAPAPARSE_VERSION}" >/dev/null
tar -xzf papaparse-*.tgz -C papaparse --strip-components=1
mkdir -p "${vendor}/papaparse"
cp papaparse/papaparse.min.js "${vendor}/papaparse/"

echo
echo "Vendored into ${vendor}:"
du -h "${vendor}"/ort/* "${vendor}"/papaparse/*
