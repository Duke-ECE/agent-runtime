#!/bin/sh
# sync-proto.sh — refresh the vendored protobuf contracts from a released tag.
#
# agent-runtime does not consume the generated Go module; it vendors the .proto
# sources and loads them at runtime with @grpc/proto-loader. The files are
# generated artifacts: never hand-edit them, change the tag below instead.
#
#   scripts/sync-proto.sh
set -eu
cd "$(dirname "$0")/.."

TAG="${PROTO_TAG:-v0.8.0}"
BASE="https://raw.githubusercontent.com/Duke-ECE/protos/${TAG}/proto"

fetch() {
  url="${BASE}/$1"
  out="proto/$1"
  mkdir -p "$(dirname "$out")"
  curl -fsSL "$url" -o "$out"
  echo "synced $out (${TAG})"
}

# v1 stays vendored while the platform migrates; v2 is the canonical-message and
# durable-request contract.
fetch runtime/v1/agent.proto
fetch session/v1/session.proto
fetch runtime/v2/agent.proto
fetch session/v2/session.proto

echo "sync-proto: vendored from protos ${TAG}"
