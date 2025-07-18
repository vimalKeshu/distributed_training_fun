#!/usr/bin/env bash
set -euo pipefail

# 1) Generate version based on current UTC time (YYYYMMDDHHMMSS)
version="$(date -u +%Y%m%d%H%M%S)"
echo "▶ Using version: $version"

# 2) Build the image locally without cache
docker build --no-cache -t dtpp:"${version}" .

# 3) Tag the image for GCR
docker tag dtpp:"${version}" \
  gcr.io/mde-cloud/image-repo/dtpp:"${version}"

# 4) Push to GCR
docker push gcr.io/mde-cloud/image-repo/dtpp:"${version}"

echo "✅ Docker image gcr.io/mde-cloud/image-repo/dtpp:${version} built, tagged, and pushed to gcr.io/mde-cloud/image-repo."