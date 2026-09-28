#!/usr/bin/env bash
# Build an sdist + wheel for mpcdset into dist/.
#
# Usage:
#   scripts/build.sh            # run the test suite, then build
#   scripts/build.sh --no-test  # skip the test suite, just build
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

if [[ "${1:-}" != "--no-test" ]]; then
    uv run pytest
fi

uv build
echo "built into $(pwd)/dist/"
