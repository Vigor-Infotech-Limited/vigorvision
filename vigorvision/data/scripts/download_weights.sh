#!/bin/bash
# vigorvision  AGPL-3.0 License - https://vigorvision.com/license

# Download latest models from https://github.com/vigorvision/assets/releases
# Example usage: bash vigorvision/data/scripts/download_weights.sh
# parent
# └── weights
#     ├── Visionv8n.pt  ← downloads here
#     ├── Visionv8s.pt
#     └── ...

python << EOF
from vigorvision.utils.downloads import attempt_download_asset

assets = [f"Visionv8{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose")]
for x in assets:
    attempt_download_asset(f"weights/{x}")
EOF
