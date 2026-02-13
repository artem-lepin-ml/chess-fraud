#!/bin/bash

command -v wget >/dev/null || { echo "Install wget"; exit 1; }
command -v zstd >/dev/null || { echo "Install zstd"; exit 1; }

ZST_DIR="../data/0-raw"
mkdir -p "$ZST_DIR"

files=(
    # 2025 March all blitz rated games
    "lichess_db_standard_rated_2025-03.pgn.zst"
)

BASE_URL="https://database.lichess.org/standard"

download_file() {
    local file="$1"
    local output="$ZST_DIR/$file"
    local url="$BASE_URL/$file"

    [ -f "$output" ] && echo "Exist: $file"

    echo "Download: $file"
    if wget -q --tries=3 --timeout=30 -O "$output" "$url" && zstd -t "$output" &>/dev/null; then
        echo "Done: $file"
    else
        echo "Error: $file"
        rm -f "$output"
        return 1
    fi
}

echo "=== Downloading ${#files[@]} files ==="
for f in "${files[@]}"; do
    download_file "$f" || exit 1
done

echo "=== Summary ==="
echo "ZST: $(find "$ZST_DIR" -name '*.zst' | wc -l) files"
