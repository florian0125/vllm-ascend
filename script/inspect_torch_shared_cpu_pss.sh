#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /dev/shm/shared-weight-file" >&2
  exit 2
fi

shared_file=$1
if [[ ! -e "$shared_file" ]]; then
  echo "Shared file does not exist: $shared_file" >&2
  exit 1
fi

mapfile -t worker_pids < <(pgrep -f '[v]alidate_torch_shared_cpu_h2d_950dt.py' || true)
if [[ ${#worker_pids[@]} -eq 0 ]]; then
  echo "No validate_torch_shared_cpu_h2d_950dt.py workers found" >&2
  exit 1
fi

file_bytes=$(stat -c '%s' "$shared_file")
total_pss_kib=0
matched_processes=0

echo "shared_file=$shared_file"
echo "file_bytes=$file_bytes"

for pid in "${worker_pids[@]}"; do
  smaps=/proc/$pid/smaps
  [[ -r "$smaps" ]] || continue
  stats=$(awk -v path="$shared_file" '
    index($0, path) > 0 {
      in_mapping = 1
      print
      next
    }
    in_mapping && /^[0-9a-f]+-[0-9a-f]+ / {
      in_mapping = 0
    }
    in_mapping && /^(Size|Rss|Pss|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty):/ {
      print
    }
  ' "$smaps")
  [[ -n "$stats" ]] || continue

  echo
  echo "PID=$pid"
  echo "$stats"
  pss_kib=$(awk '$1 == "Pss:" {sum += $2} END {print sum + 0}' <<<"$stats")
  total_pss_kib=$((total_pss_kib + pss_kib))
  matched_processes=$((matched_processes + 1))
done

if [[ $matched_processes -eq 0 ]]; then
  echo "No worker mapping matched $shared_file" >&2
  exit 1
fi

awk -v file_bytes="$file_bytes" -v pss_kib="$total_pss_kib" \
  -v processes="$matched_processes" 'BEGIN {
    file_mib = file_bytes / 1048576
    pss_mib = pss_kib / 1024
    ratio = file_bytes > 0 ? (pss_kib * 1024) / file_bytes : 0
    printf "\nmatched_processes=%d\n", processes
    printf "shared_file_size_mib=%.2f\n", file_mib
    printf "mapping_total_pss_mib=%.2f\n", pss_mib
    printf "pss_to_file_ratio=%.3f\n", ratio
    print "Interpretation: ratio near 1 indicates one resident physical mapping;"
    print "do not sum per-process RSS to estimate shared physical memory."
  }'
