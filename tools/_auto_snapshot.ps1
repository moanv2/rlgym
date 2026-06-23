# Periodic protected checkpoint snapshots, mirrored to BOTH drives.
# Copies the 2nd-newest (guaranteed-complete, never mid-save) training checkpoint every
# 90 min to:
#   C: checkpoints\_eval_snapshots\auto_snapshots\  (keep last 6)
#   D: D:\rlgym_checkpoint_backups\auto_snapshots\   (keep last 20 -- separate physical disk)
# So neither pruning, a mid-save power-off, NOR a C: disk failure can lose the checkpoints.
# Self-terminates after ~8h.
$ErrorActionPreference = 'SilentlyContinue'
$root = 'C:\Users\Lasca\Desktop\Pro\IE\Courses\REINFORCEMENT LEARNING & AUTONOMOUS SYSTEMS\Group Project\rlgym'
Set-Location $root
$snapC = 'checkpoints\_eval_snapshots\auto_snapshots'
$snapD = 'D:\rlgym_checkpoint_backups\auto_snapshots'
New-Item -ItemType Directory -Path $snapC -Force | Out-Null
New-Item -ItemType Directory -Path $snapD -Force | Out-Null
$log = Join-Path $snapD '_snapshot_log.txt'
$end = (Get-Date).AddHours(8)
while ((Get-Date) -lt $end) {
    $cks = Get-ChildItem 'checkpoints\exp_recipeH_distill-*\*' -Directory |
        Where-Object { $_.Name -match '^\d+$' } | Sort-Object { [long]$_.Name }
    if ($cks.Count -ge 2) {
        $safe = $cks[-2]                       # 2nd-newest = complete, not mid-save
        foreach ($dst in @((Join-Path $snapC $safe.Name), (Join-Path $snapD $safe.Name))) {
            if (-not (Test-Path $dst)) {
                New-Item -ItemType Directory -Path $dst -Force | Out-Null
                Copy-Item (Join-Path $safe.FullName '*') $dst -Force
            }
        }
        ("{0}  snapshotted {1} (~{2}B) -> C + D" -f (Get-Date -Format 'MM-dd HH:mm'), $safe.Name, [math]::Round([long]$safe.Name/1e9,3)) | Out-File -Append $log
        # prune: C keep 6, D keep 20
        $cC = Get-ChildItem $snapC -Directory | Where-Object { $_.Name -match '^\d+$' } | Sort-Object { [long]$_.Name }
        if ($cC.Count -gt 6)  { $cC[0..($cC.Count - 7)]  | Remove-Item -Recurse -Force }
        $cD = Get-ChildItem $snapD -Directory | Where-Object { $_.Name -match '^\d+$' } | Sort-Object { [long]$_.Name }
        if ($cD.Count -gt 20) { $cD[0..($cD.Count - 21)] | Remove-Item -Recurse -Force }
    }
    Start-Sleep -Seconds 5400
}
