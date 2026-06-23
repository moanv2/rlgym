# Autonomous: launch smooth aerial viewer -> maximize/close-menu rlviser -> ddagrab+NVENC capture
# -> stop viewer -> auto-clip top aerials -> speed up ~1.3x -> copy to Downloads. Runs to completion.
$ErrorActionPreference = 'Continue'
$repo = 'c:\Users\Lasca\Desktop\Pro\IE\Courses\REINFORCEMENT LEARNING & AUTONOMOUS SYSTEMS\Group Project\rlgym'
$py = 'C:\Users\Lasca\miniconda3\envs\rl-group-project\python.exe'
$base = "$repo\highlights_3d"
$ff = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
$log = "$base\_auto_reel.log"
function L($m){ "{0} | {1}" -f (Get-Date -Format 'HH:mm:ss'), $m | Out-File -Append $log -Encoding ascii }
Set-Location $repo
Remove-Item "$base\_goals.jsonl","$base\_raw_session.mp4" -ErrorAction SilentlyContinue
"" | Out-File $log -Encoding ascii
L "launching viewer"
$env:PYTHONPATH = "$repo\src"; $env:WANDB_MODE = 'offline'
$v = Start-Process $py -ArgumentList 'tools\_smooth_viewer.py','--orange','C:\Users\Lasca\rlgym_tourney_wt\teammates\nachi','--episodes','800','--speed','1.3','--goal-log','highlights_3d/_goals.jsonl' -PassThru -WindowStyle Hidden -WorkingDirectory $repo -RedirectStandardOutput "$base\_auto_viewer.log" -RedirectStandardError "$base\_auto_viewer.err"

Add-Type @"
using System; using System.Runtime.InteropServices;
public class WC { [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
[StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; } }
"@
for ($i=0; $i -lt 40; $i++){ if (Get-Process rlviser -ErrorAction SilentlyContinue) { break }; Start-Sleep -Seconds 2 }
$rl = Get-Process rlviser -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $rl) { L "rlviser NOT up - abort"; return }
Start-Sleep -Seconds 3
[void][WC]::ShowWindow($rl.MainWindowHandle, 3); Start-Sleep -Milliseconds 800
[void][WC]::SetForegroundWindow($rl.MainWindowHandle); Start-Sleep -Milliseconds 400
(New-Object -ComObject WScript.Shell).SendKeys("{ESC}"); Start-Sleep -Milliseconds 700
$r = New-Object WC+RECT; [void][WC]::GetWindowRect($rl.MainWindowHandle, [ref]$r)
$Lx=[Math]::Max(0,$r.Left); $Ty=[Math]::Max(0,$r.Top)
$Wd=([Math]::Min(2560,$r.Right)-$Lx); $Hd=([Math]::Min(1440,$r.Bottom)-$Ty); $Wd=$Wd-($Wd%2); $Hd=$Hd-($Hd%2)
L "crop ${Wd}:${Hd}:${Lx}:${Ty}"
$t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0
$t0.ToString("F3") | Out-File "$base\_capture_t0.txt" -Encoding ascii
L "capturing 350s"
& $ff -y -filter_complex "ddagrab=output_idx=0:framerate=60,hwdownload,format=bgra,crop=${Wd}:${Hd}:${Lx}:${Ty}" -t 350 -c:v h264_nvenc -preset p2 -cq 20 -pix_fmt yuv420p "$base\_raw_session.mp4" 2>$null
L "capture done; stopping viewer"
Stop-Process -Id $v.Id -Force -ErrorAction SilentlyContinue
Get-Process rlviser -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
L "clipping"
& $py "$repo\tools\_clip_reel.py" 8 2>$null | Out-File -Append $log -Encoding ascii
L "finalizing (viewer already ~1.3x, no post-speedup)"
& $ff -y -i "$base\RLVISER_HIGHLIGHTS_HIGH.mp4" -vf "setpts=1.0*PTS" -r 54 -c:v h264_nvenc -preset p2 -cq 20 -pix_fmt yuv420p -an "$base\RLVISER_AERIAL_FAST.mp4" 2>$null
Copy-Item "$base\RLVISER_AERIAL_FAST.mp4" "C:\Users\Lasca\Downloads\RLVISER_AERIAL_HIGHLIGHTS_FAST.mp4" -Force -ErrorAction SilentlyContinue
Copy-Item "$base\RLVISER_HIGHLIGHTS_HIGH.mp4" "C:\Users\Lasca\Downloads\RLVISER_AERIAL_HIGHLIGHTS_NORMALSPEED.mp4" -Force -ErrorAction SilentlyContinue
L "DONE -> Downloads\RLVISER_AERIAL_HIGHLIGHTS_FAST.mp4"
