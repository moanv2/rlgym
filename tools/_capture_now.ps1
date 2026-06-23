# Capture the ALREADY-RUNNING viewer. Forces rlviser TOPMOST + FULLSCREEN so it stays above the
# editor/desktop for the WHOLE capture (SetForegroundWindow alone fails under Windows' foreground
# lock -> a prior run recorded the editor). Records directly at the viewer's speed (no post-speedup).
$ErrorActionPreference = 'Continue'
$repo = 'c:\Users\Lasca\Desktop\Pro\IE\Courses\REINFORCEMENT LEARNING & AUTONOMOUS SYSTEMS\Group Project\rlgym'
$py = 'C:\Users\Lasca\miniconda3\envs\rl-group-project\python.exe'
$base = "$repo\highlights_3d"
$ff = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
$log = "$base\_capture_now.log"
function L($m){ "{0} | {1}" -f (Get-Date -Format 'HH:mm:ss'), $m | Out-File -Append $log -Encoding ascii }
"" | Out-File $log -Encoding ascii
Set-Location $repo

Add-Type @"
using System; using System.Runtime.InteropServices;
public class WTOP {
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
  public static IntPtr TOP = new IntPtr(-1); public static IntPtr NOTOP = new IntPtr(-2);
}
"@
Add-Type -AssemblyName System.Windows.Forms
$sb = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds      # FULL screen (covers taskbar) = true 16:9
$SW = $sb.Width - ($sb.Width % 2); $SH = $sb.Height - ($sb.Height % 2)

$rl = Get-Process rlviser -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $rl) { L "rlviser NOT running - abort"; return }
$h = $rl.MainWindowHandle
# force rlviser to cover the whole screen, ABOVE everything (topmost), and keep behind-car cam
[void][WTOP]::ShowWindow($h, 9)                                # SW_RESTORE so SetWindowPos can resize it
Start-Sleep -Milliseconds 300
[void][WTOP]::SetWindowPos($h, [WTOP]::TOP, 0, 0, $SW, $SH, 0x0040)   # HWND_TOPMOST | SWP_SHOWWINDOW
Start-Sleep -Milliseconds 400
[void][WTOP]::SetForegroundWindow($h); Start-Sleep -Milliseconds 300
(New-Object -ComObject WScript.Shell).SendKeys("1"); Start-Sleep -Milliseconds 500
L "crop ${SW}:${SH}:0:0 (topmost fullscreen)"
$t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0
$t0.ToString("F3") | Out-File "$base\_capture_t0.txt" -Encoding ascii
L "capturing 380s at 60fps"
& $ff -y -filter_complex "ddagrab=output_idx=0:framerate=60,hwdownload,format=bgra,crop=${SW}:${SH}:0:0" -t 380 -r 60 -fps_mode cfr -c:v h264_nvenc -preset p2 -cq 20 -pix_fmt yuv420p "$base\_raw_session.mp4" 2>$null
L "capture done; releasing topmost + stopping viewer"
[void][WTOP]::SetWindowPos($h, [WTOP]::NOTOP, 0, 0, $SW, $SH, 0x0040)   # HWND_NOTOPMOST (release)
$pid0 = (Get-Content "$base\_viewer_pid.txt" -ErrorAction SilentlyContinue | Select-Object -First 1)
if ($pid0) { Stop-Process -Id ([int]$pid0) -Force -ErrorAction SilentlyContinue }
Get-Process rlviser -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
L "clipping top 5 goals"
& $py "$repo\tools\_clip_reel.py" 5 2>$null | Out-File -Append $log -Encoding ascii
$tag = "r" + (Get-Date -Format 'HHmmss')
L "storing all good highlights (tag $tag)"
& $py "$repo\tools\_store_highlights.py" $tag 2>$null | Out-File -Append $log -Encoding ascii
if (Test-Path "$base\RLVISER_HIGHLIGHTS_HIGH.mp4") {
  Copy-Item "$base\RLVISER_HIGHLIGHTS_HIGH.mp4" "C:\Users\Lasca\Downloads\MARTIN_10B_AERIAL_HIGHLIGHTS.mp4" -Force -ErrorAction SilentlyContinue
  L "DONE -> Downloads\MARTIN_10B_AERIAL_HIGHLIGHTS.mp4 + review library"
} else {
  L "ERROR: reel not produced"
}
