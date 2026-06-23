# Short BALL-CAM test capture (viewer already running with Ball cam ON, rlviser topmost fullscreen).
# Captures 140s, stores every goal as BALLCAM-tagged clips for a side-by-side vs behind-car.
$ErrorActionPreference = 'Continue'
$repo = 'c:\Users\Lasca\Desktop\Pro\IE\Courses\REINFORCEMENT LEARNING & AUTONOMOUS SYSTEMS\Group Project\rlgym'
$py = 'C:\Users\Lasca\miniconda3\envs\rl-group-project\python.exe'
$base = "$repo\highlights_3d"
$ff = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
$log = "$base\_ballcam_test.log"; "" | Out-File $log -Encoding ascii
function L($m){ "{0} | {1}" -f (Get-Date -Format 'HH:mm:ss'), $m | Out-File -Append $log -Encoding ascii }
Set-Location $repo
Add-Type @"
using System; using System.Runtime.InteropServices;
public class WTB {
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr a, int x, int y, int cx, int cy, uint f);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
  public static IntPtr TOP = new IntPtr(-1);
}
"@
$rl = Get-Process rlviser -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $rl) { L "rlviser NOT running - abort"; return }
[void][WTB]::ShowWindow($rl.MainWindowHandle, 9); Start-Sleep -Milliseconds 200
[void][WTB]::SetWindowPos($rl.MainWindowHandle, [WTB]::TOP, 0, 0, 2048, 1152, 0x0040); Start-Sleep -Milliseconds 400
$t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0
$t0.ToString("F3") | Out-File "$base\_capture_t0.txt" -Encoding ascii
L "ballcam capture 140s"
& $ff -y -filter_complex "ddagrab=output_idx=0:framerate=60,hwdownload,format=bgra,crop=2048:1152:0:0" -t 140 -r 60 -fps_mode cfr -c:v h264_nvenc -preset p2 -cq 20 -pix_fmt yuv420p "$base\_raw_session.mp4" 2>$null
L "storing BALLCAM clips"
& $py "$repo\tools\_store_highlights.py" BALLCAM 2>$null | Out-File -Append $log -Encoding ascii
L "done"
