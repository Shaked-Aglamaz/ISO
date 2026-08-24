# Export one PNG per slide from the defense deck, for visual checking.
#
# LibreOffice and poppler are not installed on this machine, so the powerpoint
# skill's pptx_render.py cannot run; PowerPoint itself is installed, and its COM
# interface exports slide images directly.
#
# Usage (from repo root):
#   powershell -File code/defense/render_deck.ps1
#   powershell -File code/defense/render_deck.ps1 -Deck thesis/defense/ISFS_defense_V1.pptx -OutDir render
param(
    [string]$Deck = "thesis/defense/ISFS_defense_V2.pptx",
    [string]$OutDir = "thesis/defense/render",
    [int]$Width = 1920,
    [int]$Height = 1080
)

$deckPath = (Resolve-Path $Deck).Path
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$outPath = (Resolve-Path $OutDir).Path

Get-ChildItem -Path $outPath -Filter '*.PNG' -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -Path $outPath -Filter '*.png' -ErrorAction SilentlyContinue | Remove-Item -Force

# Remember which PowerPoint processes already existed, so only the instance this
# script starts is cleaned up at the end. Quit() on a COM-started PowerPoint
# regularly leaves the process alive, and a lingering instance blocks the next run.
$before = @(Get-Process POWERPNT -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)

$ppt = New-Object -ComObject PowerPoint.Application
try {
    # ReadOnly, no title prompt, no window
    $pres = $ppt.Presentations.Open($deckPath, $true, $false, $false)
    $pres.Export($outPath, "PNG", $Width, $Height)
    $count = $pres.Slides.Count
    $pres.Close()
    "Exported $count slides to $outPath"
} finally {
    try { $ppt.Quit() } catch {}
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
    Start-Sleep -Seconds 2
    Get-Process POWERPNT -ErrorAction SilentlyContinue |
        Where-Object { $before -notcontains $_.Id } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}
