# Synthesize every minimal-pair word with each installed Windows SAPI voice
# into <OutDir>\<Voice>_<word>.wav (16 kHz mono), for scripts/check_pairs.py.
# Usage: powershell -File scripts\synth_pairs.ps1 -OutDir <dir>
param([Parameter(Mandatory = $true)][string]$OutDir)
Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force $OutDir | Out-Null
$words = python -c "import sys; sys.path.insert(0, '.'); from src.pairs import PAIR_SETS; print('\n'.join(sorted({w for s in PAIR_SETS for p in s['pairs'] for w in p})))"
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($v in $synth.GetInstalledVoices()) {
  $name = ($v.VoiceInfo.Name -split ' ')[1]   # "Microsoft David Desktop" -> "David"
  $synth.SelectVoice($v.VoiceInfo.Name)
  foreach ($w in $words) {
    $synth.SetOutputToWaveFile((Join-Path $OutDir "$($name)_$w.wav"), $fmt)
    $synth.Speak($w)
  }
}
$synth.SetOutputToNull()
Write-Output "synthesized $($words.Count) words x $($synth.GetInstalledVoices().Count) voices into $OutDir"
