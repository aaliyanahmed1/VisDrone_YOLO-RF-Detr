# Create venv with Python 3.10+ and install deps (for YOLO26 + RF-DETR)
# Run: .\setup_venv.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv with Python 3.11..."
    py -3.11 -m venv .venv
}
Write-Host "Activating .venv and installing packages..."
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "git+https://github.com/roboflow/rf-detr.git"
Write-Host ""
Write-Host "Done. To activate later: .\.venv\Scripts\Activate.ps1"
Write-Host "Then run: python scripts/run_test_on_testdev.py --test-only --output-dir test_results_testdev"
