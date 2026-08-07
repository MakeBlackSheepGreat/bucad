@echo off
rem Run the local BUCAD development launcher with the repository script.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-dev.ps1"
endlocal
