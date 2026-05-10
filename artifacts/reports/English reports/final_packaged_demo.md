# Final Packaged Demo Smoke

Date: 2026-04-24

## Result

- Command: start `dist/bucad-demo/bucad-demo.exe` from `dist/bucad-demo`.
- Smoke duration: 20 seconds.
- Result: PASS, process stayed alive for 20 seconds and was then stopped by the validation script.
- Stdout: `artifacts/reports/packaged_demo_stdout.txt`.
- Stderr: `artifacts/reports/packaged_demo_stderr.txt`.
- Observed stdout/stderr byte count: 0/0.

## Notes

- This verifies executable startup stability with packaged configs and checkpoints.
- A live browser rehearsal is still recommended before presentation.
