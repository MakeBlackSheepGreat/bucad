# Demo Rehearsal Checklist

Date: 2026-04-24

Status: automated and packaged smoke evidence recorded; final in-person presentation rehearsal is still recommended.

- [x] Confirm automated Gradio smoke test passes.
- [x] Export final BUSI visual evidence samples.
- [x] Run final BUSI batch inference export.
- [x] Confirm lesion overlay is present or the missing reason is visible.
- [x] Confirm Grad-CAM heatmap is present or the missing reason is visible.
- [x] Confirm packaged assets exist under `artifacts/release_v1/`.
- [x] Confirm `artifacts/reports/release_v1_manifest.md` is available for handoff verification.
- [x] Confirm packaged executable starts and stays alive for a 20-second smoke window.
- [ ] Capture final live browser screenshots for the PPT.
- [ ] Confirm the auxiliary-use disclaimer is visible in a live browser rehearsal.

Chinese UI rehearsal note: present the system as an auxiliary diagnostic prototype, explain probability output first, then lesion overlay, then Grad-CAM heatmap, and explicitly state that medical diagnosis remains clinician-led.
