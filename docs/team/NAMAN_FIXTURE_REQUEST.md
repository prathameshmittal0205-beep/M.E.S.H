# Request for Mock Fixture

**To:** Naman (Data Engineering Owner)
**From:** Prathamesh (ML Owner) / MESH Agent
**Date:** 2026-09-14

Hey Naman, great work on getting the data pipeline merged! We're ready to integrate the ML layer against it, but since the raw and processed datasets are (rightfully) gitignored, I'm currently blocked from physically running the forward pass validation locally.

Could you provide a tiny, version-controlled dummy `.npz` fixture in `tests/fixtures/` that strictly matches your real output contract (shapes, naming, types)? This was exactly the kind of test asset we originally outlined in `DATA_CONTRACT_REQUEST.md`. Having a small versioned sample would permanently unblock ML-side validation for the whole team without us needing to download and compile the massive raw datasets just to test the integration. Let me know if I can help set it up!
