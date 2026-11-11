from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from safeai.pipeline.offline import LocalAcquisition


def test_local_acquisition_propagates_session_id_into_service(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    sid = "session_acq_forced_001"

    snapshot = LocalAcquisition().run_file(wav, session_id=sid)

    assert snapshot.status == "completed"
    assert snapshot.session_id == sid
