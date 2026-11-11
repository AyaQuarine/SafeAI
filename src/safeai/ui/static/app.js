const els = {
  source: document.getElementById("source"),
  file: document.getElementById("file"),
  fileField: document.getElementById("file-field"),
  device: document.getElementById("device"),
  deviceField: document.getElementById("device-field"),
  consent: document.getElementById("consent"),
  consentField: document.getElementById("consent-field"),
  energyVad: document.getElementById("energy-vad"),
  run: document.getElementById("run"),
  stop: document.getElementById("stop"),
  sourceLabel: document.getElementById("source-label"),
  stream: document.getElementById("m-stream"),
  speech: document.getElementById("m-speech"),
  rate: document.getElementById("m-rate"),
  quality: document.getElementById("m-quality"),
  rows: document.getElementById("rows"),
  summary: document.getElementById("summary"),
  contracts: document.getElementById("contracts"),
  incidents: document.getElementById("incidents"),
  wave: document.getElementById("wave"),
};

let pollTimer = null;

function syncSourceFields() {
  const mode = els.source.value;
  els.fileField.classList.toggle("hidden", mode !== "file");
  els.deviceField.classList.toggle("hidden", mode !== "microphone");
  els.consentField.classList.toggle("hidden", mode !== "microphone");
  els.run.textContent = mode === "microphone" ? "Démarrer la capture" : "Analyser";
}

async function loadDevices() {
  const response = await fetch("/api/devices");
  const data = await response.json();
  els.device.innerHTML = "";
  const devices = data.devices || [];
  if (!devices.length) {
    els.device.innerHTML = `<option value="">Aucun microphone détecté</option>`;
    return;
  }
  for (const device of devices) {
    const option = document.createElement("option");
    option.value = device.id;
    option.textContent = `${device.name} · ${device.default_samplerate} Hz`;
    els.device.appendChild(option);
  }
}

function qualityClass(status) {
  if (status === "DEGRADED") return "degraded";
  if (status === "UNUSABLE") return "unusable";
  return "ok";
}

function speechLabel(events) {
  if (!events.length) return "Inconnue";
  const last = events[events.length - 1];
  if (last.speech_activity === "SPEECH") return "Parole détectée";
  if (last.speech_activity === "NON_SPEECH") return "Silence";
  return "Inconnue";
}

function qualityLabel(status) {
  if (status === "OK") return "Exploitable";
  if (status === "DEGRADED") return "Dégradée";
  if (status === "UNUSABLE") return "Inexploitable";
  return "Inconnue";
}

function streamLabel(status) {
  if (status === "ACTIVE") return "Fonctionnel";
  if (status === "INTERRUPTED") return "Interrompu";
  if (status === "UNAVAILABLE") return "Indisponible";
  return "Inconnu";
}

function render(result) {
  const events = result.quality_events || [];
  const last = events[events.length - 1] || {};
  els.sourceLabel.textContent = `Source : ${result.source || "inconnue"}`;
  els.stream.textContent = streamLabel(result.signal_status || last.stream_status);
  els.speech.textContent = speechLabel(events);
  els.rate.textContent = `${(result.output_rate || 16000).toLocaleString("fr-FR")} Hz`;
  els.quality.textContent = qualityLabel(last.quality_status);
  els.quality.className = qualityClass(last.quality_status);

  const timeline = result.timeline || [];
  if (!timeline.length) {
    els.rows.innerHTML = `<tr class="empty"><td colspan="4">Aucun segment publié.</td></tr>`;
  } else {
    els.rows.innerHTML = timeline
      .map(
        (row) => `<tr>
          <td>${row.start_s.toFixed(1)}–${row.end_s.toFixed(1)} s</td>
          <td>${row.label}</td>
          <td class="${qualityClass(row.quality_status)}">${qualityLabel(row.quality_status)}</td>
          <td>${row.reason}</td>
        </tr>`
      )
      .join("");
  }
  els.summary.textContent = `${result.chunks_generated || 0} blocs · ${
    (result.speech_segments || []).length
  } segments de parole · ${result.timestamp_errors || 0} erreur(s) d'horodatage`;

  const incidents = result.incidents || [];
  els.incidents.innerHTML = incidents.length
    ? incidents.map((item) => `<li>${item}</li>`).join("")
    : `<li>Aucun incident signalé.</li>`;

    els.contracts.innerHTML = `
    <li>Session ${result.session_id || "—"} · VAD ${result.vad_backend || "—"}</li>
    <li>Latence médiane ${(result.latency?.median_s || 0).toFixed(3)} s · p95 ${(result.latency?.p95_s || 0).toFixed(3)} s</li>
    <li>Pertes publisher ${result.publisher_drops || 0} · alignement parole ${result.speech_alignment_errors || 0}</li>
    <li>${result.disclaimer || "Aucun échantillon brut n'est envoyé hors de la machine."}</li>
  `;
  drawWave(result);
}

function drawWave(result) {
  const ctx = els.wave.getContext("2d");
  const width = els.wave.width;
  const height = els.wave.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101815";
  ctx.fillRect(0, 0, width, height);

  const wave = result.waveform || {};
  const times = wave.times || [];
  const mins = wave.min || [];
  const maxs = wave.max || [];
  const duration = result.duration_s || times[times.length - 1] || 1;

  for (const row of result.timeline || []) {
    const x0 = (row.start_s / duration) * width;
    const x1 = (row.end_s / duration) * width;
    if (row.quality_status === "DEGRADED" || row.quality_status === "UNUSABLE") {
      ctx.fillStyle = row.quality_status === "UNUSABLE" ? "rgba(225,107,91,0.16)" : "rgba(227,179,65,0.14)";
    } else {
      ctx.fillStyle = row.kind === "speech" ? "rgba(62,207,142,0.16)" : "rgba(111,143,156,0.10)";
    }
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), height);
  }

  if (!times.length) return;
  ctx.beginPath();
  ctx.strokeStyle = "rgba(215,239,227,0.85)";
  ctx.lineWidth = 1.4;
  const mid = height / 2;
  const amp = height * 0.42;
  for (let i = 0; i < times.length; i += 1) {
    const x = (times[i] / duration) * width;
    const y = mid - maxs[i] * amp;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let i = times.length - 1; i >= 0; i -= 1) {
    const x = (times[i] / duration) * width;
    const y = mid - mins[i] * amp;
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(126,224,176,0.28)";
  ctx.fill();
  ctx.stroke();
}

async function parseError(response) {
  try {
    const data = await response.json();
    return data.detail || JSON.stringify(data);
  } catch {
    return response.statusText;
  }
}

async function runAnalysis() {
  const mode = els.source.value;
  els.run.disabled = true;
  try {
    if (mode === "demo") {
      const response = await fetch("/api/inspect/demo", { method: "POST" });
      if (!response.ok) throw new Error(await parseError(response));
      render(await response.json());
      return;
    }
    if (mode === "file") {
      if (!els.file.files[0]) throw new Error("Choisissez un fichier WAV local.");
      const body = new FormData();
      body.append("file", els.file.files[0]);
      body.append("energy_vad", els.energyVad.checked ? "true" : "false");
      const response = await fetch("/api/inspect/file", { method: "POST", body });
      if (!response.ok) throw new Error(await parseError(response));
      render(await response.json());
      return;
    }
    if (!els.consent.checked) {
      throw new Error("Le consentement est obligatoire avant une capture microphone.");
    }
    const body = new FormData();
    body.append("device_id", els.device.value);
    body.append("consent", "true");
    body.append("energy_vad", els.energyVad.checked ? "true" : "false");
    const response = await fetch("/api/session/start", { method: "POST", body });
    if (!response.ok) throw new Error(await parseError(response));
    els.stop.classList.remove("hidden");
    els.sourceLabel.textContent = "Source : microphone actif";
    pollTimer = setInterval(pollLive, 1000);
  } catch (error) {
    els.summary.textContent = error.message;
    els.incidents.innerHTML = `<li>${error.message}</li>`;
  } finally {
    els.run.disabled = false;
  }
}

async function pollLive() {
  const response = await fetch("/api/session/status");
  if (!response.ok) return;
  const data = await response.json();
  if (data.status) {
    els.stream.textContent = streamLabel(data.status.stream_status);
    els.speech.textContent =
      data.status.last_speech_activity === "SPEECH" ? "Parole détectée" : "Silence";
    els.quality.textContent = qualityLabel(data.status.last_quality_status);
  }
  if (data.result) {
    render(data.result);
    stopPolling();
  }
}

async function stopCapture() {
  stopPolling();
  const response = await fetch("/api/session/stop", { method: "POST" });
  if (!response.ok) {
    els.summary.textContent = await parseError(response);
    return;
  }
  render(await response.json());
  els.stop.classList.add("hidden");
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

els.source.addEventListener("change", syncSourceFields);
els.run.addEventListener("click", runAnalysis);
els.stop.addEventListener("click", stopCapture);
syncSourceFields();
loadDevices();
