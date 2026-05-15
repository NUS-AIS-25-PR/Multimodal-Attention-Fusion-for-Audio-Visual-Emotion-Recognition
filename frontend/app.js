/**
 * CareVision – Emotion Recognition Frontend
 * Elderly Care Emotion Monitoring System
 */

/* ── Config ── */
const CONFIG = {
    BACKEND_URL:            resolveBackendUrl(),
    PREDICT_ENDPOINT:       "/predict",
    STREAM_ENDPOINT:        "/ws/stream",
    STREAM_FRAME_INTERVAL_MS: 250,
    AUDIO_BUFFER_SIZE:      2048,
    MAX_HISTORY:            10,
    ALERT_THRESHOLD:        55,   // confidence% above which alert triggers
};

/* ── Emotion metadata ── */
const EMOTION_CONFIG = {
    neutral:   { icon: "😐", display: "Neutral",   color: "#6B7280", bg: "#F9FAFB", alert: false },
    calm:      { icon: "😌", display: "Calm",       color: "#0284C7", bg: "#EFF6FF", alert: false },
    happy:     { icon: "😊", display: "Happy",      color: "#16A34A", bg: "#F0FDF4", alert: false },
    sad:       { icon: "😢", display: "Sad",        color: "#475569", bg: "#F8FAFC", alert: true  },
    angry:     { icon: "😠", display: "Angry",      color: "#DC2626", bg: "#FEF2F2", alert: true  },
    fearful:   { icon: "😨", display: "Fearful",    color: "#7C3AED", bg: "#F5F3FF", alert: true  },
    disgust:   { icon: "🤢", display: "Disgust",    color: "#B45309", bg: "#FFFBEB", alert: true  },
    surprised: { icon: "😮", display: "Surprised",  color: "#0891B2", bg: "#ECFEFF", alert: false },
};

/* ── Helpers ── */
function resolveBackendUrl() {
    const fromQuery  = typeof window !== "undefined"
        ? new URLSearchParams(window.location.search).get("backend") : "";
    const fromWindow = typeof window !== "undefined" ? window.__EMO_BACKEND_URL : "";
    const fallback   = typeof window !== "undefined"
        ? `${window.location.protocol}//${window.location.hostname}:8000`
        : "http://localhost:8000";
    return (fromQuery || fromWindow || fallback).replace(/\/$/, "");
}

function toWebSocketUrl(httpUrl, path) {
    const url = new URL(path, httpUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
}

function float32ToInt16(buf) {
    const out = new Int16Array(buf.length);
    for (let i = 0; i < buf.length; i++) {
        const s = Math.max(-1, Math.min(1, buf[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
}

function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let b = "";
    for (let i = 0; i < bytes.length; i++) b += String.fromCharCode(bytes[i]);
    return btoa(b);
}

/* ═══════════════════════════════════════════
   Main App Class
   ═══════════════════════════════════════════ */
class EmotionRecognitionApp {
    constructor(cfg) {
        this.config = cfg;

        // State
        this.stream          = null;
        this.websocket       = null;
        this.streaming       = false;
        this.streamStartTime = 0;
        this.frameInterval   = null;
        this.timerInterval   = null;
        this.clockInterval   = null;
        this.audioContext    = null;
        this.audioSource     = null;
        this.audioProcessor  = null;
        this.canvas          = document.createElement("canvas");
        this.emotionHistory  = [];   // [{label, cfg}]

        // DOM refs – camera side
        this.preview     = document.getElementById("preview");
        this.startBtn    = document.getElementById("startBtn");
        this.stopBtn     = document.getElementById("stopBtn");
        this.uploadBtn   = document.getElementById("uploadBtn");
        this.timerEl     = document.getElementById("timer");
        this.liveBadgeEl = document.getElementById("liveBadge");

        // DOM refs – status
        this.statusDotEl  = document.getElementById("statusDot");    // header dot
        this.statusTextEl = document.getElementById("statusText");   // header label
        this.sessionEl    = document.getElementById("status");       // session panel text

        // DOM refs – hero
        this.emotionCardEl  = document.getElementById("emotionCard");
        this.heroWaitingEl  = document.getElementById("heroWaiting");
        this.heroResultEl   = document.getElementById("heroResult");
        this.emotionIconEl  = document.getElementById("emotionIcon");
        this.emotionNameEl  = document.getElementById("emotionName");
        // emotionNameZh element removed from HTML (English-only UI)
        this.emotionConfEl  = document.getElementById("emotionConf");
        this.confArcEl      = document.getElementById("confArc");
        this.alertBannerEl  = document.getElementById("alertBanner");

        // DOM refs – history + breakdown
        this.historyDotsEl     = document.getElementById("historyDots");
        this.emotionBreakdownEl = document.getElementById("emotionBreakdown");
    }

    /* ─────────────────────────────────────
       Init
    ───────────────────────────────────── */
    async initialize() {
        this.startClock();
        await this.initMediaStream();
        this.bindEvents();
        await this.checkHealth();
    }

    /* ─────────────────────────────────────
       Clock
    ───────────────────────────────────── */
    startClock() {
        const tick = () => {
            const d = new Date();
            const pad = (n) => String(n).padStart(2, "0");
            const clockEl = document.getElementById("clock");
            if (clockEl) clockEl.textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
        };
        tick();
        this.clockInterval = setInterval(tick, 1000);
    }

    /* ─────────────────────────────────────
       Status helpers
    ───────────────────────────────────── */
    setStatus(state, headerText, sessionText) {
        // state: 'idle' | 'active' | 'error'
        if (this.statusDotEl)  this.statusDotEl.className = `status-dot ${state}`;
        if (this.statusTextEl) this.statusTextEl.textContent = headerText;
        if (sessionText !== undefined && this.sessionEl) this.sessionEl.textContent = sessionText;
    }

    setSessionText(text) {
        if (this.sessionEl) this.sessionEl.textContent = text;
    }

    /* ─────────────────────────────────────
       Media stream
    ───────────────────────────────────── */
    async initMediaStream() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240 },
                audio: true,
            });
            this.preview.srcObject = this.stream;
            this.setStatus("idle", "System Ready", "Camera connected. Click Start Monitoring.");
            this.startBtn.disabled  = false;
            this.uploadBtn.disabled = false;
        } catch (err) {
            this.setStatus("error", "Camera Error", `Camera access failed: ${err.message}`);
        }
    }

    /* ─────────────────────────────────────
       Event listeners
    ───────────────────────────────────── */
    bindEvents() {
        this.startBtn.addEventListener("click",  () => this.startStreaming());
        this.stopBtn.addEventListener("click",   () => this.stopStreaming());
        this.uploadBtn.addEventListener("click", () => this.runOneShotPrediction());
    }

    /* ─────────────────────────────────────
       Streaming
    ───────────────────────────────────── */
    async startStreaming() {
        if (!this.stream || this.streaming) return;

        // Prevent repeated clicks while connection is being established
        this.startBtn.disabled = true;

        const wsUrl = toWebSocketUrl(this.config.BACKEND_URL, this.config.STREAM_ENDPOINT);
        this.websocket = new WebSocket(wsUrl);
        this.setSessionText("Connecting to server...");

        this.websocket.onopen = async () => {
            this.streaming       = true;
            this.streamStartTime = performance.now();

            this.startBtn.disabled  = true;
            this.stopBtn.disabled   = false;
            this.uploadBtn.disabled = true;

            if (this.liveBadgeEl) this.liveBadgeEl.classList.add("active");
            this.setStatus("active", "Live Monitoring", "Monitoring active, awaiting first prediction...");
            this.showHeroWaiting("Starting Monitor", "Waiting for first emotion analysis...");

            this.websocket.send(JSON.stringify({ type: "start", timestamp: Date.now() / 1000 }));
            this.startTimer();
            this.startFrameStreaming();
            await this.startAudioStreaming();
        };

        this.websocket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === "prediction") {
                    this.displayResults(msg.payload);
                    this.setSessionText("Prediction updated");
                } else if (msg.type === "error") {
                    this.setSessionText(`Recognition error: ${msg.detail}`);
                }
            } catch (_) { /* ignore malformed messages */ }
        };

        this.websocket.onerror = () => {
            this.setStatus("error", "Connection Error", "WebSocket connection failed. Check backend service.");
        };

        this.websocket.onclose = () => {
            this.cleanupStreaming();
        };
    }

    stopStreaming() {
        if (!this.streaming) return;
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type: "flush", timestamp: Date.now() / 1000 }));
            this.websocket.send(JSON.stringify({ type: "stop",  timestamp: Date.now() / 1000 }));
            this.websocket.close();
        } else {
            this.cleanupStreaming();
        }
    }

    cleanupStreaming() {
        this.streaming = false;

        this.startBtn.disabled  = false;
        this.stopBtn.disabled   = true;
        this.uploadBtn.disabled = false;

        if (this.liveBadgeEl) this.liveBadgeEl.classList.remove("active");
        this.setStatus("idle", "Monitoring Stopped", "Session ended");

        clearInterval(this.frameInterval);
        this.frameInterval = null;
        this.stopTimer();

        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor.onaudioprocess = null;
            this.audioProcessor = null;
        }
        if (this.audioSource)  { this.audioSource.disconnect();  this.audioSource  = null; }
        if (this.audioContext) { this.audioContext.close();       this.audioContext = null; }
        this.websocket = null;
    }

    startFrameStreaming() {
        const w = this.preview.videoWidth  || 320;
        const h = this.preview.videoHeight || 240;
        this.canvas.width  = w;
        this.canvas.height = h;
        const ctx = this.canvas.getContext("2d");

        this.frameInterval = setInterval(() => {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) return;
            ctx.drawImage(this.preview, 0, 0, w, h);
            this.websocket.send(JSON.stringify({
                type:      "frame",
                timestamp: Date.now() / 1000,
                image_b64: this.canvas.toDataURL("image/jpeg", 0.7),
            }));
        }, this.config.STREAM_FRAME_INTERVAL_MS);
    }

    async startAudioStreaming() {
        this.audioContext   = new (window.AudioContext || window.webkitAudioContext)();
        this.audioSource    = this.audioContext.createMediaStreamSource(this.stream);
        this.audioProcessor = this.audioContext.createScriptProcessor(this.config.AUDIO_BUFFER_SIZE, 1, 1);

        this.audioProcessor.onaudioprocess = (ev) => {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) return;
            const pcm16 = float32ToInt16(ev.inputBuffer.getChannelData(0));
            this.websocket.send(JSON.stringify({
                type:        "audio",
                timestamp:   Date.now() / 1000,
                sample_rate: this.audioContext.sampleRate,
                pcm_b64:     arrayBufferToBase64(pcm16.buffer),
            }));
        };

        this.audioSource.connect(this.audioProcessor);
        this.audioProcessor.connect(this.audioContext.destination);
    }

    /* ─────────────────────────────────────
       One-shot prediction
    ───────────────────────────────────── */
    async runOneShotPrediction() {
        if (!this.stream) { this.setSessionText("No camera available"); return; }

        this.startBtn.disabled  = true;
        this.uploadBtn.disabled = true;
        this.setStatus("active", "Sampling", "Recording 3-second clip...");
        this.showHeroWaiting("Recording...", "Capturing audio/video, hold your expression");

        try {
            const chunks   = [];
            const recorder = new MediaRecorder(this.stream, { mimeType: "video/webm" });
            recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
            const done = new Promise((res) => { recorder.onstop = res; });
            recorder.start();
            await new Promise((res) => setTimeout(res, 3000));
            recorder.stop();
            await done;

            this.setSessionText("Analyzing emotion, please wait...");
            this.showHeroWaiting("Analyzing...", "AI is processing data");

            const blob = new Blob(chunks, { type: "video/webm" });
            const fd   = new FormData();
            fd.append("file", blob, "recording.webm");

            const res = await fetch(`${this.config.BACKEND_URL}${this.config.PREDICT_ENDPOINT}`, {
                method: "POST",
                body:   fd,
            });
            if (!res.ok) throw new Error(`Server error ${res.status}`);

            this.displayResults(await res.json());
            this.setStatus("idle", "Scan Complete", "Quick scan analysis complete");

        } catch (err) {
            this.setStatus("error", "Scan Failed", `${err.message}`);
            this.showHeroWaiting("Scan Failed", err.message);
        } finally {
            this.startBtn.disabled  = false;
            this.uploadBtn.disabled = false;
        }
    }

    /* ─────────────────────────────────────
       Hero display helpers
    ───────────────────────────────────── */
    showHeroWaiting(title, sub) {
        if (this.heroResultEl)  this.heroResultEl.classList.add("hidden");
        if (this.heroWaitingEl) {
            this.heroWaitingEl.classList.remove("hidden");
            const t = this.heroWaitingEl.querySelector(".waiting-text");
            const s = this.heroWaitingEl.querySelector(".waiting-sub");
            if (t) t.textContent = title;
            if (s) s.textContent = sub;
        }
        if (this.emotionCardEl) {
            this.emotionCardEl.style.backgroundColor = "";
            this.emotionCardEl.style.borderColor     = "";
            this.emotionCardEl.classList.remove("alert");
        }
    }

    /* ─────────────────────────────────────
       Display results
    ───────────────────────────────────── */
    displayResults(result) {
        if (result.error) {
            this.setSessionText(`Recognition error: ${result.error}`);
            return;
        }

        const { labels, probs, top1 } = result;
        const cfg  = EMOTION_CONFIG[top1.label] || {
            icon: "❓", display: top1.label,
            color: "#6B7280", bg: "#F9FAFB", alert: false,
        };
        const conf = Math.round(top1.prob);

        /* Show result, hide waiting */
        if (this.heroWaitingEl) this.heroWaitingEl.classList.add("hidden");
        if (this.heroResultEl)  this.heroResultEl.classList.remove("hidden");

        /* Animate icon on change */
        if (this.emotionIconEl) {
            const prev = this.emotionIconEl.textContent;
            if (prev !== cfg.icon) {
                this.emotionIconEl.style.animation = "none";
                void this.emotionIconEl.offsetHeight; // reflow
                this.emotionIconEl.style.animation = "pop-in 0.35s ease";
            }
            this.emotionIconEl.textContent = cfg.icon;
        }

        /* Labels */
        if (this.emotionNameEl) {
            this.emotionNameEl.textContent = top1.label.toUpperCase();
            this.emotionNameEl.style.color = cfg.color;
        }

        /* Confidence ring */
        if (this.emotionConfEl) this.emotionConfEl.textContent = `${conf}%`;
        if (this.confArcEl) {
            this.confArcEl.setAttribute("stroke-dasharray", `${conf}, 100`);
            this.confArcEl.style.stroke = cfg.color;
        }

        /* Hero card colour */
        if (this.emotionCardEl) {
            this.emotionCardEl.style.backgroundColor = cfg.bg;
            this.emotionCardEl.style.borderColor     = cfg.color + "55";
        }

        /* Alert */
        const isAlert = cfg.alert && conf >= this.config.ALERT_THRESHOLD;
        if (this.alertBannerEl) {
            isAlert
                ? this.alertBannerEl.classList.remove("hidden")
                : this.alertBannerEl.classList.add("hidden");
        }
        if (this.emotionCardEl) {
            isAlert
                ? this.emotionCardEl.classList.add("alert")
                : this.emotionCardEl.classList.remove("alert");
        }

        /* History */
        this.emotionHistory.push({ label: top1.label, cfg });
        if (this.emotionHistory.length > this.config.MAX_HISTORY) this.emotionHistory.shift();
        this.renderHistory();

        /* Breakdown bars */
        this.renderBreakdown(labels, probs, top1.label);
    }

    /* ─────────────────────────────────────
       History dots
    ───────────────────────────────────── */
    renderHistory() {
        if (!this.historyDotsEl) return;
        if (!this.emotionHistory.length) {
            this.historyDotsEl.innerHTML = '<span class="history-empty">No history yet</span>';
            return;
        }
        this.historyDotsEl.innerHTML = this.emotionHistory.map((h) =>
            `<div class="history-dot"
                  style="background:${h.cfg.bg};border-color:${h.cfg.color}"
                  title="${h.cfg.display} (${h.label})">${h.cfg.icon}</div>`
        ).join("");
    }

    /* ─────────────────────────────────────
       Breakdown bars
    ───────────────────────────────────── */
    renderBreakdown(labels, probs, topLabel) {
        if (!this.emotionBreakdownEl) return;
        const html = labels.map((label, i) => {
            const prob = probs[i];
            const cfg  = EMOTION_CONFIG[label] || { icon: "●", display: label, color: "#6B7280" };
            const w    = Math.max(0, prob).toFixed(1);
            const top  = label === topLabel ? " bar-top" : "";
            return `
                <div class="bar-row${top}">
                    <span class="bar-icon">${cfg.icon}</span>
                    <span class="bar-label">${cfg.display}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:${w}%;background:${cfg.color}"></div>
                    </div>
                    <span class="bar-pct">${Math.round(prob)}%</span>
                </div>`;
        }).join("");
        this.emotionBreakdownEl.innerHTML = html;
    }

    /* ─────────────────────────────────────
       Session timer
    ───────────────────────────────────── */
    startTimer() {
        this.stopTimer();
        if (this.timerEl) this.timerEl.textContent = "00:00";
        this.timerInterval = setInterval(() => {
            const s   = Math.floor((performance.now() - this.streamStartTime) / 1000);
            const mm  = String(Math.floor(s / 60)).padStart(2, "0");
            const ss  = String(s % 60).padStart(2, "0");
            if (this.timerEl) this.timerEl.textContent = `${mm}:${ss}`;
        }, 1000);
    }

    stopTimer() {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
        if (this.timerEl) this.timerEl.textContent = "";
    }

    /* ─────────────────────────────────────
       Health check
    ───────────────────────────────────── */
    async checkHealth() {
        try {
            const res = await fetch(`${this.config.BACKEND_URL}/health`);
            if (res.ok) {
                const data = await res.json();
                if (data.mock_mode) {
                    this.setSessionText("⚠ Demo mode: no real model loaded, predictions are random");
                }
            }
        } catch (_) {
            this.setStatus("error", "Backend Offline", "⚠ Cannot connect to backend. Make sure the server is running.");
        }
    }
}

/* ── Bootstrap ── */
document.addEventListener("DOMContentLoaded", async () => {
    const app = new EmotionRecognitionApp(CONFIG);
    await app.initialize();
});
