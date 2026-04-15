#!/usr/bin/env node
/**
 * gemini_adapter.js  (Role B — Examinee)
 *
 * Bridges the orchestrator (WebRTC @ 48 kHz PCM16) to Google Gemini Live
 * via LiteLLM's OpenAI-compatible Realtime WebSocket endpoint.
 *
 * Architecture
 *   Orchestrator ←WebRTC→ [this process] ←WebSocket→ LiteLLM → Gemini Live
 *
 * Audio path
 *   Uplink:   48 kHz PCM16  → downsample → 16 kHz PCM16 → base64 → WS send
 *   Downlink: base64 from WS → 24 kHz PCM16 → upsample → 48 kHz PCM16 → WebRTC
 *
 * LiteLLM endpoint
 *   ws[s]://{LITELLM_BASE}/v1/realtime?model=gemini/gemini-2.0-flash-live-001
 *   Headers: Authorization: Bearer {LITELLM_API_KEY}
 *            OpenAI-Beta: realtime=v1
 *
 * Env (all overridable via CLI flags)
 *   LITELLM_BASE_URL   - LiteLLM proxy base, e.g. http://localhost:4000
 *   LITELLM_API_KEY    - LiteLLM API key
 *   WIRE_SAMPLE_RATE   - orchestrator wire rate (default 48000)
 */

require('dotenv').config();
const wrtc      = require('wrtc');
const WebSocket = require('ws');
const yargs     = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;

const DEFAULT_LITELLM_BASE = process.env.LITELLM_BASE_URL  || 'http://localhost:4000';
const DEFAULT_LITELLM_KEY  = process.env.LITELLM_API_KEY   || '';
const DEFAULT_MODEL        = 'gemini/gemini-live-2.5-flash-preview-native-audio-09-2025';
const WIRE_SR              = parseInt(process.env.WIRE_SAMPLE_RATE, 10) || 48000;

// Gemini Live native rates (through LiteLLM's compatibility layer)
const GEMINI_INPUT_SR  = 16000; // input expected at 16 kHz
const GEMINI_OUTPUT_SR = 24000; // output delivered at 24 kHz

const argv = yargs(hideBin(process.argv))
  .scriptName('gemini_adapter')
  .option('role',           { alias: 'r', type: 'string', choices: ['A', 'B'], default: 'B' })
  .option('signalUrl',      { alias: 's', type: 'string', demandOption: true })
  .option('litellmBaseUrl', { type: 'string', default: DEFAULT_LITELLM_BASE })
  .option('litellmApiKey',  { type: 'string', default: DEFAULT_LITELLM_KEY })
  .option('model',          { type: 'string', default: DEFAULT_MODEL })
  .option('systemPrompt',   { type: 'string', default: '' })
  .option('logEvents',      { type: 'boolean', default: false })
  .help().argv;

const ROLE          = argv.role;
const SIGNAL_URL    = argv.signalUrl;
const LITELLM_BASE  = argv.litellmBaseUrl.replace(/\/$/, '');
const LITELLM_KEY   = argv.litellmApiKey;
const MODEL         = argv.model;
const SYSTEM_PROMPT = argv.systemPrompt || '';

if (!LITELLM_KEY) {
  console.error('[GeminiAdapter] Missing --litellmApiKey / LITELLM_API_KEY');
  process.exit(1);
}

// ------------- Globals -------------
let orchPC, orchSource, orchSink;    // orchestrator WebRTC leg
let geminiWS;                        // LiteLLM WebSocket leg
let ws;                              // signaling socket to orchestrator

let uplinkBuf   = Buffer.alloc(0);   // PCM16 @ 48 kHz from orchestrator
let downlinkBuf = Buffer.alloc(0);   // PCM16 @ 48 kHz to orchestrator (upsampled)
let uplinkPacer   = null;
let downlinkPacer = null;

let sessionReady   = false;          // session.created / session.updated received
let bridgeStartTime = Date.now();

process.on('SIGINT',  shutdown);
process.on('SIGTERM', shutdown);

(async function main() {
  console.log(`[GeminiAdapter-${ROLE}]  model=${MODEL}  litellm=${LITELLM_BASE}`);
  await connectOrchestrator();
  await connectGemini();
  startBridgePacers();
})();

// ------------- Resampling helpers -------------

/**
 * Downsample PCM16 from 48 kHz to 16 kHz (integer decimation, factor 3).
 * Simple but sufficient for speech; no anti-aliasing filter needed at these rates.
 */
function downsample48to16(buf) {
  const src = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength >> 1);
  const out = new Int16Array(Math.floor(src.length / 3));
  for (let i = 0; i < out.length; i++) out[i] = src[i * 3];
  return Buffer.from(out.buffer);
}

/**
 * Upsample PCM16 from 24 kHz to 48 kHz (integer interpolation, factor 2).
 * Linear interpolation between adjacent samples avoids hard discontinuities.
 */
function upsample24to48(buf) {
  const src = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength >> 1);
  const out = new Int16Array(src.length * 2);
  for (let i = 0; i < src.length; i++) {
    out[i * 2]     = src[i];
    out[i * 2 + 1] = i + 1 < src.length
      ? Math.round((src[i] + src[i + 1]) / 2)
      : src[i];
  }
  return Buffer.from(out.buffer);
}

// ------------- Orchestrator leg (WebRTC) — identical to gptRealtime_adapter -------------
async function connectOrchestrator() {
  orchPC     = new wrtc.RTCPeerConnection();
  orchSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
  orchPC.addTrack(orchSource.createTrack());

  orchPC.ontrack = ({ track }) => {
    if (track.kind !== 'audio') return;
    if (orchSink) orchSink.stop();
    orchSink = new RTCAudioSink(track, { sampleRate: WIRE_SR, channelCount: 1, bitDepth: 16 });
    orchSink.ondata = ({ samples }) => {
      uplinkBuf = Buffer.concat([uplinkBuf, Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength)]);
    };
  };

  ws = new WebSocket(SIGNAL_URL);
  ws.on('error', (err) => console.error(`[GeminiAdapter-${ROLE}] Orchestrator WS error:`, err));
  await new Promise((res) => ws.once('open', res));

  orchPC.onicecandidate = ({ candidate }) => {
    if (!candidate) return;
    ws.send(JSON.stringify({
      role: ROLE, type: 'candidate',
      candidate: { candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex },
    }));
  };

  const offer = await orchPC.createOffer();
  await orchPC.setLocalDescription(offer);
  ws.send(JSON.stringify({ role: ROLE, type: 'offer', sdp: offer.sdp }));

  ws.on('message', async (msg) => {
    const data = JSON.parse(msg);
    if (data.type === 'answer' && data.role === ROLE) {
      await orchPC.setRemoteDescription({ type: 'answer', sdp: data.sdp });
      console.log(`[GeminiAdapter-${ROLE}] Orchestrator answer applied.`);
    } else if (data.type === 'candidate' && data.role === ROLE) {
      try { await orchPC.addIceCandidate(data.candidate); } catch {}
    }
  });
}

// ------------- Gemini Live leg (WebSocket via LiteLLM) -------------
async function connectGemini() {
  // Build WebSocket URL: http → ws, https → wss
  const wsBase = LITELLM_BASE
    .replace(/^https:\/\//, 'wss://')
    .replace(/^http:\/\//, 'ws://');
  const wsUrl = `${wsBase}/realtime?model=${encodeURIComponent(MODEL)}`;

  console.log(`[GeminiAdapter-${ROLE}] Connecting to ${wsUrl}`);

  geminiWS = new WebSocket(wsUrl, {
    headers: {
      'Authorization': `Bearer ${LITELLM_KEY}`,
      'OpenAI-Beta':   'realtime=v1',
    },
  });

  geminiWS.on('error', (err) => console.error(`[GeminiAdapter-${ROLE}] Gemini WS error:`, err));

  geminiWS.on('open', () => {
    console.log(`[GeminiAdapter-${ROLE}] WebSocket open.`);
    // Session config — mirror the OpenAI Realtime session.update format;
    // LiteLLM translates this to Gemini Live's native setup message.
    geminiWS.send(JSON.stringify({
      type: 'session.update',
      session: {
        modalities:                ['text', 'audio'],
        instructions:              SYSTEM_PROMPT,
        input_audio_format:        'pcm16',
        output_audio_format:       'pcm16',
        turn_detection: {
          type:               'server_vad',
          create_response:    true,
          interrupt_response: true,
        },
      },
    }));
  });

  geminiWS.on('message', (raw) => {
    let evt;
    try { evt = JSON.parse(raw); } catch { return; }

    if (argv.logEvents) console.log(`[GeminiAdapter-${ROLE}] ←`, evt.type);

    switch (evt.type) {
      case 'session.created':
      case 'session.updated':
        sessionReady = true;
        console.log(`[GeminiAdapter-${ROLE}] Session ready.`);
        break;

      case 'response.audio.delta':
        if (evt.delta) {
          // Decode base64 PCM16 @ 24 kHz, upsample to 48 kHz
          const raw24 = Buffer.from(evt.delta, 'base64');
          downlinkBuf = Buffer.concat([downlinkBuf, upsample24to48(raw24)]);
        }
        break;

      case 'error':
        console.error(`[GeminiAdapter-${ROLE}] Gemini error:`, JSON.stringify(evt.error));
        break;

      default:
        break;
    }
  });

  geminiWS.on('close', (code, reason) => {
    console.log(`[GeminiAdapter-${ROLE}] Gemini WS closed: ${code} ${reason}`);
  });
}

// ------------- Bridge pacers -------------
function startBridgePacers() {
  if (uplinkPacer)   clearInterval(uplinkPacer);
  if (downlinkPacer) clearInterval(downlinkPacer);

  bridgeStartTime = Date.now();
  const FRAME_10MS_48K = Math.floor(WIRE_SR * 0.01) * 2; // bytes per 10 ms at 48 kHz

  // Collect 100 ms of orchestrator audio, downsample, send to Gemini via WebSocket
  uplinkPacer = setInterval(() => {
    // Mute first 4 s so Gemini doesn't respond before the Examiner speaks
    const shouldMute = Date.now() - bridgeStartTime < 4000;

    let collected = Buffer.alloc(0);
    for (let i = 0; i < 10; i++) {
      let frame;
      if (uplinkBuf.length >= FRAME_10MS_48K) {
        frame = uplinkBuf.slice(0, FRAME_10MS_48K);
        uplinkBuf = uplinkBuf.slice(FRAME_10MS_48K);
      } else if (uplinkBuf.length > 0) {
        frame = Buffer.alloc(FRAME_10MS_48K);
        uplinkBuf.copy(frame, 0, 0, uplinkBuf.length);
        uplinkBuf = Buffer.alloc(0);
      } else {
        frame = Buffer.alloc(FRAME_10MS_48K);
      }
      if (shouldMute) frame.fill(0);
      collected = Buffer.concat([collected, frame]);
    }

    // Downsample 48 kHz → 16 kHz and send as base64 chunk
    if (sessionReady && geminiWS && geminiWS.readyState === WebSocket.OPEN) {
      const chunk16k = downsample48to16(collected);
      geminiWS.send(JSON.stringify({
        type:  'input_audio_buffer.append',
        audio: chunk16k.toString('base64'),
      }));
    }
  }, 100);

  // Drain downlink buffer into orchestrator WebRTC source at 10 ms granularity
  downlinkPacer = setInterval(() => {
    for (let i = 0; i < 10; i++) {
      let out;
      if (downlinkBuf.length >= FRAME_10MS_48K) {
        out = downlinkBuf.slice(0, FRAME_10MS_48K);
        downlinkBuf = downlinkBuf.slice(FRAME_10MS_48K);
      } else if (downlinkBuf.length > 0) {
        out = Buffer.alloc(FRAME_10MS_48K);
        downlinkBuf.copy(out, 0, 0, downlinkBuf.length);
        downlinkBuf = Buffer.alloc(0);
      } else {
        out = Buffer.alloc(FRAME_10MS_48K);
      }
      const ab = new ArrayBuffer(FRAME_10MS_48K);
      new Uint8Array(ab).set(out);
      orchSource.onData({ samples: new Int16Array(ab), sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });
    }
  }, 100);
}

// ------------- Shutdown -------------
function shutdown() {
  console.log(`[GeminiAdapter-${ROLE}] Shutting down...`);
  try { if (uplinkPacer)   clearInterval(uplinkPacer);   } catch {}
  try { if (downlinkPacer) clearInterval(downlinkPacer); } catch {}
  try { if (orchSink)  orchSink.stop();  } catch {}
  try { if (orchPC)    orchPC.close();   } catch {}
  try { if (geminiWS && geminiWS.readyState === WebSocket.OPEN) geminiWS.close(); } catch {}
  try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch {}
  process.exit(0);
}
