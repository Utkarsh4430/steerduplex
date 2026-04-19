#!/usr/bin/env node
/**
 * gptRealtime_adapter.js  (Role A — Examiner)
 *
 * Bridges orchestrator (WebRTC @ 48 kHz PCM16) <-> OpenAI Realtime
 * via LiteLLM WebSocket (same pattern as gemini_adapter.js).
 *
 * NOTE: Previously used WebRTC (SDP exchange) for the AI side, which
 * required a token-server to mint ephemeral keys via /v1/realtime/sessions.
 * That endpoint returns 404 on this LiteLLM proxy. The WebSocket endpoint
 * (/v1/realtime) works fine with a direct LITELLM_API_KEY, so we switched.
 *
 * Audio path
 *   Uplink:   48 kHz PCM16 → downsample ÷2 → 24 kHz PCM16 → base64 → WS send
 *   Downlink: base64 from WS → 24 kHz PCM16 → upsample ×2  → 48 kHz PCM16 → WebRTC
 *
 * Env (all overridable via CLI flags)
 *   LITELLM_BASE_URL      — LiteLLM proxy base, e.g. https://proxy/v1
 *   LITELLM_API_KEY       — LiteLLM API key (used directly; no token-server needed)
 *   OPENAI_REALTIME_MODEL — model string, e.g. openai/gpt-4o-realtime-preview-2024-12-17
 *   WIRE_SAMPLE_RATE      — orchestrator wire rate (default 48000)
 */

require('dotenv').config();
const fs   = require('fs');
const path = require('path');
const wrtc = require('wrtc');
const WebSocket = require('ws');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;

const DEFAULT_MODEL        = process.env.OPENAI_REALTIME_MODEL || 'openai/gpt-4o-realtime-preview-2024-12-17';
const DEFAULT_LITELLM_BASE = process.env.LITELLM_BASE_URL      || 'http://localhost:4000';
const DEFAULT_LITELLM_KEY  = process.env.LITELLM_API_KEY       || '';
const WIRE_SR              = parseInt(process.env.WIRE_SAMPLE_RATE, 10) || 48000;

// GPT Realtime native audio rate (input and output)
const GPT_SR = 24000;

const argv = yargs(hideBin(process.argv))
  .scriptName('gptRealtime_adapter')
  .option('role',           { alias: 'r', type: 'string', choices: ['A', 'B'], demandOption: true })
  .option('signalUrl',      { alias: 's', type: 'string', demandOption: true })
  .option('litellmBaseUrl', { type: 'string', default: DEFAULT_LITELLM_BASE })
  .option('litellmApiKey',  { type: 'string', default: DEFAULT_LITELLM_KEY })
  .option('model',          { type: 'string', default: DEFAULT_MODEL })
  .option('voice',          { type: 'string', default: 'alloy' })
  .option('systemPrompt',   { type: 'string', default: '' })
  .option('turnMode',       { type: 'string', choices: ['server_vad', 'none'], default: 'server_vad' })
  .option('vadMode',        { type: 'string', choices: ['slow', 'fast'], default: 'slow' })
  .option('logEvents',      { type: 'boolean', default: false })
  .option('prefill',        { type: 'string', default: '' })
  .option('prefillFile',    { type: 'string', default: '' })
  .option('autostart',      { type: 'boolean', default: true })
  // --tokenServer is accepted but no longer used (WebSocket needs no ephemeral token)
  .option('tokenServer',    { type: 'string', default: '' })
  .help().argv;

const ROLE          = argv.role;
const SIGNAL_URL    = argv.signalUrl;
const LITELLM_BASE  = argv.litellmBaseUrl.replace(/\/$/, '');
const LITELLM_KEY   = argv.litellmApiKey;
const MODEL         = argv.model;
const VOICE         = argv.voice;
const SYSTEM_PROMPT = argv.systemPrompt || '';
const TURN_MODE     = argv.turnMode;
const VAD_MODE      = argv.vadMode;
const PREFILL_TEXT  = argv.prefill || '';
const PREFILL_FILE  = argv.prefillFile || '';
const AUTOSTART     = argv.autostart !== undefined ? argv.autostart : (ROLE === 'A');

if (!LITELLM_KEY) {
  console.error('[GPTAdapter] Missing --litellmApiKey / LITELLM_API_KEY');
  process.exit(1);
}

// ------------- Globals -------------
let orchPC, orchSource, orchSink;
let gptWS;
let ws;   // signaling socket to orchestrator

let uplinkBuf   = Buffer.alloc(0);
let downlinkBuf = Buffer.alloc(0);
let uplinkPacer   = null;
let downlinkPacer = null;

let sessionReady = false;
let prefillSent  = false;
let bridgeStartTime = Date.now();

// ------------- Cost tracking (Role A only) -------------
const costAccumulator = {
  role:  ROLE,
  model: MODEL,
  response_count:            0,
  input_text_tokens:         0,
  input_audio_tokens:        0,
  input_cached_tokens:       0,
  input_cached_audio_tokens: 0,
  output_text_tokens:        0,
  output_audio_tokens:       0,
  responses: [],
};

const PRICING_PER_1M = {
  input_text:   5.00,
  input_audio:  40.00,
  cached_text:  2.50,
  cached_audio: 20.00,
  output_text:  20.00,
  output_audio: 80.00,
};

function recordUsage(evt) {
  const usage = evt?.response?.usage;
  if (!usage) return;
  const itd = usage.input_token_details  || {};
  const otd = usage.output_token_details || {};
  const ctd = itd.cached_tokens_details  || {};
  costAccumulator.response_count            += 1;
  costAccumulator.input_text_tokens         += itd.text_tokens   || 0;
  costAccumulator.input_audio_tokens        += itd.audio_tokens  || 0;
  costAccumulator.input_cached_tokens       += itd.cached_tokens || 0;
  costAccumulator.input_cached_audio_tokens += ctd.audio_tokens  || 0;
  costAccumulator.output_text_tokens        += otd.text_tokens   || 0;
  costAccumulator.output_audio_tokens       += otd.audio_tokens  || 0;
  costAccumulator.responses.push({ response_id: evt.response?.id, usage });
}

function writeCostMetadata() {
  const acc = costAccumulator;
  const cachedTextTokens  = acc.input_cached_tokens - acc.input_cached_audio_tokens;
  const cachedAudioTokens = acc.input_cached_audio_tokens;
  const estimated_cost_usd = (
    acc.input_text_tokens   * PRICING_PER_1M.input_text   / 1e6 +
    acc.input_audio_tokens  * PRICING_PER_1M.input_audio  / 1e6 +
    cachedTextTokens        * PRICING_PER_1M.cached_text  / 1e6 +
    cachedAudioTokens       * PRICING_PER_1M.cached_audio / 1e6 +
    acc.output_text_tokens  * PRICING_PER_1M.output_text  / 1e6 +
    acc.output_audio_tokens * PRICING_PER_1M.output_audio / 1e6
  );
  const metadata = {
    ...acc,
    estimated_cost_usd,
    pricing_note: 'Based on GPT-4o Realtime rates circa 2025 — verify at platform.openai.com/pricing.',
    recorded_at:  new Date().toISOString(),
  };
  const outDir  = process.env.RECORD_DIR || '.';
  const outFile = path.join(outDir, `cost_${ROLE}.json`);
  try {
    fs.writeFileSync(outFile, JSON.stringify(metadata, null, 2));
    console.log(`[GPTAdapter-${ROLE}] Cost metadata → ${outFile}  (~$${estimated_cost_usd.toFixed(5)})`);
  } catch (e) {
    console.error(`[GPTAdapter-${ROLE}] Failed to write cost metadata:`, e.message);
  }
}

process.on('SIGINT',  shutdown);
process.on('SIGTERM', shutdown);

(async function main() {
  console.log(`[GPTAdapter-${ROLE}]  model=${MODEL}  vad=${VAD_MODE}  litellm=${LITELLM_BASE}`);
  await connectOrchestrator();
  await connectGPT();
  startBridgePacers();
})();

// ------------- Resampling helpers -------------

/**
 * Downsample PCM16 from 48 kHz to 24 kHz (integer decimation, factor 2).
 */
function downsample48to24(buf) {
  const src = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength >> 1);
  const out = new Int16Array(Math.floor(src.length / 2));
  for (let i = 0; i < out.length; i++) out[i] = src[i * 2];
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

// ------------- Orchestrator leg (WebRTC) — unchanged -------------
async function connectOrchestrator() {
  orchPC     = new wrtc.RTCPeerConnection();
  orchSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
  orchPC.addTrack(orchSource.createTrack());

  orchPC.ontrack = ({ track }) => {
    if (track.kind !== 'audio') return;
    if (orchSink) orchSink.stop();
    orchSink = new RTCAudioSink(track, { sampleRate: WIRE_SR, channelCount: 1, bitDepth: 16 });
    orchSink.ondata = ({ samples }) => {
      uplinkBuf = Buffer.concat([uplinkBuf,
        Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength)]);
    };
  };

  ws = new WebSocket(SIGNAL_URL);
  ws.on('error', (err) => console.error(`[GPTAdapter-${ROLE}] Orchestrator WS error:`, err));
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
      console.log(`[GPTAdapter-${ROLE}] Orchestrator answer applied.`);
    } else if (data.type === 'candidate' && data.role === ROLE) {
      try { await orchPC.addIceCandidate(data.candidate); } catch {}
    }
  });
}

// ------------- GPT Realtime leg (WebSocket via LiteLLM) -------------
async function connectGPT() {
  const wsBase = LITELLM_BASE
    .replace(/^https:\/\//, 'wss://')
    .replace(/^http:\/\//, 'ws://');
  const wsUrl = `${wsBase}/realtime?model=${encodeURIComponent(MODEL)}`;

  console.log(`[GPTAdapter-${ROLE}] Connecting to ${wsUrl}`);

  gptWS = new WebSocket(wsUrl, {
    headers: {
      'Authorization': `Bearer ${LITELLM_KEY}`,
      'OpenAI-Beta':   'realtime=v1',
    },
  });

  gptWS.on('error', (err) => console.error(`[GPTAdapter-${ROLE}] GPT WS error:`, err));

  gptWS.on('open', () => {
    console.log(`[GPTAdapter-${ROLE}] WebSocket open.`);
    // Send session config right away; wait for session.created/updated to confirm
    sendSessionUpdate();
  });

  gptWS.on('message', (raw) => {
    let evt;
    try { evt = JSON.parse(raw); } catch { return; }

    if (argv.logEvents) console.log(`[GPTAdapter-${ROLE}] ←`, evt.type);

    switch (evt.type) {
      case 'session.created':
      case 'session.updated':
        if (!sessionReady) {
          sessionReady = true;
          console.log(`[GPTAdapter-${ROLE}] Session ready.`);
          if (!prefillSent) sendPrefillHistory();
        }
        break;

      case 'response.audio.delta':
        if (evt.delta) {
          const raw24 = Buffer.from(evt.delta, 'base64');
          downlinkBuf = Buffer.concat([downlinkBuf, upsample24to48(raw24)]);
        }
        break;

      case 'response.done':
        recordUsage(evt);
        break;

      case 'error':
        console.error(`[GPTAdapter-${ROLE}] GPT error:`, JSON.stringify(evt.error));
        break;

      default:
        break;
    }
  });

  gptWS.on('close', (code, reason) => {
    console.log(`[GPTAdapter-${ROLE}] GPT WS closed: ${code} ${reason}`);
  });
}

function sendSessionUpdate() {
  let turn_detection;
  if (TURN_MODE === 'none') {
    turn_detection = undefined;
  } else if (VAD_MODE === 'fast') {
    turn_detection = {
      type:               'server_vad',
      create_response:    true,
      interrupt_response: false,
      threshold:          0.6,
      prefix_padding_ms:  1200,
      silence_duration_ms: 600,
    };
  } else {
    turn_detection = {
      type:               'server_vad',
      create_response:    true,
      interrupt_response: true,
    };
  }

  const sessionConfig = {
    modalities:          ['text', 'audio'],
    instructions:        SYSTEM_PROMPT,
    voice:               VOICE,
    input_audio_format:  'pcm16',
    output_audio_format: 'pcm16',
  };
  if (turn_detection) sessionConfig.turn_detection = turn_detection;

  safeSend({ type: 'session.update', session: sessionConfig });
}

function safeSend(obj) {
  if (!gptWS || gptWS.readyState !== WebSocket.OPEN) return;
  try { gptWS.send(JSON.stringify(obj)); } catch (e) {
    console.error(`[GPTAdapter-${ROLE}] WS send error:`, e);
  }
}

// ------------- Bridge pacers -------------
function startBridgePacers() {
  if (uplinkPacer)   clearInterval(uplinkPacer);
  if (downlinkPacer) clearInterval(downlinkPacer);

  bridgeStartTime = Date.now();
  const FRAME_10MS_WIRE = Math.floor(WIRE_SR * 0.01) * 2; // bytes at 48 kHz

  // Orchestrator → GPT Realtime (collect 100 ms, downsample, send as base64)
  uplinkPacer = setInterval(() => {
    const shouldMute = (ROLE === 'B' && (Date.now() - bridgeStartTime < 4000));

    let collected = Buffer.alloc(0);
    for (let i = 0; i < 10; i++) {
      let frame;
      if (uplinkBuf.length >= FRAME_10MS_WIRE) {
        frame = uplinkBuf.slice(0, FRAME_10MS_WIRE);
        uplinkBuf = uplinkBuf.slice(FRAME_10MS_WIRE);
      } else if (uplinkBuf.length > 0) {
        frame = Buffer.alloc(FRAME_10MS_WIRE);
        uplinkBuf.copy(frame, 0, 0, uplinkBuf.length);
        uplinkBuf = Buffer.alloc(0);
      } else {
        frame = Buffer.alloc(FRAME_10MS_WIRE);
      }
      if (shouldMute) frame.fill(0);
      collected = Buffer.concat([collected, frame]);
    }

    if (sessionReady && gptWS && gptWS.readyState === WebSocket.OPEN) {
      const chunk24k = downsample48to24(collected);
      safeSend({ type: 'input_audio_buffer.append', audio: chunk24k.toString('base64') });
    }
  }, 100);

  // GPT Realtime → Orchestrator (drain downlink buffer at 10 ms intervals)
  downlinkPacer = setInterval(() => {
    for (let i = 0; i < 10; i++) {
      let out;
      if (downlinkBuf.length >= FRAME_10MS_WIRE) {
        out = downlinkBuf.slice(0, FRAME_10MS_WIRE);
        downlinkBuf = downlinkBuf.slice(FRAME_10MS_WIRE);
      } else if (downlinkBuf.length > 0) {
        out = Buffer.alloc(FRAME_10MS_WIRE);
        downlinkBuf.copy(out, 0, 0, downlinkBuf.length);
        downlinkBuf = Buffer.alloc(0);
      } else {
        out = Buffer.alloc(FRAME_10MS_WIRE);
      }
      const ab = new ArrayBuffer(FRAME_10MS_WIRE);
      new Uint8Array(ab).set(out);
      orchSource.onData({
        samples:       new Int16Array(ab),
        sampleRate:    WIRE_SR,
        bitsPerSample: 16,
        channelCount:  1,
      });
    }
  }, 100);
}

// ------------- Prefill helpers -------------
function coercePrefillTurns() {
  try {
    if (PREFILL_FILE) {
      const arr = JSON.parse(fs.readFileSync(PREFILL_FILE, 'utf8'));
      if (Array.isArray(arr) && arr.length > 0) {
        return arr
          .filter(t => t && typeof t.role === 'string' && typeof t.text === 'string' && t.text.length > 0)
          .map(t => ({ role: t.role, text: t.text }));
      }
    }
  } catch (e) {
    console.warn(`[GPTAdapter-${ROLE}] Failed to parse prefillFile:`, e?.message || e);
  }
  if (PREFILL_TEXT) return [{ role: 'user', text: PREFILL_TEXT }];
  return [];
}

function sendPrefillHistory() {
  const turns = coercePrefillTurns();
  prefillSent = true;
  if (!turns.length) {
    if (AUTOSTART) {
      safeSend({ type: 'response.create', response: { modalities: ['audio', 'text'] } });
    }
    return;
  }
  for (const t of turns) {
    safeSend({
      type: 'conversation.item.create',
      item: { type: 'message', role: t.role, content: [{ type: 'input_text', text: t.text }] },
    });
  }
  if (AUTOSTART) {
    safeSend({ type: 'response.create', response: { modalities: ['audio', 'text'] } });
  }
}

// ------------- Shutdown -------------
function shutdown() {
  console.log(`[GPTAdapter-${ROLE}] Shutting down...`);
  writeCostMetadata();
  try { if (uplinkPacer)   clearInterval(uplinkPacer);   } catch {}
  try { if (downlinkPacer) clearInterval(downlinkPacer); } catch {}
  try { if (orchSink)  orchSink.stop();  } catch {}
  try { if (orchPC)    orchPC.close();   } catch {}
  try { if (gptWS && gptWS.readyState === WebSocket.OPEN) gptWS.close(); } catch {}
  try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch {}
  process.exit(0);
}
