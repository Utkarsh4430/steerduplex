#!/usr/bin/env node
/**
 * token_server.js
 * Mints short-lived Realtime session tokens by forwarding to a LiteLLM proxy.
 * Drop-in replacement for the original openai_token_server.js, with the upstream
 * endpoint and key sourced from LiteLLM config instead of OpenAI directly.
 *
 * Endpoints
 *   POST /session  -> { client_secret: { value, expires_at }, id, ... }
 *   GET  /healthz  -> 200 OK
 *
 * Env
 *   LITELLM_BASE_URL=http://localhost:4000   (LiteLLM proxy)
 *   LITELLM_API_KEY=sk-...
 *   OPENAI_REALTIME_MODEL=openai/gpt-4o-realtime-preview-2024-12-17  (optional)
 *   TOKEN_SERVER_PORT=3002                   (optional)
 */

require('dotenv').config();
const express = require('express');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

const argv = yargs(hideBin(process.argv))
  .scriptName('token_server')
  .usage('$0 [options]')
  .option('port',  { alias: 'p', type: 'number', default: parseInt(process.env.TOKEN_SERVER_PORT, 10) || 3002 })
  .option('model', { alias: 'm', type: 'string',  default: process.env.OPENAI_REALTIME_MODEL || 'openai/gpt-4o-realtime-preview-2024-12-17' })
  .option('litellmBaseUrl', { type: 'string', default: process.env.LITELLM_BASE_URL || 'http://localhost:4000' })
  .option('litellmApiKey',  { type: 'string', default: process.env.LITELLM_API_KEY  || '' })
  .help().argv;

const PORT           = argv.port;
const DEFAULT_MODEL  = argv.model;
const LITELLM_BASE   = argv.litellmBaseUrl.replace(/\/$/, '');
const LITELLM_KEY    = argv.litellmApiKey;

if (!LITELLM_KEY) {
  console.error('[token-server] Missing LITELLM_API_KEY. Set it via env or --litellmApiKey.');
  process.exit(1);
}

const app = express();
app.use(express.json());

app.get('/healthz', (_req, res) => res.status(200).send('ok'));

/**
 * Forward session-creation request to LiteLLM, which proxies to OpenAI Realtime.
 * The returned client_secret.value is the ephemeral Bearer token the adapter uses
 * for the SDP exchange.
 */
app.post('/session', async (req, res) => {
  try {
    const body  = req.body || {};
    const model = body.model || DEFAULT_MODEL;

    const payload = { model };
    if (body.voice)                    payload.voice = body.voice;
    if (body.instructions)             payload.instructions = body.instructions;
    if (body.turn_detection)           payload.turn_detection = body.turn_detection;
    if (body.input_audio_transcription) payload.input_audio_transcription = body.input_audio_transcription;

    const upstream = `${LITELLM_BASE}/realtime/sessions`;
    const r = await fetch(upstream, {
      method:  'POST',
      headers: {
        'Authorization': `Bearer ${LITELLM_KEY}`,
        'Content-Type':  'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!r.ok) {
      const errText = await r.text().catch(() => '');
      console.error('[token-server] LiteLLM /realtime/sessions error:', r.status, errText);
      return res.status(500).json({ error: 'failed_to_create_session', status: r.status, detail: errText });
    }

    const json = await r.json();
    res.json(json);
  } catch (e) {
    console.error('[token-server] /session exception:', e);
    res.status(500).json({ error: 'exception', detail: String(e) });
  }
});

app.listen(PORT, () => {
  console.log(`[token-server] Listening on http://localhost:${PORT} → ${LITELLM_BASE}`);
});
