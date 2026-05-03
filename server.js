// server.js — Multi-model OpenAI-compatible proxy for NVIDIA NIM
// Key fix: non-streaming requests are fulfilled via an internal stream
//          to avoid NIM gateway 504 on cold-starting large models.
'use strict';

const express = require('express');
const cors    = require('cors');
const axios   = require('axios');

// ─────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────
const PORT         = process.env.PORT         || 3000;
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('[FATAL] NIM_API_KEY is not set. Exiting.');
  process.exit(1);
}

const SHOW_REASONING = process.env.SHOW_REASONING !== 'false'; // default ON
const DEFAULT_MODEL  = process.env.DEFAULT_MODEL  || 'deepseek-ai/deepseek-v4-pro';

// For non-stream requests we open an internal stream.
// STREAM_TIMEOUT is the max time we'll wait for the *entire* streamed response.
const STREAM_TIMEOUT   = parseInt(process.env.STREAM_TIMEOUT   || '600000', 10); // 10 min
const CONNECT_TIMEOUT  = parseInt(process.env.CONNECT_TIMEOUT  || '15000',  10); // 15 s

const MAX_RETRIES    = parseInt(process.env.MAX_RETRIES    || '3',    10);
const RETRY_DELAY_MS = parseInt(process.env.RETRY_DELAY_MS || '3000', 10);

const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

// ─────────────────────────────────────────────
//  Model Registry  (trimmed to your account)
// ─────────────────────────────────────────────
const MODEL_REGISTRY = {

  // ── DeepSeek ────────────────────────────────
  'deepseek-ai/deepseek-v4-pro': {
    nimId: 'deepseek-ai/deepseek-v4-pro', thinking: true,
    temperature: 0.6, max_tokens: 8192, top_p: 0.9,
    aliases: ['deepseek-v4-pro', 'deepseek-v4', 'gpt-4o', 'gpt-4'],
  },
  'deepseek-ai/deepseek-v4-flash': {
    nimId: 'deepseek-ai/deepseek-v4-flash', thinking: true,
    temperature: 0.6, max_tokens: 8192, top_p: 0.9,
    aliases: ['deepseek-v4-flash', 'deepseek-flash'],
  },
  'deepseek-ai/deepseek-v3.2': {
    nimId: 'deepseek-ai/deepseek-v3.2', thinking: false,
    temperature: 0.6, max_tokens: 8192, top_p: 0.9,
    aliases: ['deepseek-v3', 'deepseek-v3.2'],
  },

  // ── Qwen ────────────────────────────────────
  'qwen/qwen3.5-397b-a17b': {
    nimId: 'qwen/qwen3.5-397b-a17b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3.5-397b', 'qwen3.5', 'qwen-large'],
  },
  'qwen/qwen3.5-122b-a10b': {
    nimId: 'qwen/qwen3.5-122b-a10b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3.5-122b', 'qwen-mid'],
  },
  'qwen/qwen3-coder-480b-a35b-instruct': {
    nimId: 'qwen/qwen3-coder-480b-a35b-instruct', thinking: false,
    temperature: 0.2, max_tokens: 16384, top_p: 0.95,
    aliases: ['qwen3-coder', 'qwen-coder', 'qwen3-coder-480b'],
  },
  'qwen/qwen3-next-80b-a3b-thinking': {
    nimId: 'qwen/qwen3-next-80b-a3b-thinking', thinking: false, // uses /think token natively
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3-thinking', 'qwen3-80b-thinking'],
  },
  'qwen/qwen3-next-80b-a3b-instruct': {
    nimId: 'qwen/qwen3-next-80b-a3b-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3-80b', 'qwen3-instruct'],
  },

  // ── Llama ────────────────────────────────────
  'meta/llama-3.3-70b-instruct': {
    nimId: 'meta/llama-3.3-70b-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['llama-3.3-70b', 'llama-70b', 'gpt-3.5-turbo'],
  },
  'meta/llama-3.1-405b-instruct': {
    nimId: 'meta/llama-3.1-405b-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['llama-405b', 'llama-3.1-405b'],
  },
  'meta/llama-4-maverick-17b-128e-instruct': {
    nimId: 'meta/llama-4-maverick-17b-128e-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['llama-4-maverick', 'llama4', 'gpt-3.5-turbo-16k'],
  },

  // ── NVIDIA Nemotron ──────────────────────────
  'nvidia/llama-3.1-nemotron-ultra-253b-v1': {
    nimId: 'nvidia/llama-3.1-nemotron-ultra-253b-v1', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['nemotron-ultra', 'llama-nemotron-ultra'],
  },
  'nvidia/llama-3.3-nemotron-super-49b-v1': {
    nimId: 'nvidia/llama-3.3-nemotron-super-49b-v1', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['nemotron-super', 'nemotron-49b'],
  },

  // ── Mistral ──────────────────────────────────
  'mistralai/mistral-large-3-675b-instruct-2512': {
    nimId: 'mistralai/mistral-large-3-675b-instruct-2512', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['mistral-large-3', 'mistral-large', 'claude-3-opus'],
  },
  'mistralai/mistral-medium-3.5-128b': {
    nimId: 'mistralai/mistral-medium-3.5-128b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['mistral-medium', 'mistral-medium-3.5'],
  },

  // ── Moonshot Kimi ────────────────────────────
  'moonshotai/kimi-k2-instruct': {
    nimId: 'moonshotai/kimi-k2-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['kimi-k2', 'moonshot-kimi'],
  },
  'moonshotai/kimi-k2-thinking': {
    nimId: 'moonshotai/kimi-k2-thinking', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['kimi-k2-thinking', 'kimi-thinking'],
  },

  // ── OpenAI via NIM ───────────────────────────
  'openai/gpt-oss-120b': {
    nimId: 'openai/gpt-oss-120b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['gpt-oss-120b'],
  },
  'openai/gpt-oss-20b': {
    nimId: 'openai/gpt-oss-20b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['gpt-oss-20b', 'claude-3-sonnet', 'claude-3-haiku'],
  },
};

// ─────────────────────────────────────────────
//  Build lookup maps
// ─────────────────────────────────────────────
const ALIAS_MAP = {};
for (const cfg of Object.values(MODEL_REGISTRY)) {
  ALIAS_MAP[cfg.nimId.toLowerCase()] = cfg;
  for (const a of (cfg.aliases ?? [])) ALIAS_MAP[a.toLowerCase()] = cfg;
}
const DEFAULT_CONFIG = ALIAS_MAP[DEFAULT_MODEL.toLowerCase()] ?? Object.values(MODEL_REGISTRY)[0];

// ─────────────────────────────────────────────
//  App
// ─────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ─────────────────────────────────────────────
//  Utilities
// ─────────────────────────────────────────────
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function resolveConfig(requested) {
  if (!requested) return DEFAULT_CONFIG;
  const cfg = ALIAS_MAP[requested.toLowerCase()];
  if (!cfg) {
    console.warn(`[proxy] Unknown model "${requested}" — falling back to ${DEFAULT_CONFIG.nimId}`);
    return DEFAULT_CONFIG;
  }
  return cfg;
}

function buildNimBody({ model, messages, temperature, max_tokens, top_p }, forceStream) {
  const cfg = resolveConfig(model);
  const body = {
    model:       cfg.nimId,
    messages,
    temperature: temperature ?? cfg.temperature,
    max_tokens:  max_tokens  ?? cfg.max_tokens,
    top_p:       top_p       ?? cfg.top_p,
    stream:      forceStream ?? false,
  };
  if (cfg.thinking) body.extra_body = { chat_template_kwargs: { thinking: true } };
  return { body, cfg };
}

function mergeContent(reasoningContent, content) {
  if (!SHOW_REASONING || !reasoningContent) return content || '';
  return `<think>\n${reasoningContent}\n</think>\n\n${content || ''}`;
}

function makeStreamChunk(model, content) {
  return {
    id: `chatcmpl-inject-${Date.now()}`, object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000), model,
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  };
}

const nimHeaders = () => ({
  Authorization: `Bearer ${NIM_API_KEY}`,
  'Content-Type': 'application/json',
});

function describeError(err) {
  return {
    message:       err.message,
    code:          err.code,
    http_status:   err.response?.status,
    nim_error:     err.response?.data,
    request_model: (() => { try { return JSON.parse(err.config?.data)?.model; } catch { return undefined; } })(),
  };
}

async function withRetry(fn, { maxRetries = MAX_RETRIES, baseDelay = RETRY_DELAY_MS, label = '' } = {}) {
  let lastError;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try { return await fn(attempt); } catch (err) {
      lastError = err;
      const status      = err.response?.status;
      const isRetryable = !status || RETRYABLE_STATUSES.has(status);
      if (!isRetryable || attempt === maxRetries) break;
      let delay = Math.min(baseDelay * Math.pow(2, attempt), 30_000);
      if (status === 429) {
        const ra = parseInt(err.response?.headers?.['retry-after'] || '0', 10);
        if (ra > 0) delay = ra * 1000;
      }
      console.warn(`[retry] ${label} attempt ${attempt + 1}/${maxRetries} — ${status ?? err.code}. Waiting ${delay / 1000}s…`);
      await sleep(delay);
    }
  }
  throw lastError;
}

// ─────────────────────────────────────────────
//  Core: consume a NIM stream → assembled response object
//  Used for both "fake non-stream" and diagnostics.
// ─────────────────────────────────────────────
function consumeStream(nimStream, cfg) {
  return new Promise((resolve, reject) => {
    let buffer      = '';
    let thinkOpen   = false;
    let contentAcc  = '';
    let reasoningAcc = '';
    let lastData    = null;

    nimStream.on('data', (raw) => {
      buffer += raw.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;

        let data;
        try { data = JSON.parse(payload); } catch { continue; }
        lastData = data;

        const delta = data.choices?.[0]?.delta ?? {};
        if (delta.reasoning_content) reasoningAcc += delta.reasoning_content;
        if (delta.content)           contentAcc   += delta.content;
      }
    });

    nimStream.on('end', () => {
      resolve({
        id:      lastData?.id      ?? `chatcmpl-${Date.now()}`,
        created: lastData?.created ?? Math.floor(Date.now() / 1000),
        model:   lastData?.model,
        usage:   lastData?.usage   ?? null,
        role:    'assistant',
        content: contentAcc,
        reasoning_content: reasoningAcc || null,
        finish_reason: lastData?.choices?.[0]?.finish_reason ?? 'stop',
      });
    });

    nimStream.on('error', reject);
  });
}

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────
app.get('/health', (_req, res) => {
  res.json({
    status: 'ok', service: 'NVIDIA NIM Multi-Model Proxy',
    default_model: DEFAULT_CONFIG.nimId, show_reasoning: SHOW_REASONING,
    models_loaded: Object.keys(MODEL_REGISTRY).length,
    stream_timeout_ms: STREAM_TIMEOUT, connect_timeout_ms: CONNECT_TIMEOUT,
  });
});

app.get('/v1/models', (_req, res) => {
  const seen = new Set(), data = [];
  for (const cfg of Object.values(MODEL_REGISTRY)) {
    for (const id of [cfg.nimId, ...(cfg.aliases ?? [])]) {
      if (seen.has(id)) continue;
      seen.add(id);
      data.push({ id, object: 'model', created: Math.floor(Date.now() / 1000),
                  owned_by: 'nvidia-nim-proxy', root: cfg.nimId, thinking: cfg.thinking });
    }
  }
  res.json({ object: 'list', data });
});

// ─────────────────────────────────────────────
//  🔍 Diagnostic endpoint
// ─────────────────────────────────────────────
app.get('/api/test', async (req, res) => {
  const modelToTest = req.query.model || DEFAULT_CONFIG.nimId;
  const report = {
    timestamp: new Date().toISOString(),
    proxy_config: {
      nim_api_base:    NIM_API_BASE,
      api_key_present: !!NIM_API_KEY,
      api_key_prefix:  NIM_API_KEY ? NIM_API_KEY.slice(0, 8) + '…' : null,
      default_model:   DEFAULT_CONFIG.nimId,
    },
    steps: [],
  };

  // Step 1: Connectivity
  try {
    const t0 = Date.now();
    await axios.get(`${NIM_API_BASE}/models`, { headers: nimHeaders(), timeout: CONNECT_TIMEOUT });
    report.steps.push({ step: '1_connectivity', status: 'ok', ms: Date.now() - t0 });
  } catch (err) {
    report.steps.push({ step: '1_connectivity', status: 'fail', detail: describeError(err) });
    report.verdict = '❌ Cannot reach NIM API.';
    return res.json(report);
  }

  // Step 2: Auth + model list
  let availableIds = [];
  try {
    const t0 = Date.now();
    const r = await axios.get(`${NIM_API_BASE}/models`, { headers: nimHeaders(), timeout: CONNECT_TIMEOUT });
    availableIds = (r.data?.data ?? []).map(m => m.id);
    report.steps.push({ step: '2_auth_and_models', status: 'ok', ms: Date.now() - t0,
                        available_model_count: availableIds.length });
  } catch (err) {
    const status = err.response?.status;
    report.steps.push({ step: '2_auth_and_models', status: 'fail', detail: describeError(err) });
    report.verdict = (status === 401 || status === 403) ? '❌ Invalid API key.' : '❌ Auth check failed.';
    return res.json(report);
  }

  // Step 3: Model availability
  if (availableIds.length > 0 && !availableIds.includes(modelToTest)) {
    report.steps.push({ step: '3_model_availability', status: 'warn',
                        detail: `"${modelToTest}" not in your model list.` });
  } else {
    report.steps.push({ step: '3_model_availability', status: 'ok', model: modelToTest });
  }

  // Step 4: Inference via stream (avoids cold-start 504)
  try {
    const t0  = Date.now();
    const cfg = resolveConfig(modelToTest);
    const reqBody = {
      model: cfg.nimId,
      messages: [{ role: 'user', content: 'Reply with the single word: OK' }],
      max_tokens: 10, temperature: 0, stream: true,
    };
    if (cfg.thinking) reqBody.extra_body = { chat_template_kwargs: { thinking: true } };

    const upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, reqBody, {
      headers: nimHeaders(), responseType: 'stream', timeout: CONNECT_TIMEOUT,
    });

    const assembled = await Promise.race([
      consumeStream(upstream.data, cfg),
      new Promise((_, rej) => setTimeout(() => rej(new Error('stream_timeout')), 60_000)),
    ]);

    report.steps.push({
      step: '4_inference_stream', status: 'ok', ms: Date.now() - t0,
      response: assembled.content,
      reasoning_present: !!assembled.reasoning_content,
    });
    report.verdict = `✅ Model "${modelToTest}" is working. First token in ~${Date.now() - t0}ms.`;
  } catch (err) {
    const detail = describeError(err);
    report.steps.push({ step: '4_inference_stream', status: 'fail', detail });
    report.verdict = err.message === 'stream_timeout'
      ? `⏱ Model "${modelToTest}" did not respond within 60s even via stream. It may be unavailable right now.`
      : `❌ Inference failed: ${detail.http_status ?? err.code ?? err.message}`;
  }

  res.json(report);
});

// ─────────────────────────────────────────────
//  Main proxy endpoint
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const clientWantsStream = !!req.body.stream;
  const { body: nimBody, cfg } = buildNimBody(req.body, true); // always stream to NIM

  console.log(`[proxy] ${req.body.model ?? '(none)'} → ${nimBody.model} | client_stream=${clientWantsStream} | thinking=${cfg.thinking}`);

  // Helper: open a streaming connection to NIM (with retry)
  const openUpstream = () => withRetry(
    () => axios.post(`${NIM_API_BASE}/chat/completions`, nimBody, {
      headers: nimHeaders(), responseType: 'stream', timeout: CONNECT_TIMEOUT,
    }),
    { label: nimBody.model }
  );

  try {
    const upstream = await openUpstream();

    // ── Client wants a real stream ───────────────────────────────────
    if (clientWantsStream) {
      res.setHeader('Content-Type',  'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');
      res.flushHeaders();

      let buffer = '', thinkOpen = false;

      upstream.data.on('data', (raw) => {
        buffer += raw.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') { res.write('data: [DONE]\n\n'); continue; }

          let data;
          try { data = JSON.parse(payload); } catch { res.write(line + '\n'); continue; }

          const delta = data.choices?.[0]?.delta;
          if (!delta) { res.write(`data: ${JSON.stringify(data)}\n\n`); continue; }

          const reasoning = delta.reasoning_content ?? '';
          const content   = delta.content           ?? '';
          delete delta.reasoning_content;

          if (SHOW_REASONING && cfg.thinking) {
            let out = '';
            if (reasoning) { if (!thinkOpen) { out += '<think>\n'; thinkOpen = true; } out += reasoning; }
            if (content)   { if (thinkOpen)  { out += '\n</think>\n\n'; thinkOpen = false; } out += content; }
            delta.content = out;
          } else {
            delta.content = content;
          }
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
      });

      upstream.data.on('end', () => {
        if (thinkOpen && SHOW_REASONING) {
          res.write(`data: ${JSON.stringify(makeStreamChunk(nimBody.model, '\n</think>\n\n'))}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });
      upstream.data.on('error', (err) => { console.error('[stream error]', describeError(err)); res.end(); });
      res.on('close', () => upstream.data.destroy?.());

    // ── Client wants normal JSON — assemble from internal stream ─────
    } else {
      let assembled;
      try {
        assembled = await Promise.race([
          consumeStream(upstream.data, cfg),
          new Promise((_, rej) =>
            setTimeout(() => rej(Object.assign(new Error('stream_timeout'), { code: 'STREAM_TIMEOUT' })),
            STREAM_TIMEOUT)
          ),
        ]);
      } catch (err) {
        upstream.data.destroy?.();
        throw err;
      }

      const fullContent = mergeContent(
        cfg.thinking ? assembled.reasoning_content : null,
        assembled.content
      );

      res.json({
        id:      assembled.id,
        object:  'chat.completion',
        created: assembled.created,
        model:   req.body.model ?? assembled.model ?? nimBody.model,
        choices: [{
          index:         0,
          message:       { role: assembled.role, content: fullContent },
          finish_reason: assembled.finish_reason,
        }],
        usage: assembled.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (err) {
    const detail    = describeError(err);
    const status    = detail.http_status;
    const isTimeout = err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT'
                   || err.code === 'STREAM_TIMEOUT' || status === 504;

    console.error('[proxy error]', JSON.stringify(detail, null, 2));

    const message = isTimeout
      ? `Model ${nimBody.model} did not respond within the timeout. It may be overloaded — retry in a moment.`
      : (detail.nim_error?.detail ?? detail.nim_error?.error?.message ?? err.message ?? 'Proxy error');

    if (res.headersSent) { res.end(); return; }

    res.status(status ?? (isTimeout ? 504 : 500)).json({
      error: {
        message,
        type:       status === 429 ? 'rate_limit_error'
                  : isTim
