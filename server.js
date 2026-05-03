// server.js — Multi-model OpenAI-compatible proxy for NVIDIA NIM
// Fixes: retry on 502/503/504, separate connect/response timeouts,
//        per-model timeout config, better error reporting
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

// Retry settings
const MAX_RETRIES       = parseInt(process.env.MAX_RETRIES       || '3',  10);
const RETRY_DELAY_MS    = parseInt(process.env.RETRY_DELAY_MS    || '2000', 10); // base delay
const CONNECT_TIMEOUT   = parseInt(process.env.CONNECT_TIMEOUT   || '10000', 10); // 10s to establish connection
const RESPONSE_TIMEOUT  = parseInt(process.env.RESPONSE_TIMEOUT  || '300000', 10); // 5min for full response

// HTTP status codes that are worth retrying
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

// ─────────────────────────────────────────────
//  Model Registry
// ─────────────────────────────────────────────
const MODEL_REGISTRY = {

  'deepseek-ai/deepseek-v4-pro': {
    nimId:       'deepseek-ai/deepseek-v4-pro',
    thinking:    true,
    temperature: 0.6,
    max_tokens:  8192,
    top_p:       0.9,
    timeout:     300_000,   // large model, give it 5 min
    aliases:     ['deepseek-v4-pro', 'deepseek-v4', 'gpt-4o', 'gpt-4'],
  },

  'qwen/qwen3-235b-a22b': {
    nimId:       'qwen/qwen3-235b-a22b',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    timeout:     240_000,
    aliases:     ['qwen3-235b', 'qwen3.5-397b-a17b', 'qwen-235b', 'qwen3'],
  },

  'qwen/qwen3-30b-a3b': {
    nimId:       'qwen/qwen3-30b-a3b',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    timeout:     120_000,
    aliases:     ['qwen3-30b', 'qwen3-small'],
  },

  'qwen/qwen3-coder-480b-a35b-instruct': {
    nimId:       'qwen/qwen3-coder-480b-a35b-instruct',
    thinking:    false,
    temperature: 0.2,
    max_tokens:  16384,
    top_p:       0.95,
    timeout:     300_000,
    aliases:     ['qwen3-coder', 'qwen-coder', 'qwen3-coder-480b'],
  },

  'nvidia/llama-3.1-nemotron-ultra-253b-v1': {
    nimId:       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    timeout:     240_000,
    aliases:     ['nemotron-ultra', 'llama-nemotron', 'gpt-3.5-turbo'],
  },

  'meta/llama-3.3-70b-instruct': {
    nimId:       'meta/llama-3.3-70b-instruct',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    timeout:     120_000,
    aliases:     ['llama-3.3-70b', 'llama-70b', 'gpt-3.5-turbo-16k'],
  },

  'moonshotai/kimi-k2-instruct': {
    nimId:       'moonshotai/kimi-k2-instruct',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    timeout:     180_000,
    aliases:     ['kimi-k2', 'moonshot-kimi'],
  },

  'openai/gpt-oss-120b': {
    nimId:       'openai/gpt-oss-120b',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    timeout:     180_000,
    aliases:     ['gpt-oss-120b', 'claude-3-opus'],
  },
};

// ─────────────────────────────────────────────
//  Build lookup maps
// ─────────────────────────────────────────────
const ALIAS_MAP = {};
for (const cfg of Object.values(MODEL_REGISTRY)) {
  ALIAS_MAP[cfg.nimId.toLowerCase()] = cfg;
  for (const alias of (cfg.aliases ?? [])) {
    ALIAS_MAP[alias.toLowerCase()] = cfg;
  }
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
//  Retry helper
// ─────────────────────────────────────────────

/**
 * Call fn() up to maxRetries times.
 * Retries on network errors and RETRYABLE_STATUSES.
 * Uses exponential backoff: delay * 2^attempt, capped at 30s.
 */
async function withRetry(fn, { maxRetries = MAX_RETRIES, baseDelay = RETRY_DELAY_MS, label = '' } = {}) {
  let lastError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn(attempt);
    } catch (err) {
      lastError = err;

      const status     = err.response?.status;
      const isTimeout  = err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT' || status === 504;
      const isRetryable = !status || RETRYABLE_STATUSES.has(status);

      if (!isRetryable || attempt === maxRetries) break;

      // 429: respect Retry-After header if present
      let delay = baseDelay * Math.pow(2, attempt);   // 2s → 4s → 8s
      if (status === 429) {
        const retryAfter = parseInt(err.response?.headers?.['retry-after'] || '0', 10);
        if (retryAfter > 0) delay = retryAfter * 1000;
      }
      delay = Math.min(delay, 30_000);

      console.warn(
        `[retry] ${label} attempt ${attempt + 1}/${maxRetries} failed` +
        ` (${status ?? err.code ?? 'network'}${isTimeout ? '/timeout' : ''}).` +
        ` Retrying in ${delay / 1000}s…`
      );

      await sleep(delay);
    }
  }

  throw lastError;
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────
function resolveConfig(requested) {
  if (!requested) return DEFAULT_CONFIG;
  const cfg = ALIAS_MAP[requested.toLowerCase()];
  if (!cfg) {
    console.warn(`[proxy] Unknown model "${requested}" — falling back to ${DEFAULT_CONFIG.nimId}`);
    return DEFAULT_CONFIG;
  }
  return cfg;
}

function buildNimRequest(body) {
  const { model, messages, temperature, max_tokens, top_p, stream } = body;
  const cfg = resolveConfig(model);

  const request = {
    model:       cfg.nimId,
    messages,
    temperature: temperature ?? cfg.temperature,
    max_tokens:  max_tokens  ?? cfg.max_tokens,
    top_p:       top_p       ?? cfg.top_p,
    stream:      stream      ?? false,
  };

  if (cfg.thinking) {
    request.extra_body = { chat_template_kwargs: { thinking: true } };
  }

  return { request, cfg };
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
  Authorization:  `Bearer ${NIM_API_KEY}`,
  'Content-Type': 'application/json',
});

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────
app.get('/health', (_req, res) => {
  res.json({
    status:          'ok',
    service:         'NVIDIA NIM Multi-Model Proxy',
    default_model:   DEFAULT_CONFIG.nimId,
    show_reasoning:  SHOW_REASONING,
    models_loaded:   Object.keys(MODEL_REGISTRY).length,
    retry:           { max: MAX_RETRIES, base_delay_ms: RETRY_DELAY_MS },
    timeouts:        { connect_ms: CONNECT_TIMEOUT, response_ms: RESPONSE_TIMEOUT },
  });
});

app.get('/v1/models', (_req, res) => {
  const seen = new Set();
  const data = [];
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
//  Main proxy
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const { stream = false } = req.body;
  const { request: nimRequest, cfg } = buildNimRequest(req.body);

  console.log(`[proxy] ${req.body.model ?? '(none)'} → ${nimRequest.model} | stream=${stream} | thinking=${cfg.thinking}`);

  try {
    // ── Streaming — no retry (stream already sent headers) ──────────
    if (stream) {
      const upstream = await withRetry(
        () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers:      nimHeaders(),
          responseType: 'stream',
          timeout:      CONNECT_TIMEOUT,   // only connect timeout; data flows freely
        }),
        { label: nimRequest.model }
      );

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

          const delta     = data.choices?.[0]?.delta;
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
          res.write(`data: ${JSON.stringify(makeStreamChunk(nimRequest.model, '\n</think>\n\n'))}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });

      upstream.data.on('error', (err) => { console.error('[stream error]', err.message); res.end(); });
      res.on('close', () => upstream.data.destroy?.());

    // ── Non-streaming — full retry ───────────────────────────────────
    } else {
      const responseTimeout = cfg.timeout ?? RESPONSE_TIMEOUT;

      const upstream = await withRetry(
        (attempt) => {
          if (attempt > 0) console.log(`[proxy] retry attempt ${attempt} for ${nimRequest.model}`);
          return axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
            headers:     nimHeaders(),
            responseType:'json',
            // axios splits timeout into socket (connect) + a single total timeout.
            // We set socketPath timeout separately via httpAgent if needed;
            // for simplicity set a generous single deadline here.
            timeout: responseTimeout,
          });
        },
        { label: nimRequest.model }
      );

      const nimData = upstream.data;
      const choices = (nimData.choices ?? []).map((choice, idx) => ({
        index: idx,
        message: {
          role:    choice.message.role,
          content: mergeContent(
            cfg.thinking ? choice.message.reasoning_content : null,
            choice.message.content
          ),
        },
        finish_reason: choice.finish_reason ?? 'stop',
      }));

      res.json({
        id:      `chatcmpl-${Date.now()}`,
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model:   req.body.model ?? nimRequest.model,
        choices,
        usage:   nimData.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (err) {
    const status  = err.response?.status;
    const isTimeout = err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT' || status === 504;

    const message = isTimeout
      ? `Request timed out after ${MAX_RETRIES + 1} attempts (model: ${nimRequest.model}). Try using stream:true or a smaller model.`
      : (err.response?.data?.detail ?? err.response?.data?.error?.message ?? err.message ?? 'Internal proxy error');

    const httpStatus = status ?? (isTimeout ? 504 : 500);
    console.error(`[proxy error] ${httpStatus}: ${message}`);

    // If streaming headers already sent, we can only close
    if (res.headersSent) { res.end(); return; }

    res.status(httpStatus).json({
      error: {
        message,
        type:     httpStatus === 429 ? 'rate_limit_error'
                : isTimeout          ? 'timeout_error'
                : httpStatus >= 500  ? 'api_error'
                :                     'invalid_request_error',
        code:     httpStatus,
        model:    nimRequest.model,
        retries:  MAX_RETRIES,
        tip:      isTimeout ? 'Set stream:true in your request to avoid gateway timeouts on large models.' : undefined,
      },
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.method} ${req.path} not supported`, type: 'invalid_request_error', code: 404 },
  });
});

// ─────────────────────────────────────────────
//  Boot
// ─────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀  NVIDIA NIM Multi-Model Proxy — port ${PORT}`);
  console.log(`    Default model  : ${DEFAULT_CONFIG.nimId}`);
  console.log(`    Show reasoning : ${SHOW_REASONING ? '✅ ON' : '❌ OFF'}`);
  console.log(`    Retry          : up to ${MAX_RETRIES}x, base delay ${RETRY_DELAY_MS}ms`);
  console.log(`    Timeouts       : connect ${CONNECT_TIMEOUT}ms / response ${RESPONSE_TIMEOUT}ms`);
  console.log(`    Registered     : ${Object.keys(MODEL_REGISTRY).length} models`);
  for (const cfg of Object.values(MODEL_REGISTRY)) {
    console.log(`      · ${cfg.nimId}${cfg.thinking ? ' 🧠' : ''} (timeout: ${cfg.timeout / 1000}s)`);
  }
  console.log(`    Health check   : http://localhost:${PORT}/health\n`);
});
