// server.js — Multi-model OpenAI-compatible proxy for NVIDIA NIM
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

const SHOW_REASONING = process.env.SHOW_REASONING !== 'false';
const DEFAULT_MODEL  = process.env.DEFAULT_MODEL  || 'deepseek-ai/deepseek-v4-pro';
const MAX_RETRIES    = parseInt(process.env.MAX_RETRIES    || '3',      10);
const RETRY_DELAY_MS = parseInt(process.env.RETRY_DELAY_MS || '2000',   10);
const CONNECT_TIMEOUT  = parseInt(process.env.CONNECT_TIMEOUT  || '15000',  10);
const RESPONSE_TIMEOUT = parseInt(process.env.RESPONSE_TIMEOUT || '300000', 10);

const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

// ─────────────────────────────────────────────
//  Model Registry
// ─────────────────────────────────────────────
const MODEL_REGISTRY = {
  'deepseek-ai/deepseek-v4-pro': {
    nimId: 'deepseek-ai/deepseek-v4-pro', thinking: true,
    temperature: 0.6, max_tokens: 8192, top_p: 0.9, timeout: 300_000,
    aliases: ['deepseek-v4-pro', 'deepseek-v4', 'gpt-4o', 'gpt-4'],
  },
  'qwen/qwen3-235b-a22b': {
    nimId: 'qwen/qwen3-235b-a22b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9, timeout: 240_000,
    aliases: ['qwen3-235b', 'qwen3.5-397b-a17b', 'qwen-235b', 'qwen3'],
  },
  'qwen/qwen3-30b-a3b': {
    nimId: 'qwen/qwen3-30b-a3b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9, timeout: 120_000,
    aliases: ['qwen3-30b', 'qwen3-small'],
  },
  'qwen/qwen3-coder-480b-a35b-instruct': {
    nimId: 'qwen/qwen3-coder-480b-a35b-instruct', thinking: false,
    temperature: 0.2, max_tokens: 16384, top_p: 0.95, timeout: 300_000,
    aliases: ['qwen3-coder', 'qwen-coder', 'qwen3-coder-480b'],
  },
  'nvidia/llama-3.1-nemotron-ultra-253b-v1': {
    nimId: 'nvidia/llama-3.1-nemotron-ultra-253b-v1', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9, timeout: 240_000,
    aliases: ['nemotron-ultra', 'llama-nemotron', 'gpt-3.5-turbo'],
  },
  'meta/llama-3.3-70b-instruct': {
    nimId: 'meta/llama-3.3-70b-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9, timeout: 120_000,
    aliases: ['llama-3.3-70b', 'llama-70b', 'gpt-3.5-turbo-16k'],
  },
  'moonshotai/kimi-k2-instruct': {
    nimId: 'moonshotai/kimi-k2-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9, timeout: 180_000,
    aliases: ['kimi-k2', 'moonshot-kimi'],
  },
  'openai/gpt-oss-120b': {
    nimId: 'openai/gpt-oss-120b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9, timeout: 180_000,
    aliases: ['gpt-oss-120b', 'claude-3-opus'],
  },
};

// Build alias lookup
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
  if (cfg.thinking) request.extra_body = { chat_template_kwargs: { thinking: true } };
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
  Authorization: `Bearer ${NIM_API_KEY}`,
  'Content-Type': 'application/json',
});

/**
 * Stringify any axios error into a readable object.
 * This is the key diagnostic helper — it captures everything NIM returned.
 */
function describeError(err) {
  return {
    message:       err.message,
    code:          err.code,                          // ECONNABORTED, ETIMEDOUT, ENOTFOUND …
    http_status:   err.response?.status,
    nim_error:     err.response?.data,                // full NIM error body
    request_url:   err.config?.url,
    request_model: (() => {
      try { return JSON.parse(err.config?.data)?.model; } catch { return undefined; }
    })(),
  };
}

async function withRetry(fn, { maxRetries = MAX_RETRIES, baseDelay = RETRY_DELAY_MS, label = '' } = {}) {
  let lastError;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn(attempt);
    } catch (err) {
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
//  Routes
// ─────────────────────────────────────────────
app.get('/health', (_req, res) => {
  res.json({
    status: 'ok', service: 'NVIDIA NIM Multi-Model Proxy',
    default_model: DEFAULT_CONFIG.nimId,
    show_reasoning: SHOW_REASONING,
    models_loaded: Object.keys(MODEL_REGISTRY).length,
    retry: { max: MAX_RETRIES, base_delay_ms: RETRY_DELAY_MS },
    timeouts: { connect_ms: CONNECT_TIMEOUT, response_ms: RESPONSE_TIMEOUT },
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
//  🔍 Diagnostic endpoint — hit /api/test in browser
// ─────────────────────────────────────────────
app.get('/api/test', async (req, res) => {
  const modelToTest = req.query.model || DEFAULT_CONFIG.nimId;
  const report = {
    timestamp:       new Date().toISOString(),
    proxy_config: {
      nim_api_base:    NIM_API_BASE,
      api_key_present: !!NIM_API_KEY,
      api_key_prefix:  NIM_API_KEY ? NIM_API_KEY.slice(0, 8) + '…' : null,
      default_model:   DEFAULT_CONFIG.nimId,
    },
    steps: [],
  };

  // Step 1: DNS / connectivity — can we reach NIM at all?
  try {
    const t0 = Date.now();
    await axios.get(`${NIM_API_BASE}/models`, {
      headers: nimHeaders(),
      timeout: CONNECT_TIMEOUT,
    });
    report.steps.push({ step: '1_connectivity', status: 'ok', ms: Date.now() - t0 });
  } catch (err) {
    report.steps.push({ step: '1_connectivity', status: 'fail', detail: describeError(err) });
    report.verdict = '❌ Cannot reach NIM API. Check network / NIM_API_BASE.';
    return res.status(200).json(report);
  }

  // Step 2: Is the API key valid?
  try {
    const t0 = Date.now();
    const r = await axios.get(`${NIM_API_BASE}/models`, {
      headers: nimHeaders(),
      timeout: CONNECT_TIMEOUT,
    });
    const availableIds = (r.data?.data ?? []).map(m => m.id);
    report.steps.push({ step: '2_auth_and_models', status: 'ok', ms: Date.now() - t0,
                        available_model_count: availableIds.length,
                        available_models: availableIds });

    // Step 3: Is the requested model in the list?
    if (availableIds.length > 0 && !availableIds.includes(modelToTest)) {
      report.steps.push({
        step: '3_model_availability', status: 'warn',
        detail: `"${modelToTest}" not found in your account's model list.`,
        suggestion: 'Use one of the available_models listed in step 2, or check your NIM subscription.',
      });
      report.verdict = `⚠️  Model "${modelToTest}" may not be available on your account.`;
    } else {
      report.steps.push({ step: '3_model_availability', status: 'ok', model: modelToTest });
    }
  } catch (err) {
    const status = err.response?.status;
    if (status === 401 || status === 403) {
      report.steps.push({ step: '2_auth_and_models', status: 'fail',
                          detail: 'API key rejected (401/403). Check NIM_API_KEY.' });
      report.verdict = '❌ Invalid API key.';
    } else {
      report.steps.push({ step: '2_auth_and_models', status: 'fail', detail: describeError(err) });
      report.verdict = '❌ Auth check failed with unexpected error.';
    }
    return res.status(200).json(report);
  }

  // Step 4: Tiny real inference call (max_tokens=5, no retry)
  try {
    const t0 = Date.now();
    const cfg = resolveConfig(modelToTest);
    const body = {
      model:       cfg.nimId,
      messages:    [{ role: 'user', content: 'Reply with the single word: OK' }],
      max_tokens:  5,
      temperature: 0,
      stream:      false,
    };
    if (cfg.thinking) body.extra_body = { chat_template_kwargs: { thinking: true } };

    const r = await axios.post(`${NIM_API_BASE}/chat/completions`, body, {
      headers: nimHeaders(),
      timeout: 30_000,   // short — we just want to see if the model responds
    });

    report.steps.push({
      step:     '4_inference',
      status:   'ok',
      ms:       Date.now() - t0,
      response: r.data?.choices?.[0]?.message?.content,
    });
    report.verdict = `✅ Everything OK. Model "${modelToTest}" is reachable and responding.`;

  } catch (err) {
    const status = err.response?.status;
    const detail = describeError(err);
    report.steps.push({ step: '4_inference', status: 'fail', detail });

    if (status === 504 || err.code === 'ECONNABORTED') {
      report.verdict = `⏱  Model "${modelToTest}" timed out on inference (30s limit for test). ` +
                       `The model may be cold-starting. Try again in 30–60s, or use stream:true in real requests.`;
    } else if (status === 404) {
      report.verdict = `❌ Model "${modelToTest}" returned 404. It may not be deployed on your account.`;
    } else if (status === 422) {
      report.verdict = `❌ Request rejected (422 Unprocessable). The model may not support the extra_body thinking parameter. ` +
                       `Try setting ENABLE_THINKING=false for this model.`;
    } else {
      report.verdict = `❌ Inference failed: ${status ?? err.code ?? err.message}`;
    }
  }

  res.status(200).json(report);
});

// ─────────────────────────────────────────────
//  Main proxy endpoint
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const { stream = false } = req.body;
  const { request: nimRequest, cfg } = buildNimRequest(req.body);

  console.log(`[proxy] ${req.body.model ?? '(none)'} → ${nimRequest.model} | stream=${stream} | thinking=${cfg.thinking}`);

  try {
    if (stream) {
      // ── Streaming ─────────────────────────────────────────────────
      const upstream = await withRetry(
        () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: nimHeaders(), responseType: 'stream', timeout: CONNECT_TIMEOUT,
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
          res.write(`data: ${JSON.stringify(makeStreamChunk(nimRequest.model, '\n</think>\n\n'))}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });

      upstream.data.on('error', (err) => {
        console.error('[stream error]', describeError(err));
        res.end();
      });
      res.on('close', () => upstream.data.destroy?.());

    } else {
      // ── Non-streaming ──────────────────────────────────────────────
      const upstream = await withRetry(
        (attempt) => {
          if (attempt > 0) console.log(`[proxy] retry #${attempt} for ${nimRequest.model}`);
          return axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
            headers: nimHeaders(), responseType: 'json',
            timeout: cfg.timeout ?? RESPONSE_TIMEOUT,
          });
        },
        { label: nimRequest.model }
      );

      const nimData = upstream.data;
      const choices = (nimData.choices ?? []).map((choice, idx) => ({
        index: idx,
        message: {
          role:    choice.message.role,
          content: mergeContent(cfg.thinking ? choice.message.reasoning_content : null, choice.message.content),
        },
        finish_reason: choice.finish_reason ?? 'stop',
      }));

      res.json({
        id: `chatcmpl-${Date.now()}`, object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model:   req.body.model ?? nimRequest.model,
        choices,
        usage: nimData.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (err) {
    const detail    = describeError(err);
    const status    = detail.http_status;
    const isTimeout = err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT' || status === 504;

    // Always log the full detail so you can see it in server logs
    console.error('[proxy error]', JSON.stringify(detail, null, 2));

    const message = isTimeout
      ? `Timed out after ${MAX_RETRIES + 1} attempts. Model: ${nimRequest.model}. Try stream:true.`
      : (detail.nim_error?.detail ?? detail.nim_error?.error?.message ?? err.message ?? 'Internal proxy error');

    if (res.headersSent) { res.end(); return; }

    res.status(status ?? (isTimeout ? 504 : 500)).json({
      error: {
        message,
        type:     status === 429 ? 'rate_limit_error'
                : isTimeout      ? 'timeout_error'
                : status >= 500  ? 'api_error'
                :                  'invalid_request_error',
        code:     status ?? 500,
        model:    nimRequest.model,
        // Include full NIM error so client can see exactly what NIM said
        nim_detail: detail.nim_error,
        tip:      isTimeout
                  ? 'Use stream:true, or visit /api/test to diagnose.'
                  : 'Visit /api/test to run a full diagnostic.',
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
  console.log(`    Diagnostic     : http://localhost:${PORT}/api/test`);
  console.log(`    Health         : http://localhost:${PORT}/health`);
  console.log(`    Registered models:`);
  for (const cfg of Object.values(MODEL_REGISTRY)) {
    console.log(`      · ${cfg.nimId}${cfg.thinking ? ' 🧠' : ''}`);
  }
  console.log();
});
