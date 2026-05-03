// server.js — Multi-model OpenAI-compatible proxy for NVIDIA NIM
// Works on Railway, Render, Fly.io and other platforms.
// Never calls process.exit() — all config errors are reported via HTTP.
'use strict';

const express = require('express');
const cors    = require('cors');
const axios   = require('axios');

// ─────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────
const PORT         = process.env.PORT         || 3000;
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY  || '';

// Warn but do NOT exit — platform must stay alive so /health shows the problem.
if (!NIM_API_KEY) {
  console.warn('[warn] NIM_API_KEY is not set. All proxy requests will fail with 401 until it is added.');
}

const SHOW_REASONING = process.env.SHOW_REASONING !== 'false';
const DEFAULT_MODEL  = process.env.DEFAULT_MODEL  || 'deepseek-ai/deepseek-v4-pro';
const STREAM_TIMEOUT  = parseInt(process.env.STREAM_TIMEOUT  || '600000', 10);
const CONNECT_TIMEOUT = parseInt(process.env.CONNECT_TIMEOUT || '15000',  10);
const MAX_RETRIES     = parseInt(process.env.MAX_RETRIES     || '3',      10);
const RETRY_DELAY_MS  = parseInt(process.env.RETRY_DELAY_MS  || '3000',   10);

const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

// ─────────────────────────────────────────────
//  Model Registry
// ─────────────────────────────────────────────
const MODEL_REGISTRY = {
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
    nimId: 'qwen/qwen3-next-80b-a3b-thinking', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3-thinking', 'qwen3-80b-thinking'],
  },
  'qwen/qwen3-next-80b-a3b-instruct': {
    nimId: 'qwen/qwen3-next-80b-a3b-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3-80b', 'qwen3-instruct'],
  },
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
//  Build alias lookup
// ─────────────────────────────────────────────
const ALIAS_MAP = {};
for (const cfg of Object.values(MODEL_REGISTRY)) {
  ALIAS_MAP[cfg.nimId.toLowerCase()] = cfg;
  for (const a of (cfg.aliases || [])) ALIAS_MAP[a.toLowerCase()] = cfg;
}
const DEFAULT_CONFIG = ALIAS_MAP[DEFAULT_MODEL.toLowerCase()] || Object.values(MODEL_REGISTRY)[0];

// ─────────────────────────────────────────────
//  Express app
// ─────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ─────────────────────────────────────────────
//  Guard middleware — rejects all proxy calls if key is missing
// ─────────────────────────────────────────────
function requireApiKey(req, res, next) {
  if (!NIM_API_KEY) {
    return res.status(401).json({
      error: {
        message: 'NIM_API_KEY environment variable is not configured on this server.',
        type: 'auth_error',
        code: 401,
        fix: 'Add NIM_API_KEY in your platform environment variables (Railway → Variables, Render → Environment, etc.)',
      },
    });
  }
  next();
}

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

function buildNimBody(reqBody) {
  const { model, messages, temperature, max_tokens, top_p } = reqBody;
  const cfg = resolveConfig(model);
  const body = {
    model:       cfg.nimId,
    messages,
    temperature: temperature !== undefined ? temperature : cfg.temperature,
    max_tokens:  max_tokens  !== undefined ? max_tokens  : cfg.max_tokens,
    top_p:       top_p       !== undefined ? top_p       : cfg.top_p,
    stream:      true, // always stream to NIM to avoid 504 cold-start timeouts
  };
  if (cfg.thinking) body.extra_body = { chat_template_kwargs: { thinking: true } };
  return { body, cfg };
}

function mergeContent(reasoningContent, content) {
  if (!SHOW_REASONING || !reasoningContent) return content || '';
  return '<think>\n' + reasoningContent + '\n</think>\n\n' + (content || '');
}

function makeStreamChunk(model, content) {
  return {
    id: 'chatcmpl-inject-' + Date.now(),
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{ index: 0, delta: { content: content }, finish_reason: null }],
  };
}

function nimHeaders() {
  return {
    Authorization:  'Bearer ' + NIM_API_KEY,
    'Content-Type': 'application/json',
  };
}

function describeError(err) {
  return {
    message:     err.message,
    code:        err.code,
    http_status: err.response ? err.response.status : undefined,
    nim_error:   err.response ? err.response.data   : undefined,
  };
}

async function withRetry(fn, opts) {
  const maxRetries = (opts && opts.maxRetries !== undefined) ? opts.maxRetries : MAX_RETRIES;
  const baseDelay  = (opts && opts.baseDelay  !== undefined) ? opts.baseDelay  : RETRY_DELAY_MS;
  const label      = (opts && opts.label)                    ? opts.label      : '';
  let lastError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn(attempt);
    } catch (err) {
      lastError = err;
      const status      = err.response ? err.response.status : undefined;
      const isRetryable = !status || RETRYABLE_STATUSES.has(status);
      if (!isRetryable || attempt === maxRetries) break;
      let delay = Math.min(baseDelay * Math.pow(2, attempt), 30000);
      if (status === 429) {
        const ra = parseInt((err.response.headers && err.response.headers['retry-after']) || '0', 10);
        if (ra > 0) delay = ra * 1000;
      }
      console.warn('[retry] ' + label + ' attempt ' + (attempt + 1) + '/' + maxRetries + ' — ' + (status || err.code) + '. Waiting ' + (delay / 1000) + 's…');
      await sleep(delay);
    }
  }
  throw lastError;
}

// Consume a NIM SSE stream and return the fully assembled response.
function consumeStream(nimStream) {
  return new Promise(function(resolve, reject) {
    let buffer       = '';
    let contentAcc   = '';
    let reasoningAcc = '';
    let lastData     = null;

    nimStream.on('data', function(raw) {
      buffer += raw.toString();
      var lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        if (!line.startsWith('data: ')) continue;
        var payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;
        try {
          var data = JSON.parse(payload);
          lastData = data;
          var delta = data.choices && data.choices[0] && data.choices[0].delta;
          if (delta) {
            if (delta.reasoning_content) reasoningAcc += delta.reasoning_content;
            if (delta.content)           contentAcc   += delta.content;
          }
        } catch (e) { /* skip malformed chunk */ }
      }
    });

    nimStream.on('end', function() {
      resolve({
        id:                (lastData && lastData.id)      || ('chatcmpl-' + Date.now()),
        created:           (lastData && lastData.created) || Math.floor(Date.now() / 1000),
        model:             lastData && lastData.model,
        usage:             (lastData && lastData.usage)   || null,
        role:              'assistant',
        content:           contentAcc,
        reasoning_content: reasoningAcc || null,
        finish_reason:     (lastData && lastData.choices && lastData.choices[0] && lastData.choices[0].finish_reason) || 'stop',
      });
    });

    nimStream.on('error', reject);
  });
}

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────

// Health — always responds even without a key
app.get('/health', function(_req, res) {
  res.json({
    status:           'ok',
    service:          'NVIDIA NIM Multi-Model Proxy',
    node_version:     process.version,
    api_key_set:      !!NIM_API_KEY,
    api_key_prefix:   NIM_API_KEY ? NIM_API_KEY.slice(0, 8) + '…' : null,
    default_model:    DEFAULT_CONFIG.nimId,
    show_reasoning:   SHOW_REASONING,
    models_loaded:    Object.keys(MODEL_REGISTRY).length,
    stream_timeout_s: STREAM_TIMEOUT / 1000,
    connect_timeout_s: CONNECT_TIMEOUT / 1000,
    diagnostic_url:   '/api/test',
  });
});

app.get('/v1/models', function(_req, res) {
  var seen = new Set(), data = [];
  for (var cfg of Object.values(MODEL_REGISTRY)) {
    var ids = [cfg.nimId].concat(cfg.aliases || []);
    for (var id of ids) {
      if (seen.has(id)) continue;
      seen.add(id);
      data.push({ id: id, object: 'model', created: Math.floor(Date.now() / 1000),
                  owned_by: 'nvidia-nim-proxy', root: cfg.nimId, thinking: cfg.thinking });
    }
  }
  res.json({ object: 'list', data: data });
});

// ─────────────────────────────────────────────
//  Diagnostic endpoint
// ─────────────────────────────────────────────
app.get('/api/test', async function(req, res) {
  var modelToTest = req.query.model || DEFAULT_CONFIG.nimId;
  var report = {
    timestamp:    new Date().toISOString(),
    node_version: process.version,
    proxy_config: {
      nim_api_base:    NIM_API_BASE,
      api_key_set:     !!NIM_API_KEY,
      api_key_prefix:  NIM_API_KEY ? NIM_API_KEY.slice(0, 8) + '…' : '(not set)',
      default_model:   DEFAULT_CONFIG.nimId,
    },
    steps: [],
  };

  if (!NIM_API_KEY) {
    report.verdict = '❌ NIM_API_KEY is not set. Add it in your platform environment variables.';
    return res.json(report);
  }

  // Step 1: Connectivity
  try {
    var t0 = Date.now();
    await axios.get(NIM_API_BASE + '/models', { headers: nimHeaders(), timeout: CONNECT_TIMEOUT });
    report.steps.push({ step: '1_connectivity', status: 'ok', ms: Date.now() - t0 });
  } catch (err) {
    report.steps.push({ step: '1_connectivity', status: 'fail', detail: describeError(err) });
    report.verdict = '❌ Cannot reach NIM API. Check NIM_API_BASE and network.';
    return res.json(report);
  }

  // Step 2: Auth + model list
  var availableIds = [];
  try {
    var t1 = Date.now();
    var r = await axios.get(NIM_API_BASE + '/models', { headers: nimHeaders(), timeout: CONNECT_TIMEOUT });
    availableIds = (r.data && r.data.data ? r.data.data : []).map(function(m) { return m.id; });
    report.steps.push({ step: '2_auth_and_models', status: 'ok', ms: Date.now() - t1,
                        available_model_count: availableIds.length });
  } catch (err) {
    var s = err.response && err.response.status;
    report.steps.push({ step: '2_auth_and_models', status: 'fail', detail: describeError(err) });
    report.verdict = (s === 401 || s === 403) ? '❌ Invalid API key.' : '❌ Auth check failed.';
    return res.json(report);
  }

  // Step 3: Model in account list?
  if (availableIds.length > 0 && !availableIds.includes(modelToTest)) {
    report.steps.push({ step: '3_model_availability', status: 'warn',
                        detail: '"' + modelToTest + '" not found in your account model list.' });
  } else {
    report.steps.push({ step: '3_model_availability', status: 'ok', model: modelToTest });
  }

  // Step 4: Inference via stream (bypasses cold-start 504)
  try {
    var t2 = Date.now();
    var cfg = resolveConfig(modelToTest);
    var reqBody = {
      model: cfg.nimId,
      messages: [{ role: 'user', content: 'Reply with the single word: OK' }],
      max_tokens: 10, temperature: 0, stream: true,
    };
    if (cfg.thinking) reqBody.extra_body = { chat_template_kwargs: { thinking: true } };

    var upstream = await axios.post(NIM_API_BASE + '/chat/completions', reqBody, {
      headers: nimHeaders(), responseType: 'stream', timeout: CONNECT_TIMEOUT,
    });

    var assembled = await Promise.race([
      consumeStream(upstream.data),
      new Promise(function(_, rej) { setTimeout(function() { rej(new Error('stream_timeout')); }, 60000); }),
    ]);

    report.steps.push({ step: '4_inference_stream', status: 'ok', ms: Date.now() - t2,
                        response: assembled.content, reasoning_present: !!assembled.reasoning_content });
    report.verdict = '✅ Model "' + modelToTest + '" is working. Response in ~' + (Date.now() - t2) + 'ms.';
  } catch (err) {
    report.steps.push({ step: '4_inference_stream', status: 'fail', detail: describeError(err) });
    report.verdict = err.message === 'stream_timeout'
      ? '⏱ Model "' + modelToTest + '" did not respond within 60s. It may be cold-starting — try again in 30s.'
      : '❌ Inference failed: ' + (err.response && err.response.status || err.code || err.message);
  }

  res.json(report);
});

// ─────────────────────────────────────────────
//  Main proxy endpoint
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', requireApiKey, async function(req, res) {
  var clientWantsStream = !!req.body.stream;
  var built = buildNimBody(req.body);
  var nimBody = built.body;
  var cfg     = built.cfg;

  console.log('[proxy] ' + (req.body.model || '(none)') + ' → ' + nimBody.model +
              ' | client_stream=' + clientWantsStream + ' | thinking=' + cfg.thinking);

  try {
    var upstream = await withRetry(function() {
      return axios.post(NIM_API_BASE + '/chat/completions', nimBody, {
        headers: nimHeaders(), responseType: 'stream', timeout: CONNECT_TIMEOUT,
      });
    }, { label: nimBody.model });

    // ── Client wants SSE stream ──────────────────────────────────────
    if (clientWantsStream) {
      res.setHeader('Content-Type',  'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');
      res.flushHeaders();

      var buffer = '', thinkOpen = false;

      upstream.data.on('data', function(raw) {
        buffer += raw.toString();
        var lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (!line.startsWith('data: ')) continue;
          var payload = line.slice(6).trim();
          if (payload === '[DONE]') { res.write('data: [DONE]\n\n'); continue; }

          var data;
          try { data = JSON.parse(payload); } catch (e) { res.write(line + '\n'); continue; }

          var delta = data.choices && data.choices[0] && data.choices[0].delta;
          if (!delta) { res.write('data: ' + JSON.stringify(data) + '\n\n'); continue; }

          var reasoning = delta.reasoning_content || '';
          var content   = delta.content           || '';
          delete delta.reasoning_content;

          if (SHOW_REASONING && cfg.thinking) {
            var out = '';
            if (reasoning) { if (!thinkOpen) { out += '<think>\n'; thinkOpen = true; } out += reasoning; }
            if (content)   { if (thinkOpen)  { out += '\n</think>\n\n'; thinkOpen = false; } out += content; }
            delta.content = out;
          } else {
            delta.content = content;
          }
          res.write('data: ' + JSON.stringify(data) + '\n\n');
        }
      });

      upstream.data.on('end', function() {
        if (thinkOpen && SHOW_REASONING) {
          res.write('data: ' + JSON.stringify(makeStreamChunk(nimBody.model, '\n</think>\n\n')) + '\n\n');
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });

      upstream.data.on('error', function(err) {
        console.error('[stream error]', describeError(err));
        res.end();
      });

      res.on('close', function() { if (upstream.data.destroy) upstream.data.destroy(); });

    // ── Client wants JSON — assemble from internal stream ────────────
    } else {
      var assembled;
      try {
        assembled = await Promise.race([
          consumeStream(upstream.data),
          new Promise(function(_, rej) {
            setTimeout(function() {
              var e = new Error('stream_timeout');
              e.code = 'STREAM_TIMEOUT';
              rej(e);
            }, STREAM_TIMEOUT);
          }),
        ]);
      } catch (err) {
        if (upstream.data.de
