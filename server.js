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

// 🔥 Toggle: wrap reasoning_content in <think>…</think> in the output
const SHOW_REASONING  = process.env.SHOW_REASONING  !== 'false'; // default ON
// 🔥 Default fallback when the client doesn't specify a model
const DEFAULT_MODEL   = process.env.DEFAULT_MODEL   || 'deepseek-ai/deepseek-v4-pro';

// ─────────────────────────────────────────────
//  Model Registry
//  Each entry describes a NIM model and its capabilities.
//
//  Fields:
//    nimId        — exact model string sent to the NIM API
//    thinking     — send chat_template_kwargs.thinking=true  (DeepSeek-style)
//    temperature  — sane default for this model
//    max_tokens   — safe default output limit
//    top_p        — default top_p
//    aliases      — OpenAI / client-side names that map to this model
// ─────────────────────────────────────────────
const MODEL_REGISTRY = {

  // ── DeepSeek ────────────────────────────────
  'deepseek-ai/deepseek-v4-pro': {
    nimId:       'deepseek-ai/deepseek-v4-pro',
    thinking:    true,
    temperature: 0.6,
    max_tokens:  16384,
    top_p:       0.9,
    aliases:     ['deepseek-v4-pro', 'deepseek-v4', 'gpt-4o', 'gpt-4'],
  },

    'deepseek-ai/deepseek-v4-flash': {
    nimId:       'deepseek-ai/deepseek-v4-flash',
    thinking:    true,
    temperature: 0.6,
    max_tokens:  16384,
    top_p:       0.9,
    aliases:     ['deepseek-v4-pro', 'deepseek-v4', 'gpt-4o', 'gpt-4'],
  },

  'minimaxai/minimax-m2.7': {
    nimId:       'minimaxai/minimax-m2.7',
    thinking:    false,
    temperature: 0.6,
    max_tokens:  8182,
    top_p:       0.9,
    aliases:     ['deepseek-v4-pro', 'deepseek-v4', 'gpt-4o', 'gpt-4'],
  },

  // ── Qwen 3 ──────────────────────────────────
  'qwen/qwen3-235b-a22b': {
    nimId:       'qwen/qwen3-235b-a22b',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    aliases:     ['qwen3-235b', 'qwen3.5-397b-a17b', 'qwen-235b'],
  },

  'qwen/qwen3-30b-a3b': {
    nimId:       'qwen/qwen3-30b-a3b',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    aliases:     ['qwen3-30b', 'qwen3-small'],
  },

  'qwen/qwen3-coder-480b-a35b-instruct': {
    nimId:       'qwen/qwen3-coder-480b-a35b-instruct',
    thinking:    false,
    temperature: 0.2,
    max_tokens:  16384,
    top_p:       0.95,
    aliases:     ['qwen3-coder', 'qwen-coder', 'qwen3-coder-480b'],
  },

  'qwen/qwen3.5-397b-a17b': {
    nimId:       'qwen/qwen3.5-397b-a17b',
    thinking:    false,
    temperature: 0.6,
    max_tokens:  16384,
    top_p:       0.95,
    aliases:     ['qwen3.5', 'qwen-coder', 'qwen3-coder-480b'],
  },
  
  // ── Llama ───────────────────────────────────
  'nvidia/llama-3.1-nemotron-ultra-253b-v1': {
    nimId:       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    aliases:     ['nemotron-ultra', 'llama-nemotron', 'gpt-3.5-turbo'],
  },

  'meta/llama-3.3-70b-instruct': {
    nimId:       'meta/llama-3.3-70b-instruct',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    aliases:     ['llama-3.3-70b', 'llama-70b', 'gpt-3.5-turbo-16k'],
  },

  // ── Mistral / Moonshot ───────────────────────
  'moonshotai/kimi-k2-instruct': {
    nimId:       'moonshotai/kimi-k2-instruct',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    aliases:     ['kimi-k2', 'moonshot-kimi'],
  },

  'moonshotai/kimi-k2.6': {
    nimId:       'moonshotai/kimi-k2.6',
    thinking:    false,
    temperature: 1.0,
    max_tokens:  16384,
    top_p:       1.0,
    aliases:     ['kimi-k2.6', 'moonshot-kimi'],
  },

  // ── OpenAI via NIM ───────────────────────────
  'openai/gpt-oss-120b': {
    nimId:       'openai/gpt-oss-120b',
    thinking:    false,
    temperature: 0.7,
    max_tokens:  8192,
    top_p:       0.9,
    aliases:     ['gpt-oss-120b', 'claude-3-opus'],
  },
};

// ─────────────────────────────────────────────
//  Build lookup maps at startup
// ─────────────────────────────────────────────

// nimId → config
const BY_NIM_ID = {};
// any alias / nimId → config   (fast single lookup)
const ALIAS_MAP = {};

for (const config of Object.values(MODEL_REGISTRY)) {
  BY_NIM_ID[config.nimId] = config;
  ALIAS_MAP[config.nimId] = config;
  for (const alias of (config.aliases ?? [])) {
    ALIAS_MAP[alias.toLowerCase()] = config;
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
//  Helpers
// ─────────────────────────────────────────────

/** Resolve an incoming model name → registry config. */
function resolveConfig(requested) {
  if (!requested) return DEFAULT_CONFIG;
  const config = ALIAS_MAP[requested.toLowerCase()] ?? ALIAS_MAP[requested];
  if (!config) {
    console.warn(`[proxy] Unknown model "${requested}" — falling back to ${DEFAULT_CONFIG.nimId}`);
    return DEFAULT_CONFIG;
  }
  return config;
}

/** Build the NIM request body, merging per-model defaults. */
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

/** Merge reasoning + content for non-streaming responses. */
function mergeContent(reasoningContent, content) {
  if (!SHOW_REASONING || !reasoningContent) return content || '';
  return `<think>\n${reasoningContent}\n</think>\n\n${content || ''}`;
}

const nimHeaders = () => ({
  Authorization: `Bearer ${NIM_API_KEY}`,
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
  });
});

app.get('/v1/models', (_req, res) => {
  // Expose every nimId + every alias as a separate model entry
  const seen = new Set();
  const data = [];

  for (const cfg of Object.values(MODEL_REGISTRY)) {
    const ids = [cfg.nimId, ...(cfg.aliases ?? [])];
    for (const id of ids) {
      if (seen.has(id)) continue;
      seen.add(id);
      data.push({
        id,
        object:     'model',
        created:    Math.floor(Date.now() / 1000),
        owned_by:   'nvidia-nim-proxy',
        root:       cfg.nimId,
        thinking:   cfg.thinking,
      });
    }
  }

  res.json({ object: 'list', data });
});

// ─────────────────────────────────────────────
//  Main proxy endpoint
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const { stream = false } = req.body;

  try {
    const { request: nimRequest, cfg } = buildNimRequest(req.body);

    console.log(`[proxy] ${req.body.model ?? '(none)'} → ${nimRequest.model} | stream=${stream} | thinking=${cfg.thinking}`);

    const upstream = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers:      nimHeaders(),
        responseType: stream ? 'stream' : 'json',
        timeout:      stream ? 0 : 600_000,
      }
    );

    // ── Streaming ───────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type',  'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');
      res.flushHeaders();

      let buffer    = '';
      let thinkOpen = false;

      upstream.data.on('data', (raw) => {
        buffer += raw.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();

          if (payload === '[DONE]') { res.write('data: [DONE]\n\n'); continue; }

          let data;
          try { data = JSON.parse(payload); }
          catch { res.write(line + '\n'); continue; }

          const delta = data.choices?.[0]?.delta;
          if (!delta) { res.write(`data: ${JSON.stringify(data)}\n\n`); continue; }

          const reasoning = delta.reasoning_content ?? '';
          const content   = delta.content           ?? '';
          delete delta.reasoning_content;

          if (SHOW_REASONING && cfg.thinking) {
            let out = '';
            if (reasoning) {
              if (!thinkOpen) { out += '<think>\n'; thinkOpen = true; }
              out += reasoning;
            }
            if (content) {
              if (thinkOpen) { out += '\n</think>\n\n'; thinkOpen = false; }
              out += content;
            }
            delta.content = out;
          } else {
            delta.content = content;
          }

          res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
      });

      upstream.data.on('end', () => {
        if (thinkOpen && SHOW_REASONING) {
          const closing = makeStreamChunk(nimRequest.model, '\n</think>\n\n');
          res.write(`data: ${JSON.stringify(closing)}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });

      upstream.data.on('error', (err) => { console.error('[stream error]', err.message); res.end(); });
      res.on('close', () => upstream.data.destroy?.());

    // ── Non-streaming ────────────────────────────────────────────────
    } else {
      const nimData = upstream.data;

      const choices = (nimData.choices ?? []).map((choice, idx) => ({
        index: idx,
        message: {
          role: choice.message.role,
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
    const status  = err.response?.status ?? 500;
    const message = err.response?.data?.detail
      ?? err.response?.data?.error?.message
      ?? err.message
      ?? 'Internal proxy error';

    console.error(`[proxy error] ${status}: ${message}`);

    res.status(status).json({
      error: {
        message,
        type:  status === 429 ? 'rate_limit_error'
             : status >= 500  ? 'api_error'
             :                  'invalid_request_error',
        code:  status,
      },
    });
  }
});

// 404 catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.method} ${req.path} not supported`, type: 'invalid_request_error', code: 404 },
  });
});

// ─────────────────────────────────────────────
//  Utilities
// ─────────────────────────────────────────────
function makeStreamChunk(model, content) {
  return {
    id: `chatcmpl-inject-${Date.now()}`, object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000), model,
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  };
}

// ─────────────────────────────────────────────
//  Boot
// ─────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀  NVIDIA NIM Multi-Model Proxy — port ${PORT}`);
  console.log(`    Default model  : ${DEFAULT_CONFIG.nimId}`);
  console.log(`    Show reasoning : ${SHOW_REASONING ? '✅ ON' : '❌ OFF'}`);
  console.log(`    Registered     : ${Object.keys(MODEL_REGISTRY).length} models`);
  for (const cfg of Object.values(MODEL_REGISTRY)) {
    const think = cfg.thinking ? ' 🧠' : '';
    console.log(`      · ${cfg.nimId}${think}`);
    if (cfg.aliases?.length) console.log(`          aliases: ${cfg.aliases.join(', ')}`);
  }
  console.log(`    Health check   : http://localhost:${PORT}/health\n`);
});
