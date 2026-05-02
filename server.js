// server.js — OpenAI-compatible multi-model proxy for NVIDIA NIM
'use strict';

const express = require('express');
const cors    = require('cors');
const axios   = require('axios');

// ─────────────────────────────────────────────
//  Core config
// ─────────────────────────────────────────────
const PORT         = process.env.PORT         || 3000;
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('[FATAL] NIM_API_KEY is not set. Exiting.');
  process.exit(1);
}

// Default model when the client sends an unknown name
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'deepseek-ai/deepseek-v4-pro';

// ─────────────────────────────────────────────
//  Model registry
//
//  Each entry describes one NIM model:
//    nimId          — exact NIM model identifier
//    thinking       — send chat_template_kwargs: { thinking: true }
//    reasoningField — NIM returns reasoning in this delta/message field
//    defaultTemp    — sensible default temperature for this model
//    maxTokens      — safe default max_tokens
// ─────────────────────────────────────────────
const MODEL_REGISTRY = {
  // ── DeepSeek ────────────────────────────────
  'deepseek-ai/deepseek-v4-pro': {
    nimId:          'deepseek-ai/deepseek-v4-pro',
    thinking:       true,
    reasoningField: 'reasoning_content',
    defaultTemp:    0.6,
    maxTokens:      16384,
  },
  'deepseek-ai/deepseek-r1': {
    nimId:          'deepseek-ai/deepseek-r1',
    thinking:       true,
    reasoningField: 'reasoning_content',
    defaultTemp:    0.6,
    maxTokens:      8192,
  },

  // ── Qwen ────────────────────────────────────
  'qwen/qwen3-235b-a22b': {
    nimId:          'qwen/qwen3-235b-a22b',
    thinking:       true,
    reasoningField: 'reasoning_content',
    defaultTemp:    0.7,
    maxTokens:      8192,
  },
  'qwen/qwen3-30b-a3b': {
    nimId:          'qwen/qwen3-30b-a3b',
    thinking:       true,
    reasoningField: 'reasoning_content',
    defaultTemp:    0.7,
    maxTokens:      8192,
  },
  'qwen/qwen3-coder-480b-a35b-instruct': {
    nimId:          'qwen/qwen3-coder-480b-a35b-instruct',
    thinking:       false,
    reasoningField: null,
    defaultTemp:    0.6,
    maxTokens:      8192,
  },
  // User-requested alias
  'qwen/qwen3.5-397b-a17b': {
    nimId:          'qwen/qwen3.5-397b-a17b',
    thinking:       true,
    reasoningField: 'reasoning_content',
    defaultTemp:    0.7,
    maxTokens:      16384,
  },

  // ── Llama ────────────────────────────────────
  'nvidia/llama-3.1-nemotron-ultra-253b-v1': {
    nimId:          'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    thinking:       true,
    reasoningField: 'reasoning_content',
    defaultTemp:    0.6,
    maxTokens:      8192,
  },
  'moonshotai/kimi-k2.6': {
    nimId:          'moonshotai/kimi-k2.6',
    thinking:       true,
    reasoningField: null,
    defaultTemp:    1.0,
    maxTokens:      16384,
  },

  // ── Mistral ──────────────────────────────────
  'mistralai/mistral-large-2-instruct': {
    nimId:          'mistralai/mistral-large-2-instruct',
    thinking:       false,
    reasoningField: null,
    defaultTemp:    0.7,
    maxTokens:      4096,
  },

  // ── OpenAI pass-through (if enabled on your NIM) ───
  'openai/gpt-oss-120b': {
    nimId:          'openai/gpt-oss-120b',
    thinking:       false,
    reasoningField: null,
    defaultTemp:    0.7,
    maxTokens:      4096,
  },
};

// ─────────────────────────────────────────────
//  OpenAI-alias → NIM model mapping
//
//  Clients that hardcode OpenAI names will be
//  routed here. Change targets to taste.
// ─────────────────────────────────────────────
const ALIAS_MAP = {
  'moonshotai/kimi-k2.6':   'moonshotai/kimi-k2.6',
  'deepseek-ai/deepseek-v4-pro':           'deepseek-ai/deepseek-v4-pro',
  'qwen/qwen3.5-397b-a17b':          'qwen/qwen3.5-397b-a17b',
  'gpt-4o-mini':     'qwen/qwen3-30b-a3b',
  'o1':              'deepseek-ai/deepseek-r1',
  'o1-mini':         'qwen/qwen3-30b-a3b',
  'claude-3-haiku':  'qwen/qwen3-30b-a3b',
  'gemini-pro':      'qwen/qwen3-235b-a22b',
};

// ─────────────────────────────────────────────
//  Global reasoning-display toggle
//  (per-request override via x-show-reasoning header also supported)
// ─────────────────────────────────────────────
const SHOW_REASONING_DEFAULT = process.env.SHOW_REASONING !== 'false'; // default ON

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

/**
 * Resolve an incoming model name to a registry entry.
 * Priority: exact registry match → alias → DEFAULT_MODEL.
 */
function resolveEntry(requested) {
  if (!requested) return MODEL_REGISTRY[DEFAULT_MODEL];

  // 1. Exact match in registry
  if (MODEL_REGISTRY[requested]) return MODEL_REGISTRY[requested];

  // 2. Alias mapping
  const aliased = ALIAS_MAP[requested];
  if (aliased && MODEL_REGISTRY[aliased]) return MODEL_REGISTRY[aliased];

  // 3. Partial/case-insensitive match (e.g. "deepseek-v4-pro" → full NIM id)
  const lower = requested.toLowerCase();
  const partialKey = Object.keys(MODEL_REGISTRY).find(k => k.toLowerCase().includes(lower));
  if (partialKey) {
    console.warn(`[proxy] Partial match "${requested}" → "${partialKey}"`);
    return MODEL_REGISTRY[partialKey];
  }

  // 4. Fallback
  console.warn(`[proxy] Unknown model "${requested}" — falling back to ${DEFAULT_MODEL}`);
  return MODEL_REGISTRY[DEFAULT_MODEL];
}

/** Build the NIM request body. */
function buildNimRequest(reqBody, entry) {
  const { messages, temperature, max_tokens, top_p, stream } = reqBody;

  const body = {
    model:       entry.nimId,
    messages,
    temperature: temperature ?? entry.defaultTemp,
    max_tokens:  max_tokens  ?? entry.maxTokens,
    top_p:       top_p       ?? 0.9,
    stream:      stream      ?? false,
  };

  if (entry.thinking) {
    body.extra_body = { chat_template_kwargs: { thinking: true } };
  }

  return body;
}

/** Merge reasoning + content for non-streaming responses. */
function mergeContent(reasoning, content, showReasoning) {
  if (!showReasoning || !reasoning) return content || '';
  return `<think>\n${reasoning}\n</think>\n\n${content || ''}`;
}

const nimHeaders = () => ({
  Authorization:  `Bearer ${NIM_API_KEY}`,
  'Content-Type': 'application/json',
});

function makeStreamChunk(model, content) {
  return {
    id:      `chatcmpl-inject-${Date.now()}`,
    object:  'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  };
}

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({
    status:           'ok',
    service:          'NVIDIA NIM Multi-Model Proxy',
    default_model:    DEFAULT_MODEL,
    show_reasoning:   SHOW_REASONING_DEFAULT,
    available_models: Object.keys(MODEL_REGISTRY),
  });
});

/** List all known models (registry + aliases) in OpenAI format. */
app.get('/v1/models', (_req, res) => {
  const seen = new Set();
  const data = [];

  // Registry entries
  for (const [key, entry] of Object.entries(MODEL_REGISTRY)) {
    if (seen.has(key)) continue;
    seen.add(key);
    data.push({
      id:       key,
      object:   'model',
      created:  Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim-proxy',
      root:     entry.nimId,
      metadata: { thinking: entry.thinking },
    });
  }

  // Alias entries
  for (const [alias, target] of Object.entries(ALIAS_MAP)) {
    if (seen.has(alias)) continue;
    seen.add(alias);
    data.push({
      id:       alias,
      object:   'model',
      created:  Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim-proxy',
      root:     target,
    });
  }

  res.json({ object: 'list', data });
});

// ─────────────────────────────────────────────
//  Main proxy
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const { stream = false } = req.body;

  // Per-request reasoning toggle via header (overrides env default)
  const showReasoning = req.headers['x-show-reasoning'] !== undefined
    ? req.headers['x-show-reasoning'] !== 'false'
    : SHOW_REASONING_DEFAULT;

  const entry      = resolveEntry(req.body.model);
  const nimRequest = buildNimRequest(req.body, entry);
  const reasoningField = entry.reasoningField; // null for non-thinking models

  try {
    const upstream = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers:      nimHeaders(),
        responseType: stream ? 'stream' : 'json',
        timeout:      stream ? 0 : 120_000,
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

          if (payload === '[DONE]') {
            res.write('data: [DONE]\n\n');
            continue;
          }

          let data;
          try { data = JSON.parse(payload); }
          catch { res.write(line + '\n'); continue; }

          const delta = data.choices?.[0]?.delta;
          if (!delta) { res.write(`data: ${JSON.stringify(data)}\n\n`); continue; }

          // Extract reasoning chunk (only relevant if model supports it)
          const reasoning = reasoningField ? (delta[reasoningField] ?? '') : '';
          const content   = delta.content ?? '';

          // Strip all non-standard fields before forwarding
          if (reasoningField) delete delta[reasoningField];

          if (reasoning || content) {
            if (showReasoning && reasoning) {
              let out = '';
              if (!thinkOpen) { out += '<think>\n'; thinkOpen = true; }
              out += reasoning;
              if (content) { out += '\n</think>\n\n' + content; thinkOpen = false; }
              delta.content = out;
            } else if (content) {
              // Close open think block when content starts arriving
              if (thinkOpen && showReasoning) {
                delta.content = '\n</think>\n\n' + content;
                thinkOpen = false;
              } else {
                delta.content = content;
              }
            } else {
              // Pure reasoning chunk, not shown
              delta.content = showReasoning ? reasoning : '';
            }
          }

          res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
      });

      upstream.data.on('end', () => {
        if (thinkOpen && showReasoning) {
          const closing = makeStreamChunk(entry.nimId, '\n</think>\n\n');
          res.write(`data: ${JSON.stringify(closing)}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });

      upstream.data.on('error', (err) => {
        console.error('[stream error]', err.message);
        res.end();
      });

      res.on('close', () => upstream.data.destroy?.());

    // ── Non-streaming ───────────────────────────────────────────────
    } else {
      const nimData = upstream.data;

      const choices = (nimData.choices ?? []).map((choice, idx) => {
        const reasoning = reasoningField ? (choice.message?.[reasoningField] ?? '') : '';
        return {
          index: idx,
          message: {
            role:    choice.message.role,
            content: mergeContent(reasoning, choice.message.content, showReasoning),
          },
          finish_reason: choice.finish_reason ?? 'stop',
        };
      });

      res.json({
        id:      `chatcmpl-${Date.now()}`,
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model:   req.body.model ?? entry.nimId,
        choices,
        usage:   nimData.usage ?? {
          prompt_tokens:     0,
          completion_tokens: 0,
          total_tokens:      0,
        },
      });
    }

  } catch (err) {
    const status  = err.response?.status ?? 500;
    const message = err.response?.data?.detail
      ?? err.response?.data?.error?.message
      ?? err.message
      ?? 'Internal proxy error';

    console.error(`[proxy error] ${status} (${entry.nimId}): ${message}`);

    res.status(status).json({
      error: {
        message,
        type:  status === 429 ? 'rate_limit_error'
             : status >= 500  ? 'api_error'
             :                  'invalid_request_error',
        code:  status,
        model: entry.nimId,
      },
    });
  }
});

// ─────────────────────────────────────────────
//  404 catch-all
// ─────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not supported`,
      type:    'invalid_request_error',
      code:    404,
    },
  });
});

// ─────────────────────────────────────────────
//  Boot
// ─────────────────────────────────────────────
app.listen(PORT, () => {
  const modelList = Object.keys(MODEL_REGISTRY).join('\n    ');
  console.log(`\n🚀  NVIDIA NIM Multi-Model Proxy — port ${PORT}`);
  console.log(`    Default model  : ${DEFAULT_MODEL}`);
  console.log(`    Show reasoning : ${SHOW_REASONING_DEFAULT ? '✅ ON' : '❌ OFF'} (override per-request via x-show-reasoning header)`);
  console.log(`\n    Registered models:\n    ${modelList}\n`);
  console.log(`    Health check   : http://localhost:${PORT}/health`);
  console.log(`    Model list     : http://localhost:${PORT}/v1/models\n`);
});
