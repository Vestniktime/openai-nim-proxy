// server.js — OpenAI-compatible multi-model proxy for NVIDIA NIM
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

// ── Model catalogue ───────────────────────────────────────────────────────────
//
//  Each entry describes one NIM-hosted model:
//
//    nimId        — exact model string sent to the NIM API
//    aliases      — list of OpenAI-style names that map to this model
//    supportsThink — whether chat_template_kwargs.thinking is accepted
//    defaultTemp  — sensible starting temperature for this model
//    defaultTokens— default max_tokens for this model
//
const MODELS = [
  {
    nimId:         'deepseek-ai/deepseek-v4-pro',
    aliases:       ['deepseek-v4-pro', 'gpt-4o', 'gpt-4', 'gpt-4-turbo'],
    supportsThink: true,
    defaultTemp:   0.6,
    defaultTokens: 16384,
  },
  {
    nimId:         'qwen/qwen3.5-397b-a17b',
    aliases:       ['qwen3.5-397b-a17b', 'qwen3.5-397b-a17b', 'gpt-3.5-turbo', 'claude-3-haiku'],
    supportsThink: true,
    defaultTemp:   0.7,
    defaultTokens: 16384,
  },
  {
    nimId:         'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    aliases:       ['nemotron-ultra', 'llama-nemotron', 'claude-3-opus'],
    supportsThink: true,
    defaultTemp:   0.6,
    defaultTokens: 8192,
  },
  {
    nimId:         'moonshotai/kimi-k2.6',
    aliases:       ['kimi-k2', 'claude-3-sonnet'],
    supportsThink: true,
    defaultTemp:   0.7,
    defaultTokens: 16384,
  },
  {
    nimId:         'openai/gpt-oss-120b',
    aliases:       ['gpt-oss-120b', 'gemini-pro', 'o1'],
    supportsThink: false,
    defaultTemp:   0.7,
    defaultTokens: 4096,
  },
];

// Default model used when nothing matches
const DEFAULT_MODEL = MODELS[0];

// Build fast-lookup maps once at startup
const BY_NIM_ID  = new Map(); // nimId  → entry
const BY_ALIAS   = new Map(); // alias  → entry

for (const entry of MODELS) {
  BY_NIM_ID.set(entry.nimId, entry);
  for (const alias of entry.aliases) {
    BY_ALIAS.set(alias.toLowerCase(), entry);
  }
  // Also allow addressing by nimId directly
  BY_ALIAS.set(entry.nimId.toLowerCase(), entry);
}

// ── Feature toggles ───────────────────────────────────────────────────────────
// env: SHOW_REASONING=false  → strip <think> blocks from output   (default: ON)
// env: ENABLE_THINKING=false → don't send thinking param to NIM   (default: ON)
const SHOW_REASONING  = process.env.SHOW_REASONING  !== 'false';
const ENABLE_THINKING = process.env.ENABLE_THINKING !== 'false';

// ─────────────────────────────────────────────
//  Express app
// ─────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

/** Look up a model entry by any name the client might send. */
function resolveEntry(requested) {
  if (!requested) return DEFAULT_MODEL;
  const entry = BY_ALIAS.get(requested.toLowerCase())
             ?? BY_NIM_ID.get(requested);
  if (!entry) {
    console.warn(`[proxy] Unknown model "${requested}" — falling back to ${DEFAULT_MODEL.nimId}`);
    return DEFAULT_MODEL;
  }
  return entry;
}

/** Build the NIM request body. */
function buildNimRequest(body) {
  const entry = resolveEntry(body.model);

  const nimBody = {
    model:       entry.nimId,
    messages:    body.messages,
    temperature: body.temperature ?? entry.defaultTemp,
    max_tokens:  body.max_tokens  ?? entry.defaultTokens,
    top_p:       body.top_p       ?? 0.9,
    stream:      body.stream      ?? false,
  };

  if (ENABLE_THINKING && entry.supportsThink) {
    nimBody.extra_body = { chat_template_kwargs: { thinking: true } };
  }

  return { nimBody, entry };
}

/** Wrap reasoning in <think> tags if SHOW_REASONING is enabled. */
function mergeContent(reasoningContent, content) {
  if (!SHOW_REASONING || !reasoningContent) return content || '';
  return `<think>\n${reasoningContent}\n</think>\n\n${content || ''}`;
}

/** Inject a synthetic SSE chunk (used to close an open <think> block). */
function makeStreamChunk(model, content) {
  return {
    id:      `chatcmpl-inject-${Date.now()}`,
    object:  'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
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
    default_model:   DEFAULT_MODEL.nimId,
    show_reasoning:  SHOW_REASONING,
    enable_thinking: ENABLE_THINKING,
    available_models: MODELS.map(m => ({
      nim_id:        m.nimId,
      aliases:       m.aliases,
      supports_think: m.supportsThink,
    })),
  });
});

/** /v1/models — lists every alias so any OpenAI client can enumerate them. */
app.get('/v1/models', (_req, res) => {
  const data = [];
  for (const entry of MODELS) {
    // Expose the real NIM ID plus every alias as separate "model" objects
    for (const id of [entry.nimId, ...entry.aliases]) {
      data.push({
        id,
        object:         'model',
        created:        Math.floor(Date.now() / 1000),
        owned_by:       'nvidia-nim-proxy',
        root:           entry.nimId,
        supports_think: entry.supportsThink,
      });
    }
  }
  res.json({ object: 'list', data });
});

// ─────────────────────────────────────────────
//  Main proxy — /v1/chat/completions
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const { stream = false } = req.body;

  try {
    const { nimBody, entry } = buildNimRequest(req.body);

    console.log(`[proxy] ${req.body.model ?? '(none)'} → ${entry.nimId}${stream ? ' [stream]' : ''}`);

    const upstream = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimBody,
      {
        headers:      nimHeaders(),
        responseType: stream ? 'stream' : 'json',
        timeout:      stream ? 0 : 120_000,
      }
    );

    // ── Streaming ─────────────────────────────────────────────────────────────
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

          const reasoning = delta.reasoning_content ?? '';
          const content   = delta.content           ?? '';
          delete delta.reasoning_content;

          if (SHOW_REASONING && entry.supportsThink) {
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
            // Non-thinking models or SHOW_REASONING=false: just pass content through
            delta.content = content;
          }

          res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
      });

      upstream.data.on('end', () => {
        if (thinkOpen && SHOW_REASONING) {
          const closing = makeStreamChunk(req.body.model ?? entry.nimId, '\n</think>\n\n');
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

    // ── Non-streaming ─────────────────────────────────────────────────────────
    } else {
      const nimData = upstream.data;

      const choices = (nimData.choices ?? []).map((choice, idx) => ({
        index: idx,
        message: {
          role:    choice.message.role,
          content: (SHOW_REASONING && entry.supportsThink)
            ? mergeContent(choice.message.reasoning_content, choice.message.content)
            : (choice.message.content || ''),
        },
        finish_reason: choice.finish_reason ?? 'stop',
      }));

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
  console.log(`\n🚀  NVIDIA NIM Multi-Model Proxy — port ${PORT}`);
  console.log(`    Show reasoning : ${SHOW_REASONING  ? '✅ ON' : '❌ OFF'}`);
  console.log(`    Thinking mode  : ${ENABLE_THINKING ? '✅ ON' : '❌ OFF'}`);
  console.log(`    Health check   : http://localhost:${PORT}/health`);
  console.log(`\n    Loaded models:`);
  for (const m of MODELS) {
    console.log(`      • ${m.nimId}${m.supportsThink ? ' [think]' : ''}`);
    console.log(`        aliases: ${m.aliases.join(', ')}`);
  }
  console.log('');
});
