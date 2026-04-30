// server.js — OpenAI-compatible proxy for DeepSeek-V4-Pro via NVIDIA NIM
'use strict';

const express = require('express');
const cors    = require('cors');
const axios   = require('axios');

// ─────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────
const PORT          = process.env.PORT          || 3000;
const NIM_API_BASE  = process.env.NIM_API_BASE  || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY   = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('[FATAL] NIM_API_KEY is not set. Exiting.');
  process.exit(1);
}

const TARGET_MODEL = 'deepseek-ai/deepseek-v4-pro';

// 🔥 Toggle: include <think>…</think> block in the visible output
const SHOW_REASONING  = process.env.SHOW_REASONING  !== 'false'; // default ON
// 🔥 Toggle: send chat_template_kwargs.thinking = true to the NIM API
const ENABLE_THINKING = process.env.ENABLE_THINKING !== 'false'; // default ON

// Aliases that arrive from various OpenAI-compatible clients
const MODEL_ALIASES = new Set([
  'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o',
  'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
  'gemini-pro', 'o1', 'o1-mini',
  TARGET_MODEL,
]);

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

/** Resolve any incoming model name to our target. */
function resolveModel(requested) {
  if (!requested || MODEL_ALIASES.has(requested)) return TARGET_MODEL;
  console.warn(`[proxy] Unknown model "${requested}" — routing to ${TARGET_MODEL}`);
  return TARGET_MODEL;
}

/** Build the NIM request body from an OpenAI-style request. */
function buildNimRequest({ model, messages, temperature, max_tokens, top_p, stream }) {
  const body = {
    model:       resolveModel(model),
    messages,
    temperature: temperature ?? 0.6,
    max_tokens:  max_tokens  ?? 8192,
    top_p:       top_p       ?? 0.9,
    stream:      stream      ?? false,
  };

  if (ENABLE_THINKING) {
    body.extra_body = { chat_template_kwargs: { thinking: true } };
  }

  return body;
}

/**
 * Merge reasoning_content + content into a single string
 * according to the SHOW_REASONING flag.
 */
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
    service:         'DeepSeek-V4-Pro NIM Proxy',
    target_model:    TARGET_MODEL,
    show_reasoning:  SHOW_REASONING,
    enable_thinking: ENABLE_THINKING,
  });
});

app.get('/v1/models', (_req, res) => {
  const data = [...MODEL_ALIASES].map(id => ({
    id,
    object:   'model',
    created:  Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy',
    root:     TARGET_MODEL,
  }));
  res.json({ object: 'list', data });
});

// ─────────────────────────────────────────────
//  Main proxy endpoint
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const { stream = false } = req.body;

  try {
    const nimRequest = buildNimRequest(req.body);

    const upstream = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers:      nimHeaders(),
        responseType: stream ? 'stream' : 'json',
        timeout:      stream ? 0 : 120_000,
      }
    );

    // ── Streaming path ──────────────────────────────────────────────
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

          if (SHOW_REASONING) {
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
        // Close any unclosed <think> block if the stream ended mid-reasoning
        if (thinkOpen && SHOW_REASONING) {
          const closing = makeStreamChunk(req.body.model, '\n</think>\n\n');
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

    // ── Non-streaming path ──────────────────────────────────────────
    } else {
      const nimData = upstream.data;

      const choices = (nimData.choices ?? []).map((choice, idx) => ({
        index: idx,
        message: {
          role:    choice.message.role,
          content: mergeContent(
            choice.message.reasoning_content,
            choice.message.content
          ),
        },
        finish_reason: choice.finish_reason ?? 'stop',
      }));

      res.json({
        id:      `chatcmpl-${Date.now()}`,
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model:   req.body.model ?? TARGET_MODEL,
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
//  Utilities
// ─────────────────────────────────────────────

function makeStreamChunk(model, content) {
  return {
    id:      `chatcmpl-inject-${Date.now()}`,
    object:  'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model:   model ?? TARGET_MODEL,
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  };
}

// ─────────────────────────────────────────────
//  Boot
// ─────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀  DeepSeek-V4-Pro NIM Proxy — port ${PORT}`);
  console.log(`    Target model   : ${TARGET_MODEL}`);
  console.log(`    Show reasoning : ${SHOW_REASONING  ? '✅ ON' : '❌ OFF'}`);
  console.log(`    Thinking mode  : ${ENABLE_THINKING ? '✅ ON' : '❌ OFF'}`);
  console.log(`    Health check   : http://localhost:${PORT}/health\n`);
});
