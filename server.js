// server.js — OpenAI-compatible proxy → NVIDIA NIM (deepseek-ai/deepseek-v4-pro)
const express = require('express');
const cors    = require('cors');
const axios   = require('axios');

const app  = express();
const PORT = process.env.PORT || 3000;

// ─── Config ───────────────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;            // required — set in env
const NIM_MODEL    = 'deepseek-ai/deepseek-v4-pro';      // fixed target model

// Show <think>…</think> reasoning blocks in the response?
const SHOW_REASONING    = process.env.SHOW_REASONING    === 'true' || false;

// Pass chat_template_kwargs { thinking: true } to enable native thinking mode?
const ENABLE_THINKING   = process.env.ENABLE_THINKING   === 'true' || true;

// ─── Middleware ────────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '400mb' }));
app.use(express.urlencoded({ limit: '400mb', extended: true }));

// ─── Request logger ────────────────────────────────────────────────────────────

app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// ─── Health check ──────────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({
    status:          'ok',
    service:         'OpenAI → NVIDIA NIM Proxy (deepseek-v4-pro)',
    model:           NIM_MODEL,
    show_reasoning:  SHOW_REASONING,
    thinking_mode:   ENABLE_THINKING,
    nim_api_key_set: !!NIM_API_KEY,
  });
});

// ─── Models list (OpenAI-compatible) ──────────────────────────────────────────

app.get('/v1/models', (_req, res) => {
  res.json({
    object: 'list',
    data: [{
      id:         NIM_MODEL,
      object:     'model',
      created:    Math.floor(Date.now() / 1000),
      owned_by:   'nvidia-nim-proxy',
    }],
  });
});

// ─── Chat completions ──────────────────────────────────────────────────────────

app.post('/v1/chat/completions', async (req, res) => {
  if (!NIM_API_KEY) {
    return res.status(500).json({
      error: { message: 'NIM_API_KEY environment variable is not set.', type: 'server_error' },
    });
  }

  try {
    const {
      messages,
      temperature = 1.0,
      max_tokens  = 16384,
      stream      = true,
    } = req.body;

    // Build NIM request — model is always deepseek-v4-pro
    const nimRequest = {
      model:       NIM_MODEL,
      messages:    messages,
      temperature: temperature,
      max_tokens:  max_tokens,
      stream:      stream,
      ...(ENABLE_THINKING && {
        extra_body: { chat_template_kwargs: { thinking: true } },
      }),
    };

    const nimResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization:  `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
        },
        responseType: stream ? 'stream' : 'json',
      }
    );

    // ── Streaming ────────────────────────────────────────────────────────────

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');

      let buffer          = '';
      let reasoningOpen   = false;   // tracks whether <think> tag is still open

      nimResponse.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';   // keep incomplete last line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const reasoning = delta.reasoning_content ?? '';
              const content   = delta.content           ?? '';

              if (SHOW_REASONING) {
                let out = '';

                if (reasoning && !reasoningOpen) {
                  out += '<think>\n' + reasoning;
                  reasoningOpen = true;
                } else if (reasoning) {
                  out += reasoning;
                }

                if (content && reasoningOpen) {
                  out += '\n</think>\n\n' + content;
                  reasoningOpen = false;
                } else if (content) {
                  out += content;
                }

                delta.content = out;
              } else {
                // Strip reasoning; forward only real content
                delta.content = content;
              }

              delete delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            // Unparseable line — forward as-is
            res.write(line + '\n');
          }
        }
      });

      nimResponse.data.on('end',   ()    => res.end());
      nimResponse.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

      return;
    }

    // ── Non-streaming ────────────────────────────────────────────────────────

    const choices = nimResponse.data.choices.map((choice) => {
      let content = choice.message?.content ?? '';

      if (SHOW_REASONING && choice.message?.reasoning_content) {
        content = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${content}`;
      }

      return {
        index:         choice.index,
        message:       { role: choice.message.role, content },
        finish_reason: choice.finish_reason,
      };
    });

    res.json({
      id:      `chatcmpl-${Date.now()}`,
      object:  'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model:   NIM_MODEL,
      choices,
      usage:   nimResponse.data.usage ?? {
        prompt_tokens:     0,
        completion_tokens: 0,
        total_tokens:      0,
      },
    });

  } catch (error) {
    const status  = error.response?.status ?? 500;
    const message = error.response?.data?.detail ?? error.message ?? 'Internal server error';

    console.error(`Proxy error [${status}]:`, message);

    res.status(status).json({
      error: { message, type: 'proxy_error', code: status },
    });
  }
});

// ─── 404 fallback ──────────────────────────────────────────────────────────────

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} is not supported by this proxy.`,
      type:    'invalid_request_error',
      code:    404,
    },
  });
});

// ─── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(` OpenAI → NVIDIA NIM Proxy`);
  console.log(` Model:          ${NIM_MODEL}`);
  console.log(` Port:           ${PORT}`);
  console.log(` Health:         http://localhost:${PORT}/health`);
  console.log(` Show reasoning: ${SHOW_REASONING}`);
  console.log(` Thinking mode:  ${ENABLE_THINKING}`);
  console.log(` API key set:    ${!!NIM_API_KEY}`);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
});
