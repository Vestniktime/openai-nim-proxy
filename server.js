// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const axios   = require('axios');
const http    = require('http');
const https   = require('https');

const app  = express();
const PORT = process.env.PORT || 3000;

// ─────────────────────────────────────────────
// ✅ CORS
// ─────────────────────────────────────────────
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin',  '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE');
  res.setHeader('Access-Control-Allow-Headers',
    'Content-Type, Authorization, X-Requested-With, Accept, Origin');
  res.setHeader('Access-Control-Max-Age', '86400');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// ─────────────────────────────────────────────
// ✅ УНИВЕРСАЛЬНЫЙ ПАРСЕР ТЕЛА
// ─────────────────────────────────────────────
app.use((req, res, next) => {
  if (req.method === 'GET' || req.method === 'OPTIONS' || req.method === 'HEAD') return next();
  let raw = '';
  req.setEncoding('utf8');
  req.on('data', chunk => { raw += chunk; });
  req.on('end', () => {
    if (!raw) return next();
    try { req.body = JSON.parse(raw); } catch { req.body = raw; }
    next();
  });
  req.on('error', () => next());
});

// ─────────────────────────────────────────────
// ⚙️ CONFIG
// ─────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

const SHOW_REASONING            = false;
const DEEPSEEK_REASONING_EFFORT = process.env.DEEPSEEK_REASONING_EFFORT || 'low';

const FIRST_TOKEN_TIMEOUT       = parseInt(process.env.FIRST_TOKEN_TIMEOUT || '480000'); // 8 мин
const IDLE_TIMEOUT              = parseInt(process.env.IDLE_TIMEOUT        || '120000'); // 2 мин
const KEEPALIVE_INTERVAL        = parseInt(process.env.KEEPALIVE_INTERVAL  || '8000');   // 8 сек
const MAX_RETRIES_SAME_MODEL    = parseInt(process.env.MAX_RETRIES_SAME_MODEL || '2');

// RP-промпт отключён — промпты управляются через SillyTavern
const ENABLE_RP_STYLE_PROMPT = false;
const RP_STYLE_INSTRUCTION   = '';

const MIN_RESPONSE_TOKENS = parseInt(process.env.MIN_RESPONSE_TOKENS || '2048');

const httpAgent  = new http.Agent ({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

// ─────────────────────────────────────────────
// 📋 МОДЕЛИ
// ─────────────────────────────────────────────

const DEEPSEEK_V4_PRO_ID   = 'deepseek-ai/deepseek-v4-pro-0813';
const DEEPSEEK_V4_FLASH_ID = 'deepseek-ai/deepseek-v4-flash-0731';

function isDeepSeekV4(m) {
  return m.includes('deepseek-v4-pro') || m.includes('deepseek-v4-flash');
}

const OPTIONAL_THINKING_MODELS = new Set([
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-next-80b-a3b-thinking',
]);

const MODEL_MAPPING = {
  'deepseek-v4-pro':   DEEPSEEK_V4_PRO_ID,
  'deepseek-v4-flash': DEEPSEEK_V4_FLASH_ID,
  'gpt-4o':            DEEPSEEK_V4_PRO_ID,
  'gpt-3.5-turbo':     'meta/llama-3.1-8b-instruct',
  'gpt-4':             'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo':       'meta/llama-3.1-405b-instruct',
  'claude-3-opus':     'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet':   'meta/llama-3.1-70b-instruct',
  'gemini-pro':        'mistralai/mistral-large-2-instruct',
};

// ─────────────────────────────────────────────
// 🛠️ HELPERS
// ─────────────────────────────────────────────

function buildThinkingParams(nimModel) {
  if (isDeepSeekV4(nimModel)) {
    const effort = (DEEPSEEK_REASONING_EFFORT || 'low').toLowerCase();
    if (effort === 'max') {
      return { chat_template_kwargs: { enable_thinking: true, thinking: true, reasoning_effort: 'max' } };
    }
    if (effort === 'high' || effort === 'medium') {
      return { chat_template_kwargs: { enable_thinking: true, thinking: true, reasoning_effort: 'high' } };
    }
    // low / off / none → Non-think (без рассуждений, быстрее)
    return { chat_template_kwargs: { enable_thinking: false, thinking: false } };
  }
  if (OPTIONAL_THINKING_MODELS.has(nimModel)) {
    return { chat_template_kwargs: { thinking: true } };
  }
  return {};
}

function applyStylePrompt(messages) {
  if (!ENABLE_RP_STYLE_PROMPT || !Array.isArray(messages) || messages.length === 0) return messages;
  const out = messages.map(m => ({ ...m }));
  if (out[0]?.role === 'system') {
    out[0] = { ...out[0], content: `${out[0].content}\n\n${RP_STYLE_INSTRUCTION}` };
  } else {
    out.unshift({ role: 'system', content: RP_STYLE_INSTRUCTION });
  }
  return out;
}

function mergeContent(content, reasoningContent) {
  if (SHOW_REASONING && reasoningContent) return `<think>\n${reasoningContent}\n</think>\n\n${content || ''}`;
  return content || '';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function withSmartTimeout(stream, label) {
  let first = false;
  let timer = setTimeout(() => {
    console.error(`[Timeout] ${label}: нет первого токена за ${FIRST_TOKEN_TIMEOUT / 1000}с`);
    stream.destroy(new Error('first-token timeout'));
  }, FIRST_TOKEN_TIMEOUT);

  stream.on('data', () => {
    clearTimeout(timer);
    if (!first) {
      first = true;
      console.log(`[${label}] ✓ первый токен`);
    }
    timer = setTimeout(() => {
      console.error(`[Idle] ${label}: нет данных ${IDLE_TIMEOUT / 1000}с`);
      stream.destroy(new Error('idle timeout'));
    }, IDLE_TIMEOUT);
  });

  stream.on('end',   () => clearTimeout(timer));
  stream.on('error', () => clearTimeout(timer));
}

function resolveModel(model) {
  return MODEL_MAPPING[model] || model;
}

function copyDefinedParams(source, target, keys) {
  for (const key of keys) {
    if (source[key] !== undefined) target[key] = source[key];
  }
}

// ─────────────────────────────────────────────
// NVIDIA NIM: CHAT COMPLETIONS
// ─────────────────────────────────────────────

async function nimFetch(nimModel, body) {
  const isDeepSeek = isDeepSeekV4(nimModel);

  const payload = {
    model:       nimModel,
    messages:    applyStylePrompt(body.messages),
    temperature: body.temperature ?? (isDeepSeek ? 1.0 : 0.7),
    top_p:       body.top_p       ?? (isDeepSeek ? 0.95 : undefined),
    max_tokens:  isDeepSeek
      ? Math.max(body.max_tokens || 0, MIN_RESPONSE_TOKENS)
      : (body.max_tokens || 4096),
    stream: true,
    ...buildThinkingParams(nimModel),
  };

  const response = await axios.post(
    `${NIM_API_BASE}/chat/completions`,
    payload,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
      },
      responseType: 'stream',
      timeout: 0,
      httpAgent,
      httpsAgent,
      validateStatus: null,
    }
  );

  if (response.status >= 400) {
    const raw = await new Promise(r => {
      let s = '';
      response.data.on('data', c => { s += c.toString(); });
      response.data.on('end',  () => r(s));
      response.data.on('error',() => r(''));
    });
    const err = new Error(`NIM ${response.status}: ${raw}`);
    err.status = response.status;
    throw err;
  }

  withSmartTimeout(response.data, nimModel);
  return response.data;
}

async function nimFetchSameModel(nimModel, body) {
  let lastError;

  for (let attempt = 1; attempt <= MAX_RETRIES_SAME_MODEL; attempt++) {
    const isLast = attempt === MAX_RETRIES_SAME_MODEL;

    try {
      console.log(`[NIM chat] Попытка ${attempt}/${MAX_RETRIES_SAME_MODEL}: ${nimModel}`);
      return await nimFetch(nimModel, body);
    } catch (err) {
      lastError = err;
      if (isLast) throw err;

      console.warn(
        `[NIM chat] ${nimModel} → ${err.status || err.message}, повтор через 2с…`
      );
      await sleep(2000);
    }
  }

  throw lastError;
}

// ─────────────────────────────────────────────
// NVIDIA NIM: TEXT COMPLETIONS
// ─────────────────────────────────────────────

async function nimCompletionFetch(nimModel, body) {
  const isDeepSeek = isDeepSeekV4(nimModel);

  const payload = {
    model:       nimModel,
    prompt:      body.prompt,
    temperature: body.temperature ?? (isDeepSeek ? 1.0 : 0.7),
    top_p:       body.top_p       ?? (isDeepSeek ? 0.95 : undefined),
    max_tokens:  isDeepSeek
      ? Math.max(body.max_tokens || 0, MIN_RESPONSE_TOKENS)
      : (body.max_tokens || 4096),
    // Внутри всегда stream=true. Даже если клиент запросил non-stream,
    // прокси получает токены постепенно и затем собирает обычный JSON.
    stream: true,
  };

  // Параметры, применимые к OpenAI-compatible Text Completions.
  // Передаём только те, которые реально пришли от клиента.
  copyDefinedParams(body, payload, [
    'stop',
    'frequency_penalty',
    'presence_penalty',
    'seed',
    'n',
    'logprobs',
    'echo',
    'suffix',
    'user',

    // Расширения, которые могут поддерживаться NIM/vLLM конкретной модели.
    'top_k',
    'min_p',
    'repetition_penalty',
    'ignore_eos',
    'min_tokens',
    'include_stop_str_in_output',
  ]);

  const response = await axios.post(
    `${NIM_API_BASE}/completions`,
    payload,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
      },
      responseType: 'stream',
      timeout: 0,
      httpAgent,
      httpsAgent,
      validateStatus: null,
    }
  );

  if (response.status >= 400) {
    const raw = await new Promise(r => {
      let s = '';
      response.data.on('data', c => { s += c.toString(); });
      response.data.on('end',  () => r(s));
      response.data.on('error',() => r(''));
    });
    const err = new Error(`NIM ${response.status}: ${raw}`);
    err.status = response.status;
    throw err;
  }

  withSmartTimeout(response.data, `${nimModel} [completion]`);
  return response.data;
}

async function nimCompletionFetchSameModel(nimModel, body) {
  let lastError;

  for (let attempt = 1; attempt <= MAX_RETRIES_SAME_MODEL; attempt++) {
    const isLast = attempt === MAX_RETRIES_SAME_MODEL;

    try {
      console.log(
        `[NIM completion] Попытка ${attempt}/${MAX_RETRIES_SAME_MODEL}: ${nimModel}`
      );
      return await nimCompletionFetch(nimModel, body);
    } catch (err) {
      lastError = err;
      if (isLast) throw err;

      console.warn(
        `[NIM completion] ${nimModel} → ${err.status || err.message}, повтор через 2с…`
      );
      await sleep(2000);
    }
  }

  throw lastError;
}

// ─────────────────────────────────────────────
// KEEPALIVE
// ─────────────────────────────────────────────

// Валидный SSE-чанк OpenAI Chat Completions с пустой delta.
// Клиент видит живую активность и не обрывает соединение.
function makeKeepaliveChunk(model) {
  return `data: ${JSON.stringify({
    id: 'chatcmpl-keepalive',
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta: {}, finish_reason: null }],
  })}\n\n`;
}

// Валидный SSE-чанк OpenAI Text Completions с пустым text.
function makeCompletionKeepaliveChunk(model) {
  return `data: ${JSON.stringify({
    id: 'cmpl-keepalive',
    object: 'text_completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      text: '',
      index: 0,
      logprobs: null,
      finish_reason: null,
    }],
  })}\n\n`;
}

// ─────────────────────────────────────────────
// STREAM COLLECTORS
// ─────────────────────────────────────────────

function collectStream(nimStream) {
  return new Promise((resolve, reject) => {
    let buf = '';
    let fullContent = '';
    let fullReasoning = '';
    let lastChoice = null;
    let usage = null;

    nimStream.on('data', chunk => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ') || line.includes('[DONE]')) continue;

        try {
          const data = JSON.parse(line.slice(6));
          if (data.usage) usage = data.usage;

          const ch = data.choices?.[0];
          if (!ch) continue;

          lastChoice     = ch;
          fullContent   += ch.delta?.content           || '';
          fullReasoning += ch.delta?.reasoning_content || '';
        } catch (_) {}
      }
    });

    nimStream.on('end', () => {
      resolve({ fullContent, fullReasoning, lastChoice, usage });
    });
    nimStream.on('error', reject);
  });
}

function collectCompletionStream(nimStream) {
  return new Promise((resolve, reject) => {
    let buf = '';
    let usage = null;
    const choices = new Map();

    const consumeLine = line => {
      if (!line.startsWith('data: ') || line.includes('[DONE]')) return;

      try {
        const data = JSON.parse(line.slice(6));
        if (data.usage) usage = data.usage;

        for (const ch of data.choices || []) {
          const index = Number.isInteger(ch.index) ? ch.index : 0;
          const prev = choices.get(index) || {
            text: '',
            lastChoice: null,
          };

          prev.text += ch.text || '';
          prev.lastChoice = ch;
          choices.set(index, prev);
        }
      } catch (_) {}
    };

    nimStream.on('data', chunk => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) consumeLine(line);
    });

    nimStream.on('end', () => {
      if (buf) consumeLine(buf);

      const finalChoices = [...choices.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([index, state]) => ({
          index,
          text: state.text,
          logprobs: state.lastChoice?.logprobs ?? null,
          finish_reason: state.lastChoice?.finish_reason || 'stop',
        }));

      resolve({ choices: finalChoices, usage });
    });

    nimStream.on('error', reject);
  });
}

// ─────────────────────────────────────────────
// 🌐 ENDPOINTS
// ─────────────────────────────────────────────

app.get('/ping', (_, res) => res.send('pong'));

app.get('/health', (_, res) => res.json({
  status: 'ok',
  service: 'OpenAI → NVIDIA NIM Proxy',
  nim_key_set: !!NIM_API_KEY,
  deepseek_mode: DEEPSEEK_REASONING_EFFORT,
  rp_style_prompt_enabled: false,
  min_response_tokens: MIN_RESPONSE_TOKENS,
  max_retries_same_model: MAX_RETRIES_SAME_MODEL,
  first_token_timeout_ms: FIRST_TOKEN_TIMEOUT,
  idle_timeout_ms: IDLE_TIMEOUT,
  keepalive_interval_ms: KEEPALIVE_INTERVAL,
  strategy: 'SSE-заголовки + keepalive отправляются НЕМЕДЛЕННО, до ответа NIM',
  supported_endpoints: [
    '/v1/chat/completions',
    '/v1/completions',
  ],
  available_models: Object.keys(MODEL_MAPPING),
}));

app.get('/v1/models', (_, res) => res.json({
  object: 'list',
  // Показываем алиасы + реальные NIM-модели из них.
  // В SillyTavern можно вписать ЛЮБОЙ NIM model ID напрямую —
  // он будет передан в API без изменений.
  data: [
    ...Object.entries(MODEL_MAPPING).map(([alias, real]) => ({
      id: alias,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'alias',
      description: `→ ${real}`,
    })),
    {
      id: DEEPSEEK_V4_PRO_ID,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim',
    },
    {
      id: DEEPSEEK_V4_FLASH_ID,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim',
    },
    {
      id: 'meta/llama-3.1-70b-instruct',
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim',
    },
    {
      id: 'meta/llama-3.1-405b-instruct',
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim',
    },
    {
      id: 'mistralai/mistral-large-2-instruct',
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim',
    },
  ],
}));

// ─────────────────────────────────────────────
// 💬 CHAT COMPLETIONS
// ─────────────────────────────────────────────

app.post('/v1/chat/completions', async (req, res) => {
  const body = req.body || {};

  console.log(
    '[Chat request] model:', body.model,
    '| messages:',
    Array.isArray(body.messages) ? body.messages.length : typeof body.messages,
    '| stream:', body.stream
  );

  if (!NIM_API_KEY) {
    return res.status(500).json({
      error: { message: 'NIM_API_KEY не задан', code: 500 },
    });
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return res.status(400).json({
      error: { message: 'messages не должен быть пустым', code: 400 },
    });
  }

  const model = body.model || DEEPSEEK_V4_PRO_ID;
  // Модель передаётся КАК ЕСТЬ в NIM.
  // MODEL_MAPPING — только короткие алиасы для удобства.
  const nimModel = resolveModel(model);

  const clientWantsStream = body.stream === true;

  if (clientWantsStream) {
    // Открываем SSE-соединение немедленно.
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    // Keepalive пока ждём NIM.
    const keepalive = setInterval(() => {
      if (!res.writableEnded) res.write(makeKeepaliveChunk(model));
    }, KEEPALIVE_INTERVAL);

    let nimStream;

    try {
      nimStream = await nimFetchSameModel(nimModel, body);
    } catch (err) {
      clearInterval(keepalive);
      console.error('[Chat proxy error]', err.message);

      if (!res.writableEnded) {
        const errChunk = JSON.stringify({
          id: 'chatcmpl-error',
          object: 'chat.completion.chunk',
          model,
          choices: [{
            index: 0,
            delta: { content: `\n\n[Ошибка: ${err.message}]` },
            finish_reason: 'stop',
          }],
        });

        res.write(`data: ${errChunk}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }
      return;
    }

    clearInterval(keepalive);

    let buf = '';
    const rState = { started: false, closed: false };

    nimStream.on('data', chunk => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        if (line.includes('[DONE]')) {
          if (!res.writableEnded) res.write('data: [DONE]\n\n');
          continue;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            const reasoning = delta.reasoning_content || '';
            const content = delta.content || '';

            if (SHOW_REASONING) {
              let out = '';

              if (reasoning) {
                if (!rState.started) {
                  out += '<think>\n';
                  rState.started = true;
                }
                out += reasoning;
              }

              if (content && rState.started && !rState.closed) {
                out += '\n</think>\n\n';
                rState.closed = true;
              }

              if (content) out += content;
              delta.content = out;
            } else {
              delta.content = content;
            }

            delete delta.reasoning_content;
          }

          // Возвращаем клиенту тот model ID/alias, который он запросил.
          data.model = model;

          if (!res.writableEnded) {
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          }
        } catch (_) {
          if (!res.writableEnded) res.write(line + '\n\n');
        }
      }
    });

    nimStream.on('end', () => {
      if (!res.writableEnded) res.end();
    });

    nimStream.on('error', err => {
      console.error('[Chat stream error]', err.message);
      if (!res.writableEnded) res.end();
    });

  } else {
    // NON-STREAM:
    // открываем HTTP-ответ сразу и шлём пробелы, пока ждём NIM.
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Transfer-Encoding', 'chunked');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const keepalive = setInterval(() => {
      if (!res.writableEnded) res.write(' ');
    }, KEEPALIVE_INTERVAL);

    let nimStream;

    try {
      nimStream = await nimFetchSameModel(nimModel, body);
    } catch (err) {
      clearInterval(keepalive);
      console.error('[Chat proxy error]', err.message);

      if (!res.writableEnded) {
        res.end(JSON.stringify({
          error: { message: err.message, code: err.status || 500 },
        }));
      }
      return;
    }

    clearInterval(keepalive);

    let collected;

    try {
      collected = await collectStream(nimStream);
    } catch (err) {
      console.error('[Chat collect error]', err.message);

      if (!res.writableEnded) {
        res.end(JSON.stringify({
          error: { message: err.message, code: 500 },
        }));
      }
      return;
    }

    const {
      fullContent,
      fullReasoning,
      lastChoice,
      usage,
    } = collected;

    if (!res.writableEnded) {
      res.end(JSON.stringify({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: mergeContent(fullContent, fullReasoning),
          },
          finish_reason: lastChoice?.finish_reason || 'stop',
        }],
        usage: usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
        },
      }));
    }
  }
});

// ─────────────────────────────────────────────
// ✍️ TEXT COMPLETIONS
// ─────────────────────────────────────────────

app.post('/v1/completions', async (req, res) => {
  const body = req.body || {};

  console.log(
    '[Completion request] model:', body.model,
    '| prompt:',
    Array.isArray(body.prompt)
      ? `array[${body.prompt.length}]`
      : typeof body.prompt,
    '| stream:', body.stream
  );

  if (!NIM_API_KEY) {
    return res.status(500).json({
      error: { message: 'NIM_API_KEY не задан', code: 500 },
    });
  }

  const validPrompt =
    typeof body.prompt === 'string' ||
    (Array.isArray(body.prompt) && body.prompt.length > 0);

  if (!validPrompt) {
    return res.status(400).json({
      error: {
        message: 'prompt должен быть строкой или непустым массивом',
        code: 400,
      },
    });
  }

  const model = body.model || DEEPSEEK_V4_PRO_ID;
  const nimModel = resolveModel(model);
  const clientWantsStream = body.stream === true;

  if (clientWantsStream) {
    // Отправляем SSE headers немедленно — важно для Render и долгого TTFT.
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    // Пока NVIDIA ещё не прислала первый байт, отправляем валидные
    // пустые OpenAI Text Completion chunks.
    const keepalive = setInterval(() => {
      if (!res.writableEnded) {
        res.write(makeCompletionKeepaliveChunk(model));
      }
    }, KEEPALIVE_INTERVAL);

    let nimStream;

    try {
      nimStream = await nimCompletionFetchSameModel(nimModel, body);
    } catch (err) {
      clearInterval(keepalive);
      console.error('[Completion proxy error]', err.message);

      if (!res.writableEnded) {
        const errChunk = {
          id: 'cmpl-error',
          object: 'text_completion',
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [{
            text: `\n\n[Ошибка: ${err.message}]`,
            index: 0,
            logprobs: null,
            finish_reason: 'stop',
          }],
        };

        res.write(`data: ${JSON.stringify(errChunk)}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }
      return;
    }

    clearInterval(keepalive);

    let buf = '';

    nimStream.on('data', chunk => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        if (line.includes('[DONE]')) {
          if (!res.writableEnded) {
            res.write('data: [DONE]\n\n');
          }
          continue;
        }

        try {
          const data = JSON.parse(line.slice(6));

          // Сохраняем внешний alias/model, чтобы клиент видел тот ID,
          // который он сам указал в запросе.
          data.model = model;

          if (!res.writableEnded) {
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          }
        } catch (_) {
          // Если NIM вернул нестандартную SSE-строку, не теряем её.
          if (!res.writableEnded) res.write(line + '\n\n');
        }
      }
    });

    nimStream.on('end', () => {
      if (buf.trim() && !res.writableEnded) {
        // На случай, если последний SSE event пришёл без завершающего \n.
        const line = buf.trimEnd();

        if (line.startsWith('data: ')) {
          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
          } else {
            try {
              const data = JSON.parse(line.slice(6));
              data.model = model;
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (_) {
              res.write(line + '\n\n');
            }
          }
        }
      }

      if (!res.writableEnded) res.end();
    });

    nimStream.on('error', err => {
      console.error('[Completion stream error]', err.message);
      if (!res.writableEnded) res.end();
    });

  } else {
    // Клиент хочет обычный JSON. Внутри всё равно используем stream=true
    // к NVIDIA, собираем text chunks и отдаём один OpenAI-compatible response.
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Transfer-Encoding', 'chunked');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const keepalive = setInterval(() => {
      if (!res.writableEnded) res.write(' ');
    }, KEEPALIVE_INTERVAL);

    let nimStream;

    try {
      nimStream = await nimCompletionFetchSameModel(nimModel, body);
    } catch (err) {
      clearInterval(keepalive);
      console.error('[Completion proxy error]', err.message);

      if (!res.writableEnded) {
        res.end(JSON.stringify({
          error: { message: err.message, code: err.status || 500 },
        }));
      }
      return;
    }

    clearInterval(keepalive);

    let collected;

    try {
      collected = await collectCompletionStream(nimStream);
    } catch (err) {
      console.error('[Completion collect error]', err.message);

      if (!res.writableEnded) {
        res.end(JSON.stringify({
          error: { message: err.message, code: 500 },
        }));
      }
      return;
    }

    const choices = collected.choices.length
      ? collected.choices
      : [{
          index: 0,
          text: '',
          logprobs: null,
          finish_reason: 'stop',
        }];

    if (!res.writableEnded) {
      res.end(JSON.stringify({
        id: `cmpl-${Date.now()}`,
        object: 'text_completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices,
        usage: collected.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
        },
      }));
    }
  }
});

// ─────────────────────────────────────────────
// 404
// ─────────────────────────────────────────────

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `${req.path} not found`, code: 404 },
  });
});

// ─────────────────────────────────────────────
// START
// ─────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy`);
  console.log(`   Port:              ${PORT}`);
  console.log(`   DeepSeek Pro ID:   ${DEEPSEEK_V4_PRO_ID}`);
  console.log(`   DeepSeek Flash ID: ${DEEPSEEK_V4_FLASH_ID}`);
  console.log(`   DeepSeek mode:     ${DEEPSEEK_REASONING_EFFORT}`);
  console.log(`   First-token wait:  ${FIRST_TOKEN_TIMEOUT / 1000}s`);
  console.log(`   Idle timeout:      ${IDLE_TIMEOUT / 1000}s`);
  console.log(`   Keepalive every:   ${KEEPALIVE_INTERVAL / 1000}s`);
  console.log(`   RP style prompt:   ОТКЛЮЧЁН (промпты через SillyTavern)`);
  console.log(`   Strategy:          SSE открывается немедленно, keepalive до ответа NIM`);
  console.log(`   Chat endpoint:     /v1/chat/completions`);
  console.log(`   Text endpoint:     /v1/completions`);
  console.log(`   Models endpoint:   /v1/models`);
  console.log(`   Keep-alive ping:   /ping\n`);
});
