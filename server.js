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
// Sophia's LoreBary и некоторые другие клиенты шлют JSON
// с Content-Type: text/plain или вовсе без заголовка.
// express.json() в таком случае молча пропускает тело.
// Этот middleware читает сырые байты и парсит JSON вручную.
// ─────────────────────────────────────────────
app.use((req, res, next) => {
  // GET / OPTIONS / HEAD — тела нет, пропускаем
  if (req.method === 'GET' || req.method === 'OPTIONS' || req.method === 'HEAD') {
    return next();
  }

  let raw = '';
  req.setEncoding('utf8');
  req.on('data', chunk => { raw += chunk; });
  req.on('end', () => {
    if (!raw) return next();
    try {
      req.body = JSON.parse(raw);
    } catch (e) {
      // Не JSON — оставляем как строку, роуты сами разберутся
      req.body = raw;
    }
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
const FIRST_TOKEN_TIMEOUT       = parseInt(process.env.FIRST_TOKEN_TIMEOUT || '120000');
const IDLE_TIMEOUT              = parseInt(process.env.IDLE_TIMEOUT        || '60000');

const httpAgent  = new http.Agent ({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

// ─────────────────────────────────────────────
// 📋 МОДЕЛИ
// ─────────────────────────────────────────────

const DEEPSEEK_V4_MODELS = new Set([
  'deepseek-ai/deepseek-v4-pro',
  'deepseek-ai/deepseek-v4-flash',
  'nvidia/nemotron-3-ultra-550b-a55b',
  'thinkingmachines/inkling',
]);

const OPTIONAL_THINKING_MODELS = new Set([
  'nvidia/nemotron-3-ultra-550b-a55b',
  'qwen/qwen3-next-80b-a3b-thinking',
  'qwen/qwen3-coder-480b-a35b-instruct',
]);

const FALLBACK_CHAIN = {
  'deepseek-ai/deepseek-v4-pro':   ['deepseek-ai/deepseek-v4-flash', 'nvidia/nemotron-3-ultra-550b-a55b'],
  'deepseek-ai/deepseek-v4-flash': ['deepseek-ai/deepseek-v4-pro',   'meta/llama-3.1-70b-instruct'],
};

const MODEL_MAPPING = {
  'gpt-4o':            'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-pro':   'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'nemotron-3-ultra-550b-a55b':     'nvidia/nemotron-3-ultra-550b-a55b',
  'thinkingmachines/inkling':             'thinkingmachines/inkling',
  'gpt-4-turbo':       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-opus':     'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet':   'meta/llama-3.1-70b-instruct',
  'gemini-pro':        'mistralai/mistral-large-2-instruct',
};

// ─────────────────────────────────────────────
// 🛠️ HELPERS
// ─────────────────────────────────────────────

function buildThinkingParams(nimModel) {
  if (DEEPSEEK_V4_MODELS.has(nimModel)) {
    return {
      chat_template_kwargs: { enable_thinking: true, thinking: true },
      reasoning_effort: DEEPSEEK_REASONING_EFFORT,
    };
  }
  if (OPTIONAL_THINKING_MODELS.has(nimModel)) {
    return { chat_template_kwargs: { thinking: true } };
  }
  return {};
}

function mergeContent(content, reasoningContent) {
  if (SHOW_REASONING && reasoningContent) {
    return `<think>\n${reasoningContent}\n</think>\n\n${content || ''}`;
  }
  return content || '';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function withSmartTimeout(stream, label) {
  let firstTokenReceived = false;
  let timer = setTimeout(() => {
    console.error(`[First-token timeout] ${label}`);
    stream.destroy(new Error('first-token timeout'));
  }, FIRST_TOKEN_TIMEOUT);

  stream.on('data', () => {
    clearTimeout(timer);
    if (!firstTokenReceived) {
      firstTokenReceived = true;
      console.log(`[${label}] ✓ Первый токен`);
    }
    timer = setTimeout(() => {
      console.error(`[Idle timeout] ${label}`);
      stream.destroy(new Error('idle timeout'));
    }, IDLE_TIMEOUT);
  });
  stream.on('end',   () => clearTimeout(timer));
  stream.on('error', () => clearTimeout(timer));
}

async function nimRequest(nimModel, body) {
  const thinkingParams = buildThinkingParams(nimModel);
  const isDeepSeek     = DEEPSEEK_V4_MODELS.has(nimModel);

  const payload = {
    model:       nimModel,
    messages:    body.messages,
    temperature: body.temperature ?? (isDeepSeek ? 1.0 : 1.0),
    max_tokens:  body.max_tokens  || (isDeepSeek ? 16384 : 16384),
    stream:      true,
    ...thinkingParams,
  };

  const response = await axios.post(
    `${NIM_API_BASE}/chat/completions`, payload,
    {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: 'stream',
      timeout: 0,
      httpAgent, httpsAgent,
      validateStatus: null,
    }
  );

  if (response.status >= 400) {
    const raw = await new Promise(resolve => {
      let s = '';
      response.data.on('data', c => { s += c.toString(); });
      response.data.on('end',  () => resolve(s));
      response.data.on('error',() => resolve(''));
    });
    let detail = raw;
    try { detail = JSON.parse(raw)?.detail || JSON.parse(raw)?.message || raw; } catch {}
    const err  = new Error(`NIM ${response.status}: ${raw}`);
    err.status = response.status;
    throw err;
  }

  withSmartTimeout(response.data, nimModel);
  return response.data;
}

async function nimRequestWithFallback(preferredModel, body) {
  const chain = [preferredModel, ...(FALLBACK_CHAIN[preferredModel] || [])];
  let lastError;
  for (const model of chain) {
    try {
      console.log(`[NIM] Попытка: ${model}`);
      const stream = await nimRequest(model, body);
      return { stream, usedModel: model };
    } catch (err) {
      lastError = err;
      const status = err.status || 0;
      if (RETRYABLE_STATUSES.has(status) || err.message.includes('timeout')) {
        console.warn(`[NIM] ${model} → ${status || 'timeout'}, фолбэк…`);
        if (status === 429) await sleep(2000);
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

function processDelta(delta, rState) {
  const reasoning = delta.reasoning_content || '';
  const content   = delta.content || '';
  if (SHOW_REASONING) {
    let out = '';
    if (reasoning) { if (!rState.started) { out += '<think>\n'; rState.started = true; } out += reasoning; }
    if (content && rState.started && !rState.closed) { out += '\n</think>\n\n'; rState.closed = true; }
    if (content) out += content;
    delta.content = out;
  } else {
    delta.content = content;
  }
  delete delta.reasoning_content;
}

// ─────────────────────────────────────────────
// 🌐 ENDPOINTS
// ─────────────────────────────────────────────

app.get('/ping',   (req, res) => res.send('pong'));

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    nim_key_set: !!NIM_API_KEY,
    deepseek_reasoning_effort: DEEPSEEK_REASONING_EFFORT,
    first_token_timeout_ms: FIRST_TOKEN_TIMEOUT,
    idle_timeout_ms: IDLE_TIMEOUT,
    available_models: Object.keys(MODEL_MAPPING),
    fallback_chains: FALLBACK_CHAIN,
  });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id, object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim-proxy',
    })),
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const body = req.body;

    // ✅ Диагностический лог — видно что именно пришло от клиента
    console.log('[Request] model:', body?.model,
      '| messages:', Array.isArray(body?.messages) ? body.messages.length : typeof body?.messages,
      '| stream:', body?.stream,
      '| content-type:', req.headers['content-type']);

    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: { message: 'NIM_API_KEY не задан в переменных окружения Render', code: 500 }
      });
    }

    // Защита от пустого messages
    if (!body?.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
      return res.status(400).json({
        error: { message: 'messages обязателен и не должен быть пустым', code: 400 }
      });
    }

    const model = body.model || 'gpt-4o';
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const m = model.toLowerCase();
      if      (m.includes('deepseek-v4'))               nimModel = 'deepseek-ai/deepseek-v4-pro';
      else if (m.includes('nvidia/nemotron-3-ultra-550b-a55b') || m.includes('405b')) nimModel = 'nvidia/nemotron-3-ultra-550b-a55b';
      else if (m.includes('thinkingmachines/inkling') || m.includes('thinkingmachines/inkling')) nimModel = 'thinkingmachines/inkling';
      else                                                nimModel = 'thinkingmachines/inkling';
    }

    const { stream: nimStream, usedModel } = await nimRequestWithFallback(nimModel, body);
    const clientWantsStream = body.stream === true;

    // ── STREAM ──
    if (clientWantsStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const keepalive = setInterval(() => {
        if (!res.writableEnded) res.write(': keepalive\n\n');
      }, 20_000);

      let buf = '';
      const rState = { started: false, closed: false };

      nimStream.on('data', (chunk) => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) { res.write('data: [DONE]\n\n'); continue; }
          try {
            const data  = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (delta) processDelta(delta, rState);
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) { res.write(line + '\n'); }
        }
      });
      nimStream.on('end',   () => { clearInterval(keepalive); if (!res.writableEnded) res.end(); });
      nimStream.on('error', (err) => {
        clearInterval(keepalive);
        console.error('[Stream error]', err.message);
        if (!res.writableEnded) res.end();
      });

    // ── JSON ──
    } else {
      let buf = '', fullContent = '', fullReasoning = '', lastChoice = null, usage = null;

      await new Promise((resolve, reject) => {
        nimStream.on('data', (chunk) => {
          buf += chunk.toString();
          const lines = buf.split('\n');
          buf = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data: ') || line.includes('[DONE]')) continue;
            try {
              const data = JSON.parse(line.slice(6));
              if (data.usage) usage = data.usage;
              const choice = data.choices?.[0];
              if (!choice) continue;
              lastChoice     = choice;
              fullContent   += choice.delta?.content           || '';
              fullReasoning += choice.delta?.reasoning_content || '';
            } catch (_) {}
          }
        });
        nimStream.on('end',   resolve);
        nimStream.on('error', reject);
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: mergeContent(fullContent, fullReasoning) }, finish_reason: lastChoice?.finish_reason || 'stop' }],
        usage: usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (error) {
    console.error('[Proxy error]', error.message);
    if (!res.headersSent) {
      res.status(error.status || 500).json({
        error: { message: error.message || 'Internal server error', code: error.status || 500 },
      });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `${req.path} not found`, code: 404 } });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy`);
  console.log(`   Port:             ${PORT}`);
  console.log(`   NIM key set:      ${!!NIM_API_KEY}`);
  console.log(`   DeepSeek effort:  ${DEEPSEEK_REASONING_EFFORT}`);
  console.log(`   Keep-alive URL:   /ping\n`);
});
