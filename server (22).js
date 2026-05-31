// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors    = require('cors');
const axios   = require('axios');
const http    = require('http');
const https   = require('https');

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ─────────────────────────────────────────────
// ⚙️ CONFIG
// ─────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

const SHOW_REASONING            = false;
const DEEPSEEK_REASONING_EFFORT = process.env.DEEPSEEK_REASONING_EFFORT || 'low';

// NIM free tier: 40 RPM = 1 запрос каждые 1500 мс.
// MIN_REQUEST_INTERVAL — минимальная пауза между запросами к NIM.
const RPM_LIMIT           = parseInt(process.env.RPM_LIMIT || '35');       // чуть ниже лимита для запаса
const MIN_REQUEST_INTERVAL = Math.ceil(60000 / RPM_LIMIT);                 // ~1714 мс

// Backoff при 429: начальная задержка, множитель, максимум, кол-во попыток
const BACKOFF_BASE    = parseInt(process.env.BACKOFF_BASE    || '3000');   // 3 сек
const BACKOFF_MAX     = parseInt(process.env.BACKOFF_MAX     || '30000');  // 30 сек
const BACKOFF_RETRIES = parseInt(process.env.BACKOFF_RETRIES || '4');

// Ожидание первого токена и idle между чанками
const FIRST_TOKEN_TIMEOUT = parseInt(process.env.FIRST_TOKEN_TIMEOUT || '120000');
const IDLE_TIMEOUT        = parseInt(process.env.IDLE_TIMEOUT        || '60000');

const httpAgent  = new http.Agent ({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

// ─────────────────────────────────────────────
// 🚦 RATE LIMITER (очередь с минимальным интервалом)
// ─────────────────────────────────────────────

class RateLimiter {
  constructor(minIntervalMs) {
    this.minInterval = minIntervalMs;
    this.lastRequestAt = 0;
    this.queue = [];
    this.running = false;
  }

  // Добавить задачу в очередь и дождаться её выполнения
  schedule(fn) {
    return new Promise((resolve, reject) => {
      this.queue.push({ fn, resolve, reject });
      if (!this.running) this._run();
    });
  }

  async _run() {
    this.running = true;
    while (this.queue.length > 0) {
      const now  = Date.now();
      const wait = this.minInterval - (now - this.lastRequestAt);
      if (wait > 0) await sleep(wait);

      const { fn, resolve, reject } = this.queue.shift();
      this.lastRequestAt = Date.now();
      try { resolve(await fn()); } catch (err) { reject(err); }
    }
    this.running = false;
  }
}

const rateLimiter = new RateLimiter(MIN_REQUEST_INTERVAL);

// ─────────────────────────────────────────────
// 📋 МОДЕЛИ И ФОЛБЭКИ
// ─────────────────────────────────────────────

const DEEPSEEK_V4_MODELS = new Set([
  'deepseek-ai/deepseek-v4-pro',
  'deepseek-ai/deepseek-v4-flash',
]);

const OPTIONAL_THINKING_MODELS = new Set([
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-next-80b-a3b-thinking',
  'qwen/qwen3-coder-480b-a35b-instruct',
]);

// При 429/502/503/504 пробуем следующую модель в цепочке
const FALLBACK_CHAIN = {
  'deepseek-ai/deepseek-v4-pro':   ['deepseek-ai/deepseek-v4-flash', 'meta/llama-3.1-70b-instruct'],
  'deepseek-ai/deepseek-v4-flash': ['deepseek-ai/deepseek-v4-pro',   'meta/llama-3.1-70b-instruct'],
};

const MODEL_MAPPING = {
  'gpt-4o':            'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-pro':   'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'gpt-3.5-turbo':     'meta/llama-3.1-8b-instruct',
  'gpt-4':             'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo':       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-opus':     'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet':   'meta/llama-3.1-70b-instruct',
  'gemini-pro':        'mistralai/mistral-large-2-instruct',
};

// ─────────────────────────────────────────────
// 🛠️ HELPERS
// ─────────────────────────────────────────────

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

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

// ─────────────────────────────────────────────
// 🌐 NIM REQUEST (один запрос через rate limiter)
// ─────────────────────────────────────────────

async function nimSingleRequest(nimModel, body) {
  const thinkingParams = buildThinkingParams(nimModel);
  const isDeepSeek     = DEEPSEEK_V4_MODELS.has(nimModel);

  const payload = {
    model: nimModel,
    messages: body.messages,
    temperature: body.temperature ?? (isDeepSeek ? 0.6 : 0.7),
    max_tokens:  body.max_tokens  || (isDeepSeek ? 8192 : 4096),
    stream: true,
    ...thinkingParams,
  };

  // Проходим через очередь — гарантируем MIN_REQUEST_INTERVAL между запросами
  const response = await rateLimiter.schedule(() =>
    axios.post(`${NIM_API_BASE}/chat/completions`, payload, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
      },
      responseType: 'stream',
      timeout: 0,
      httpAgent,
      httpsAgent,
      validateStatus: null,
    })
  );

  if (response.status >= 400) {
    const raw = await new Promise(resolve => {
      let s = '';
      response.data.on('data',  c => { s += c.toString(); });
      response.data.on('end',   () => resolve(s));
      response.data.on('error', () => resolve(''));
    });
    let detail = raw;
    try {
      const parsed = JSON.parse(raw);
      detail = parsed?.message || parsed?.detail || raw;
    } catch {}
    const err  = new Error(`NIM ${response.status}: ${detail}`);
    err.status = response.status;
    throw err;
  }

  withSmartTimeout(response.data, nimModel);
  return response.data;
}

// ─────────────────────────────────────────────
// 🔄 FALLBACK + BACKOFF
// ─────────────────────────────────────────────

async function nimRequestWithFallback(preferredModel, body) {
  const chain = [preferredModel, ...(FALLBACK_CHAIN[preferredModel] || [])];

  for (const model of chain) {
    let delay = BACKOFF_BASE;

    for (let attempt = 0; attempt <= BACKOFF_RETRIES; attempt++) {
      try {
        console.log(`[NIM] ${model} | попытка ${attempt + 1}`);
        const stream = await nimSingleRequest(model, body);
        return { stream, usedModel: model };
      } catch (err) {
        const status = err.status || 0;
        const isRetryable = status === 429 || status >= 500 || err.message.includes('timeout');

        if (!isRetryable) throw err; // 400, 401, 403 — не повторяем

        if (attempt < BACKOFF_RETRIES) {
          // Exponential backoff с jitter ±20%
          const jitter = delay * 0.2 * (Math.random() * 2 - 1);
          const wait   = Math.min(delay + jitter, BACKOFF_MAX);
          console.warn(`[NIM] ${model} → ${status || 'timeout'}, ждём ${Math.round(wait)}мс…`);
          await sleep(wait);
          delay = Math.min(delay * 2, BACKOFF_MAX);
        } else {
          // Исчерпали попытки для этой модели — переходим к следующей
          console.warn(`[NIM] ${model} — все попытки исчерпаны, переходим к фолбэку`);
        }
      }
    }
  }

  throw new Error('Все модели в цепочке фолбэков недоступны');
}

// ─────────────────────────────────────────────
// SSE delta processing
// ─────────────────────────────────────────────

function processDelta(delta, rState) {
  const reasoning = delta.reasoning_content || '';
  const content   = delta.content || '';

  if (SHOW_REASONING) {
    let out = '';
    if (reasoning) {
      if (!rState.started) { out += '<think>\n'; rState.started = true; }
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

// ─────────────────────────────────────────────
// 🌐 ENDPOINTS
// ─────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    rpm_limit: RPM_LIMIT,
    min_request_interval_ms: MIN_REQUEST_INTERVAL,
    queue_length: rateLimiter.queue.length,
    deepseek_reasoning_effort: DEEPSEEK_REASONING_EFFORT,
    backoff_retries: BACKOFF_RETRIES,
    available_models: Object.keys(MODEL_MAPPING),
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

// ─────────────────────────────────────────────
// 💬 CHAT COMPLETIONS
// ─────────────────────────────────────────────

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, stream: clientWantsStream } = req.body;

    // Резолв модели
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const m = model.toLowerCase();
      if      (m.includes('deepseek-v4'))                            nimModel = 'deepseek-ai/deepseek-v4-pro';
      else if (m.includes('gpt-4') || m.includes('405b'))            nimModel = 'meta/llama-3.1-405b-instruct';
      else if (m.includes('claude') || m.includes('70b'))            nimModel = 'meta/llama-3.1-70b-instruct';
      else                                                            nimModel = 'meta/llama-3.1-8b-instruct';
    }

    const queuePos = rateLimiter.queue.length;
    if (queuePos > 0) console.log(`[Queue] Позиция в очереди: ${queuePos}`);

    const { stream: nimStream, usedModel } = await nimRequestWithFallback(nimModel, req.body);

    if (usedModel !== nimModel) {
      console.log(`[Proxy] Фолбэк: ${nimModel} → ${usedModel}`);
    }

    // ─────────────────────────────────────────
    // CLIENT WANTS STREAM
    // ─────────────────────────────────────────
    if (clientWantsStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // Keepalive для Render (30s idle → disconnect)
      const keepalive = setInterval(() => {
        if (!res.writableEnded) res.write(': keepalive\n\n');
      }, 20_000);

      if (usedModel !== nimModel) {
        const notice = {
          id: 'chatcmpl-notice', object: 'chat.completion.chunk', model,
          choices: [{ index: 0, delta: { role: 'assistant', content: `[⚠️ Фолбэк: ${usedModel}]\n\n` }, finish_reason: null }],
        };
        res.write(`data: ${JSON.stringify(notice)}\n\n`);
      }

      let buf    = '';
      const rState = { started: false, closed: false };

      nimStream.on('data', chunk => {
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

      nimStream.on('end', () => {
        clearInterval(keepalive);
        if (!res.writableEnded) res.end();
      });
      nimStream.on('error', err => {
        clearInterval(keepalive);
        console.error('[Stream error]', err.message);
        if (!res.writableEnded) res.end();
      });

    // ─────────────────────────────────────────
    // CLIENT WANTS JSON
    // ─────────────────────────────────────────
    } else {
      let buf = '', fullContent = '', fullReasoning = '', lastChoice = null, usage = null;

      await new Promise((resolve, reject) => {
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
        nimStream.on('end',   resolve);
        nimStream.on('error', reject);
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        'x-used-model': usedModel,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: mergeContent(fullContent, fullReasoning) },
          finish_reason: lastChoice?.finish_reason || 'stop',
        }],
        usage: usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (error) {
    console.error('[Proxy error]', error.message);
    if (!res.headersSent) {
      res.status(error.status || 500).json({
        error: {
          message: error.message || 'Internal server error',
          type: 'proxy_error',
          code: error.status || 500,
        },
      });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `${req.path} not found`, code: 404 } });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy`);
  console.log(`   Port:               ${PORT}`);
  console.log(`   Rate limit:         ${RPM_LIMIT} RPM (интервал ~${MIN_REQUEST_INTERVAL}мс)`);
  console.log(`   Backoff retries:    ${BACKOFF_RETRIES}x (${BACKOFF_BASE}мс → ${BACKOFF_MAX}мс)`);
  console.log(`   DeepSeek effort:    ${DEEPSEEK_REASONING_EFFORT}`);
  console.log(`   Fallback chain:     v4-pro → v4-flash → llama-70b`);
  console.log(`   Health:             http://localhost:${PORT}/health\n`);
});
