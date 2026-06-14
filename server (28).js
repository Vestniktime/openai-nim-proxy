// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const axios   = require('axios');
const http    = require('http');
const https   = require('https');

const app  = express();
const PORT = process.env.PORT || 3000;

// ─────────────────────────────────────────────
// CORS
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

// ✅ Лог КАЖДОГО запроса ДО парсинга тела
app.use((req, res, next) => {
  console.log(`[Incoming] ${req.method} ${req.path} | ct: ${req.headers['content-type']} | auth: ${req.headers['authorization'] ? 'yes' : 'no'}`);
  next();
});

// ✅ ИСПРАВЛЕНИЕ парсера:
// express.json с type:'*/*' принимает JSON при ЛЮБОМ Content-Type
// без зависания на req.on('end')
app.use(express.json({ type: '*/*', limit: '100mb' }));
app.use(express.text({ type: '*/*', limit: '100mb' }));
app.use((req, res, next) => {
  if (typeof req.body === 'string') {
    try { req.body = JSON.parse(req.body); } catch (_) {}
  }
  next();
});

// ─────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

const SHOW_REASONING  = false;
const EFFORT_STREAM   = process.env.DEEPSEEK_REASONING_EFFORT          || 'max';
const EFFORT_NOSTREAM = process.env.DEEPSEEK_REASONING_EFFORT_NOSTREAM || 'low';

const FIRST_TOKEN_TIMEOUT = parseInt(process.env.FIRST_TOKEN_TIMEOUT || '120000');
const IDLE_TIMEOUT        = parseInt(process.env.IDLE_TIMEOUT        || '60000');

const httpAgent  = new http.Agent ({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

// ─────────────────────────────────────────────
// МОДЕЛИ
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
// HELPERS
// ─────────────────────────────────────────────

function buildThinkingParams(nimModel, effort) {
  if (DEEPSEEK_V4_MODELS.has(nimModel)) {
    return {
      chat_template_kwargs: { enable_thinking: true, thinking: true },
      reasoning_effort: effort,
    };
  }
  if (OPTIONAL_THINKING_MODELS.has(nimModel)) {
    return { chat_template_kwargs: { thinking: true } };
  }
  return {};
}

function mergeContent(content, reasoning) {
  if (SHOW_REASONING && reasoning) return `<think>\n${reasoning}\n</think>\n\n${content || ''}`;
  return content || '';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function withSmartTimeout(stream, label) {
  let first = false;
  let timer = setTimeout(() => stream.destroy(new Error('first-token timeout')), FIRST_TOKEN_TIMEOUT);
  stream.on('data', () => {
    clearTimeout(timer);
    if (!first) { first = true; console.log(`[${label}] ✓ первый токен`); }
    timer = setTimeout(() => stream.destroy(new Error('idle timeout')), IDLE_TIMEOUT);
  });
  stream.on('end',   () => clearTimeout(timer));
  stream.on('error', () => clearTimeout(timer));
}

async function callNIM(nimModel, body, effort) {
  const isDeepSeek = DEEPSEEK_V4_MODELS.has(nimModel);
  const payload = {
    model:       nimModel,
    messages:    body.messages,
    temperature: body.temperature ?? (isDeepSeek ? 0.6 : 0.7),
    max_tokens:  body.max_tokens  || (isDeepSeek ? 8192 : 4096),
    stream:      true,
    ...buildThinkingParams(nimModel, effort),
  };

  const response = await axios.post(`${NIM_API_BASE}/chat/completions`, payload, {
    headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
    responseType: 'stream',
    timeout: 0,
    httpAgent, httpsAgent,
    validateStatus: null,
  });

  if (response.status >= 400) {
    const raw = await new Promise(resolve => {
      let s = '';
      response.data.on('data', c => { s += c.toString(); });
      response.data.on('end',  () => resolve(s));
      response.data.on('error',() => resolve(''));
    });
    const err  = new Error(`NIM ${response.status}: ${raw}`);
    err.status = response.status;
    throw err;
  }

  withSmartTimeout(response.data, nimModel);
  return response.data;
}

async function callNIMWithFallback(nimModel, body, effort) {
  const chain = [nimModel, ...(FALLBACK_CHAIN[nimModel] || [])];
  let lastError;
  for (const model of chain) {
    try {
      console.log(`[NIM] ${model} | effort=${effort}`);
      return { stream: await callNIM(model, body, effort), usedModel: model };
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

// ─────────────────────────────────────────────
// ENDPOINTS
// ─────────────────────────────────────────────

app.get('/ping',   (req, res) => res.send('pong'));
app.get('/health', (req, res) => res.json({
  status: 'ok', nim_key_set: !!NIM_API_KEY,
  effort_stream: EFFORT_STREAM, effort_nostream: EFFORT_NOSTREAM,
  available_models: Object.keys(MODEL_MAPPING),
}));
app.get('/v1/models', (req, res) => res.json({
  object: 'list',
  data: Object.keys(MODEL_MAPPING).map(id => ({
    id, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'nvidia-nim-proxy',
  })),
}));

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const body = req.body || {};
    const clientWantsStream = body.stream === true;
    const effort = clientWantsStream ? EFFORT_STREAM : EFFORT_NOSTREAM;

    console.log('[Request] model:', body.model,
      '| messages:', Array.isArray(body.messages) ? body.messages.length : typeof body.messages,
      '| stream:', clientWantsStream, '| effort:', effort);

    if (!NIM_API_KEY)
      return res.status(500).json({ error: { message: 'NIM_API_KEY не задан', code: 500 } });
    if (!Array.isArray(body.messages) || body.messages.length === 0)
      return res.status(400).json({ error: { message: 'messages пуст', code: 400 } });

    let nimModel = MODEL_MAPPING[body.model] || (() => {
      const m = (body.model || '').toLowerCase();
      if (m.includes('deepseek-v4'))                return 'deepseek-ai/deepseek-v4-pro';
      if (m.includes('gpt-4') || m.includes('405b')) return 'meta/llama-3.1-405b-instruct';
      if (m.includes('claude') || m.includes('70b')) return 'meta/llama-3.1-70b-instruct';
      return 'meta/llama-3.1-8b-instruct';
    })();

    const { stream: nimStream } = await callNIMWithFallback(nimModel, body, effort);

    if (clientWantsStream) {
      res.setHeader('Content-Type',      'text/event-stream');
      res.setHeader('Cache-Control',     'no-cache');
      res.setHeader('Connection',        'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders();

      const keepalive = setInterval(() => {
        if (!res.writableEnded) res.write(': keepalive\n\n');
      }, 15_000);

      let buf = '';
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
            if (delta) {
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
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) { res.write(line + '\n'); }
        }
      });
      nimStream.on('end',   () => { clearInterval(keepalive); if (!res.writableEnded) res.end(); });
      nimStream.on('error', err => {
        clearInterval(keepalive);
        console.error('[Stream error]', err.message);
        if (!res.writableEnded) res.end();
      });

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

      res.setHeader('Content-Type', 'application/json');
      res.json({
        id: `chatcmpl-${Date.now()}`, object: 'chat.completion',
        created: Math.floor(Date.now() / 1000), model: body.model,
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
  console.log(`   Port:              ${PORT}`);
  console.log(`   NIM key set:       ${!!NIM_API_KEY}`);
  console.log(`   stream:true  →     SSE  + effort:${EFFORT_STREAM}`);
  console.log(`   stream:false →     JSON + effort:${EFFORT_NOSTREAM}\n`);
});
