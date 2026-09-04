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

const SHOW_REASONING = false;
const DEEPSEEK_REASONING_EFFORT = process.env.DEEPSEEK_REASONING_EFFORT || 'high';

// ✅ Увеличено. Фолбэка на другие модели больше нет — если модель
//    задумалась, ей нужно дать реально дождаться ответа, а не спешить
//    сдаваться через 45 секунд.
const FIRST_TOKEN_TIMEOUT = parseInt(process.env.FIRST_TOKEN_TIMEOUT || '240000'); // 4 мин
const IDLE_TIMEOUT        = parseInt(process.env.IDLE_TIMEOUT        || '120000'); // 2 мин
const KEEPALIVE_INTERVAL  = parseInt(process.env.KEEPALIVE_INTERVAL  || '10000');

// Сколько раз пробовать ОДНУ И ТУ ЖЕ модель, прежде чем сдаться.
// Не переключается на другую модель — просто пробует ещё раз при сбое.
const MAX_RETRIES_SAME_MODEL = parseInt(process.env.MAX_RETRIES_SAME_MODEL || '2');

const ENABLE_RP_STYLE_PROMPT = process.env.ENABLE_RP_STYLE_PROMPT !== 'false';
const RP_STYLE_INSTRUCTION = process.env.RP_STYLE_INSTRUCTION ||
  'Write immersive, descriptive roleplay responses. Include vivid sensory detail, ' +
  'physical actions, body language, and internal thoughts alongside dialogue. ' +
  'Avoid short one- or two-line replies — write rich, multi-paragraph prose ' +
  '(roughly 150-400 words) that moves the scene forward while staying fully in character.';

const MIN_RESPONSE_TOKENS = parseInt(process.env.MIN_RESPONSE_TOKENS || '2048');

const httpAgent  = new http.Agent ({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

// ─────────────────────────────────────────────
// 📋 МОДЕЛИ
// ─────────────────────────────────────────────

const DEEPSEEK_V4_PRO_ID   = 'deepseek-ai/deepseek-v4-pro-0813';   // актуальный релиз Pro
const DEEPSEEK_V4_FLASH_ID = 'deepseek-ai/deepseek-v4-flash-0731'; // актуальный релиз Flash

function isDeepSeekV4(nimModel) {
  return nimModel.includes('deepseek-v4-pro') || nimModel.includes('deepseek-v4-flash');
}

const OPTIONAL_THINKING_MODELS = new Set([
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-next-80b-a3b-thinking',
  'qwen/qwen3-coder-480b-a35b-instruct',
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
    const effort = (DEEPSEEK_REASONING_EFFORT || 'high').toLowerCase();
    if (effort === 'max') {
      return { chat_template_kwargs: { enable_thinking: true, thinking: true, reasoning_effort: 'max' } };
    }
    if (effort === 'low' || effort === 'off' || effort === 'none') {
      return { chat_template_kwargs: { enable_thinking: false, thinking: false } };
    }
    return { chat_template_kwargs: { enable_thinking: true, thinking: true, reasoning_effort: 'high' } };
  }
  if (OPTIONAL_THINKING_MODELS.has(nimModel)) {
    return { chat_template_kwargs: { thinking: true } };
  }
  return {};
}

function applyStylePrompt(messages) {
  if (!ENABLE_RP_STYLE_PROMPT || !Array.isArray(messages) || messages.length === 0) {
    return messages;
  }
  const out = messages.map(m => ({ ...m }));
  if (out[0]?.role === 'system') {
    out[0] = { ...out[0], content: `${out[0].content}\n\n${RP_STYLE_INSTRUCTION}` };
  } else {
    out.unshift({ role: 'system', content: RP_STYLE_INSTRUCTION });
  }
  return out;
}

function mergeContent(content, reasoningContent) {
  if (SHOW_REASONING && reasoningContent) {
    return `<think>\n${reasoningContent}\n</think>\n\n${content || ''}`;
  }
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
    if (!first) { first = true; console.log(`[${label}] ✓ первый токен`); }
    timer = setTimeout(() => {
      console.error(`[Idle] ${label}: нет данных ${IDLE_TIMEOUT / 1000}с`);
      stream.destroy(new Error('idle timeout'));
    }, IDLE_TIMEOUT);
  });
  stream.on('end',   () => clearTimeout(timer));
  stream.on('error', () => clearTimeout(timer));
}

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
    const raw = await new Promise(r => {
      let s = '';
      response.data.on('data', c => { s += c.toString(); });
      response.data.on('end',  () => r(s));
      response.data.on('error',() => r(''));
    });
    const err  = new Error(`NIM ${response.status}: ${raw}`);
    err.status = response.status;
    throw err;
  }

  withSmartTimeout(response.data, nimModel);
  return response.data;
}

/**
 * ✅ Фолбэка на ДРУГИЕ модели больше нет. Пробуем ТОЛЬКО ту модель,
 * которую выбрал клиент, максимум MAX_RETRIES_SAME_MODEL раз подряд —
 * если она виснет/ошибается, ждём и пробуем её же снова, а не
 * подменяем на Flash или Llama.
 */
async function nimFetchSameModel(nimModel, body) {
  let lastError;
  for (let attempt = 1; attempt <= MAX_RETRIES_SAME_MODEL; attempt++) {
    const isLast = attempt === MAX_RETRIES_SAME_MODEL;
    try {
      console.log(`[NIM] Попытка ${attempt}/${MAX_RETRIES_SAME_MODEL}: ${nimModel}`);
      return { stream: await nimFetch(nimModel, body), usedModel: nimModel };
    } catch (err) {
      lastError = err;
      const reason = err.status ? `HTTP ${err.status}` : err.message;
      if (isLast) {
        console.error(`[NIM] ${nimModel} → ${reason} (последняя попытка, сдаёмся)`);
        throw err;
      }
      console.warn(`[NIM] ${nimModel} → ${reason}, пробуем ту же модель ещё раз…`);
      await sleep(1500);
    }
  }
  throw lastError;
}

function collectStream(nimStream) {
  return new Promise((resolve, reject) => {
    let buf = '', fullContent = '', fullReasoning = '', lastChoice = null, usage = null;
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
    nimStream.on('end',   () => resolve({ fullContent, fullReasoning, lastChoice, usage }));
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
  models: {
    deepseek_pro:   DEEPSEEK_V4_PRO_ID,
    deepseek_flash: DEEPSEEK_V4_FLASH_ID,
  },
  deepseek_mode: DEEPSEEK_REASONING_EFFORT,
  fallback_policy: 'НЕТ переключения на другие модели — только повтор той же модели',
  max_retries_same_model: MAX_RETRIES_SAME_MODEL,
  rp_style_prompt_enabled: ENABLE_RP_STYLE_PROMPT,
  min_response_tokens: MIN_RESPONSE_TOKENS,
  first_token_timeout_ms: FIRST_TOKEN_TIMEOUT,
  idle_timeout_ms: IDLE_TIMEOUT,
  available_models: Object.keys(MODEL_MAPPING),
}));

app.get('/v1/models', (_, res) => res.json({
  object: 'list',
  data: Object.keys(MODEL_MAPPING).map(id => ({
    id, object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy',
  })),
}));

// ─────────────────────────────────────────────
// 💬 CHAT COMPLETIONS
// ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const body = req.body || {};

    console.log('[Request] model:', body.model,
      '| messages:', Array.isArray(body.messages) ? body.messages.length : typeof body.messages,
      '| stream:', body.stream);

    if (!NIM_API_KEY) {
      return res.status(500).json({ error: { message: 'NIM_API_KEY не задан в Render → Environment', code: 500 } });
    }
    if (!Array.isArray(body.messages) || body.messages.length === 0) {
      return res.status(400).json({ error: { message: 'messages обязателен и не должен быть пустым', code: 400 } });
    }

    const model = body.model || 'gpt-4o';
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      if (model.includes('/')) {
        // Уже похоже на настоящий "provider/model" ID — передаём как есть
        nimModel = model;
      } else {
        const m = model.toLowerCase();
        if (m.includes('deepseek')) {
          nimModel = m.includes('flash') ? DEEPSEEK_V4_FLASH_ID : DEEPSEEK_V4_PRO_ID;
        } else if (m.includes('gpt-4') || m.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (m.includes('claude') || m.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    const clientWantsStream = body.stream === true;
    const { stream: nimStream, usedModel } = await nimFetchSameModel(nimModel, body);

    // ── STREAM ──
    if (clientWantsStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const keepaliveChunk = JSON.stringify({
        id: 'chatcmpl-keepalive',
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: body.model,
        choices: [{ index: 0, delta: {}, finish_reason: null }],
      });
      const keepalive = setInterval(() => {
        if (!res.writableEnded) res.write(`data: ${keepaliveChunk}\n\n`);
      }, KEEPALIVE_INTERVAL);

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

    // ── JSON ──
    } else {
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Transfer-Encoding', 'chunked');
      res.setHeader('Connection', 'keep-alive');

      const keepalive = setInterval(() => { if (!res.writableEnded) res.write(' '); }, KEEPALIVE_INTERVAL);

      let collected;
      try { collected = await collectStream(nimStream); }
      finally { clearInterval(keepalive); }

      const { fullContent, fullReasoning, lastChoice, usage } = collected;

      res.end(JSON.stringify({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: mergeContent(fullContent, fullReasoning) },
          finish_reason: lastChoice?.finish_reason || 'stop',
        }],
        usage: usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      }));
    }

  } catch (error) {
    console.error('[Proxy error]', error.message);
    if (!res.headersSent) {
      res.status(error.status || 500).json({ error: { message: error.message || 'Internal server error', code: error.status || 500 } });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `${req.path} not found`, code: 404 } });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy`);
  console.log(`   Port:              ${PORT}`);
  console.log(`   DeepSeek Pro ID:   ${DEEPSEEK_V4_PRO_ID}`);
  console.log(`   DeepSeek Flash ID: ${DEEPSEEK_V4_FLASH_ID}`);
  console.log(`   DeepSeek mode:     ${DEEPSEEK_REASONING_EFFORT}`);
  console.log(`   Fallback:          ОТКЛЮЧЁН — только повтор той же модели (${MAX_RETRIES_SAME_MODEL}x)`);
  console.log(`   First-token wait:  ${FIRST_TOKEN_TIMEOUT / 1000}s`);
  console.log(`   Idle timeout:      ${IDLE_TIMEOUT / 1000}s`);
  console.log(`   RP style prompt:   ${ENABLE_RP_STYLE_PROMPT}`);
  console.log(`   Keep-alive ping:   /ping\n`);
});
