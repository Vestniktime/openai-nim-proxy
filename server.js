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

// ✅ Снижено со 150с. Цепочка теперь может пробовать НЕСКОЛЬКО моделей подряд —
//    если каждая будет ждать по 150с перед тем как сдаться, худший случай
//    (все 3 модели зависли) — это 450+ секунд, что убьёт любой клиент.
//    45с на попытку — достаточно для "high" режима, и быстро уходим дальше по цепочке.
const FIRST_TOKEN_TIMEOUT = parseInt(process.env.FIRST_TOKEN_TIMEOUT || '45000');
const IDLE_TIMEOUT        = parseInt(process.env.IDLE_TIMEOUT        || '60000');
const KEEPALIVE_INTERVAL  = parseInt(process.env.KEEPALIVE_INTERVAL  || '10000');

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
//
// ✅ Подтверждено на официальных страницах NVIDIA:
//    - deepseek-ai/deepseek-v4-pro-0813   — ТЕКУЩИЙ официальный релиз Pro
//    - deepseek-ai/deepseek-v4-flash-0731 — ТЕКУЩИЙ официальный релиз Flash
//    Голые "deepseek-ai/deepseek-v4-pro" и "…-flash" (без даты) — это
//    отставленные preview-версии, они и были источником 410/сбоев.
// ─────────────────────────────────────────────

const DEEPSEEK_V4_PRO_ID   = 'deepseek-ai/deepseek-v4-pro-0813';
const DEEPSEEK_V4_FLASH_ID = 'deepseek-ai/deepseek-v4-flash-0731';
const STABLE_FALLBACK_ID   = 'meta/llama-3.1-70b-instruct'; // давно живёт на NIM, не менялся

// ✅ Определяем принадлежность к семейству DeepSeek V4 ПО ПОДСТРОКЕ,
//    а не точным совпадением строки. Так любой будущий суффикс (NVIDIA
//    наверняка ещё не раз переименует модель) не сломает логику —
//    раньше именно точное совпадение срезало "-0813"/"-0731" и уводило
//    запрос на другую (часто мёртвую) модель.
function isDeepSeekV4Pro(nimModel)   { return nimModel.includes('deepseek-v4-pro'); }
function isDeepSeekV4Flash(nimModel) { return nimModel.includes('deepseek-v4-flash'); }
function isDeepSeekV4(nimModel)      { return isDeepSeekV4Pro(nimModel) || isDeepSeekV4Flash(nimModel); }

const OPTIONAL_THINKING_MODELS = new Set([
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-next-80b-a3b-thinking',
  'qwen/qwen3-coder-480b-a35b-instruct',
]);

// ✅ Цепочка строится динамически, и для ЛЮБОЙ модели заканчивается
//    на стабильной Llama-70b — гарантированный последний рубеж.
function getFallbackChain(nimModel) {
  if (nimModel === STABLE_FALLBACK_ID) return [];
  if (isDeepSeekV4Pro(nimModel))       return [DEEPSEEK_V4_FLASH_ID, STABLE_FALLBACK_ID];
  if (isDeepSeekV4Flash(nimModel))     return [DEEPSEEK_V4_PRO_ID,   STABLE_FALLBACK_ID];
  return [STABLE_FALLBACK_ID];
}

const MODEL_MAPPING = {
  'deepseek-v4-pro':   DEEPSEEK_V4_PRO_ID,
  'deepseek-v4-flash': DEEPSEEK_V4_FLASH_ID,
  'gpt-4o':            DEEPSEEK_V4_PRO_ID,   // ✅ раньше указывал на мёртвый preview
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
 * ✅ ГЛАВНЫЙ ФИКС: цепочка фолбэка теперь ИДЁТ ДАЛЬШЕ при ЛЮБОЙ ошибке
 * на любом шаге, кроме последнего. Раньше конкретный список статусов
 * (429/500/502/503/504) решал, продолжать или сдаваться — и любой код
 * вне этого списка (410 Gone, 404, DEGRADED-подобные ответы, обрыв
 * соединения без стандартного статуса) обрывал ВСЮ цепочку, даже если
 * следующая модель (например, стабильная Llama-70b) была рабочей.
 * Теперь единственная причина сдаться — это когда МЫ УЖЕ НА ПОСЛЕДНЕЙ
 * модели в цепочке и её тоже не получилось вызвать.
 */
async function nimFetchWithFallback(preferredModel, body) {
  const chain = [preferredModel, ...getFallbackChain(preferredModel)];
  let lastError;

  for (let i = 0; i < chain.length; i++) {
    const model  = chain[i];
    const isLast = i === chain.length - 1;
    try {
      console.log(`[NIM] Попытка ${i + 1}/${chain.length}: ${model}`);
      return { stream: await nimFetch(model, body), usedModel: model };
    } catch (err) {
      lastError = err;
      const reason = err.status ? `HTTP ${err.status}` : err.message;
      if (isLast) {
        console.error(`[NIM] ${model} → ${reason} (последняя модель в цепочке, сдаёмся)`);
        throw err;
      }
      console.warn(`[NIM] ${model} → ${reason}, пробуем следующую модель…`);
      await sleep(400);
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
    stable_fallback: STABLE_FALLBACK_ID,
  },
  deepseek_mode: DEEPSEEK_REASONING_EFFORT,
  rp_style_prompt_enabled: ENABLE_RP_STYLE_PROMPT,
  min_response_tokens: MIN_RESPONSE_TOKENS,
  first_token_timeout_ms: FIRST_TOKEN_TIMEOUT,
  idle_timeout_ms: IDLE_TIMEOUT,
  fallback_policy: 'всегда идёт дальше по цепочке при любой ошибке, кроме последней модели',
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
        // ✅ Строка уже похожа на настоящий "provider/model" ID
        //    (например, если в Janitor вписали что-то ещё точнее наших
        //    алиасов) — передаём КАК ЕСТЬ, ничего не срезаем и не угадываем.
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
    const { stream: nimStream, usedModel } = await nimFetchWithFallback(nimModel, body);

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
  console.log(`   Stable fallback:   ${STABLE_FALLBACK_ID}`);
  console.log(`   DeepSeek mode:     ${DEEPSEEK_REASONING_EFFORT}`);
  console.log(`   First-token wait:  ${FIRST_TOKEN_TIMEOUT / 1000}s (за модель, не за всю цепочку)`);
  console.log(`   RP style prompt:   ${ENABLE_RP_STYLE_PROMPT}`);
  console.log(`   Keep-alive ping:   /ping\n`);
});
