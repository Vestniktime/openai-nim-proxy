// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http  = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ─────────────────────────────────────────────
// ⚙️ CONFIG
// ─────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

const SHOW_REASONING = false;

// low = без CoT (быстро)  |  medium/high/max = с CoT (медленнее)
const DEEPSEEK_REASONING_EFFORT = process.env.DEEPSEEK_REASONING_EFFORT || 'low';

// Сколько секунд ждать ПЕРВОГО токена от NIM (очередь на сервере)
// Используем socket-таймаут на уровне TCP, не axios
const FIRST_TOKEN_TIMEOUT = parseInt(process.env.FIRST_TOKEN_TIMEOUT || '180000'); // 3 мин

// Сколько секунд тишины в потоке считать зависанием
const IDLE_TIMEOUT = parseInt(process.env.IDLE_TIMEOUT || '120000'); // 2 мин

// HTTP/S агенты — keepAlive чтобы не пересоздавать сокеты
const httpAgent  = new http.Agent ({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

// ─────────────────────────────────────────────
// 📋 МОДЕЛИ
// ─────────────────────────────────────────────

const DEEPSEEK_V4_MODELS = new Set([
  'deepseek-ai/deepseek-v4-pro',
  'deepseek-ai/deepseek-v4-flash',
]);

const OPTIONAL_THINKING_MODELS = new Set([
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-next-80b-a3b-thinking',
  'qwen/qwen3-coder-480b-a35b-instruct',
  'moonshotai/kimi-k2-instruct',
  'mistralai/mistral-medium-3.5-128b',
]);

const MODEL_MAPPING = {
  'gpt-4o':            'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-pro':   'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'gpt-3.5-turbo':     'meta/llama-3.1-8b-instruct',
  'gpt-4':             'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo':       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-opus':     'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet':   'meta/llama-3.1-70b-instruct',
  'gemini-pro':        'mistralai/mistral-medium-3.5-128b',
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

/**
 * Два уровня защиты от зависания:
 *
 * 1. firstTokenTimer — ждём первый байт не дольше FIRST_TOKEN_TIMEOUT.
 *    Запускается сразу, сбрасывается при получении первых данных.
 *
 * 2. idleTimer — после первого байта перезапускается при каждом chunk.
 *    Если данных нет дольше IDLE_TIMEOUT — разрушаем поток.
 */
function withSmartTimeout(stream, firstTokenMs, idleMs, label) {
  let firstTokenReceived = false;

  let timer = setTimeout(() => {
    console.error(`[First-token timeout] ${label} — нет ответа от NIM за ${firstTokenMs / 1000}с`);
    stream.destroy(new Error(`First-token timeout after ${firstTokenMs}ms`));
  }, firstTokenMs);

  stream.on('data', () => {
    clearTimeout(timer);
    if (!firstTokenReceived) {
      firstTokenReceived = true;
      console.log(`[${label}] Первый токен получен`);
    }
    // После первого токена переключаемся на idle-таймер
    timer = setTimeout(() => {
      console.error(`[Idle timeout] ${label} — нет данных ${idleMs / 1000}с`);
      stream.destroy(new Error(`Idle timeout after ${idleMs}ms`));
    }, idleMs);
  });

  stream.on('end',   () => clearTimeout(timer));
  stream.on('error', () => clearTimeout(timer));
}

// ─────────────────────────────────────────────
// 🌐 ENDPOINTS
// ─────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    show_reasoning: SHOW_REASONING,
    deepseek_reasoning_effort: DEEPSEEK_REASONING_EFFORT,
    first_token_timeout_ms: FIRST_TOKEN_TIMEOUT,
    idle_timeout_ms: IDLE_TIMEOUT,
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
    const { model, messages, temperature, max_tokens, stream: clientWantsStream } = req.body;

    // Резолв модели
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const m = model.toLowerCase();
      if      (m.includes('deepseek-v4-pro'))              nimModel = 'deepseek-ai/deepseek-v4-pro';
      else if (m.includes('deepseek-v4-flash')) nimModel = 'deepseek-ai/deepseek-v4-flash';
      else if (m.includes('mistral-medium-3.5-128b')) nimModel = 'mistralai/mistral-medium-3.5-128b';
      else                                               nimModel = 'meta/llama-3.1-8b-instruct';
    }

    const thinkingParams = buildThinkingParams(nimModel);
    const isDeepSeek     = DEEPSEEK_V4_MODELS.has(nimModel);
    const useStream      = clientWantsStream || isDeepSeek;

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? (isDeepSeek ? 1.0 : 1.0),
      max_tokens:  max_tokens  || (isDeepSeek ? 128000 : 12800),
      stream: useStream,
      ...thinkingParams,
    };

    console.log(
      `[Proxy] "${model}" → "${nimModel}" | effort=${thinkingParams.reasoning_effort || 'n/a'} | stream=${useStream}`
    );

    // ✅ timeout: 0 — отключаем axios-таймаут полностью.
    //    Таймауты контролируются через withSmartTimeout() на уровне TCP-потока.
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
        },
        responseType: 'stream',
        timeout: 0, // ← без таймаута на axios, только stream-level
        httpAgent,
        httpsAgent,
      }
    );

    // Вешаем двухуровневый таймаут на поток NIM
    withSmartTimeout(response.data, FIRST_TOKEN_TIMEOUT, IDLE_TIMEOUT, nimModel);

    // ─────────────────────────────────────────
    // CLIENT WANTS STREAM
    // ─────────────────────────────────────────
    if (clientWantsStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // Keepalive для Render (idle >30s → disconnect)
      const keepalive = setInterval(() => {
        if (!res.writableEnded) res.write(': keepalive\n\n');
      }, 20_000);

      let buffer = '';
      let reasoningBuffer = '';
      let reasoningClosed = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) { res.write('data: [DONE]\n\n'); continue; }

          try {
            const data  = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (!delta) { res.write(`data: ${JSON.stringify(data)}\n\n`); continue; }

            const reasoning = delta.reasoning_content || '';
            const content   = delta.content || '';

            if (SHOW_REASONING) {
              let out = '';
              if (reasoning) {
                if (!reasoningBuffer) out += '<think>\n';
                reasoningBuffer += reasoning;
                out += reasoning;
              }
              if (content && reasoningBuffer && !reasoningClosed) {
                out += '\n</think>\n\n';
                reasoningClosed = true;
              }
              if (content) out += content;
              delta.content = out;
            } else {
              delta.content = content;
            }

            delete delta.reasoning_content;
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) {
            res.write(line + '\n');
          }
        }
      });

      response.data.on('end', () => {
        clearInterval(keepalive);
        if (!res.writableEnded) res.end();
      });
      response.data.on('error', (err) => {
        clearInterval(keepalive);
        console.error('[Stream error]', err.message);
        if (!res.writableEnded) res.end();
      });

    // ─────────────────────────────────────────
    // CLIENT WANTS JSON (буферизуем forced-stream)
    // ─────────────────────────────────────────
    } else {
      let buf = '';
      let fullContent   = '';
      let fullReasoning = '';
      let lastChoice    = null;
      let usage         = null;

      await new Promise((resolve, reject) => {
        response.data.on('data', (chunk) => {
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
        response.data.on('end',   resolve);
        response.data.on('error', reject);
      });

      res.json({
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
        usage: usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (error) {
    let nimError = error.response?.data;
    if (nimError && typeof nimError.pipe === 'function') {
      nimError = await new Promise((resolve) => {
        let raw = '';
        nimError.on('data',  c => { raw += c.toString(); });
        nimError.on('end',   () => { try { resolve(JSON.parse(raw)); } catch { resolve(raw); } });
        nimError.on('error', () => resolve(null));
      });
    }

    console.error('[Proxy error]', error.message);
    if (nimError) console.error('[NIM response]', JSON.stringify(nimError, null, 2));

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: nimError?.detail || nimError?.message || error.message || 'Internal server error',
          type: 'proxy_error',
          code: error.response?.status || 500,
          nim_error: nimError,
        },
      });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'not_found', code: 404 },
  });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy`);
  console.log(`   Port:               ${PORT}`);
  console.log(`   NIM base:           ${NIM_API_BASE}`);
  console.log(`   Show reasoning:     ${SHOW_REASONING}`);
  console.log(`   DeepSeek effort:    ${DEEPSEEK_REASONING_EFFORT}`);
  console.log(`   First-token wait:   ${FIRST_TOKEN_TIMEOUT / 1000}s`);
  console.log(`   Idle timeout:       ${IDLE_TIMEOUT / 1000}s`);
  console.log(`   Health:             http://localhost:${PORT}/health\n`);
});
