// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ─────────────────────────────────────────────
// 🔧 TOGGLES
// ─────────────────────────────────────────────

// Показывать ли блок <think>...</think> в ответе клиенту
const SHOW_REASONING = false;

// Глубина рассуждений для DeepSeek V4 Pro: 'low' | 'medium' | 'high' | 'max'
// low  = Non-think (быстрый, без CoT)
// high = Think High (логический анализ)
// max  = Think Max  (максимальная глубина)
const DEEPSEEK_REASONING_EFFORT = process.env.DEEPSEEK_REASONING_EFFORT || 'high';

// ─────────────────────────────────────────────
// 📋 МОДЕЛИ
// ─────────────────────────────────────────────

// Модели DeepSeek V4 на NIM — требуют ОБЯЗАТЕЛЬНОГО chat_template_kwargs + reasoning_effort
// ⚠️  Без этих параметров NIM зависает и не возвращает ответ!
const DEEPSEEK_V4_MODELS = new Set([
  'deepseek-ai/deepseek-v4-pro',
  'deepseek-ai/deepseek-v4-flash',
]);

// Прочие модели с опциональным thinking
const OPTIONAL_THINKING_MODELS = new Set([
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-next-80b-a3b-thinking',
  'qwen/qwen3-coder-480b-a35b-instruct',
  'moonshotai/kimi-k2-instruct',
]);

// Маппинг OpenAI-совместимых имён → реальные модели NIM
const MODEL_MAPPING = {
  // ✅ DeepSeek V4 Pro — основная модель
  'gpt-4o':              'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-pro':     'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash':   'deepseek-ai/deepseek-v4-flash',

  // Прочие модели
  'gpt-3.5-turbo':       'meta/llama-3.1-8b-instruct',
  'gpt-4':               'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo':         'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-opus':       'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet':     'meta/llama-3.1-70b-instruct',
  'gemini-pro':          'mistralai/mistral-large-2-instruct',
};

// ─────────────────────────────────────────────
// 🛠️ HELPERS
// ─────────────────────────────────────────────

/**
 * Возвращает дополнительные поля запроса в зависимости от модели.
 * DeepSeek V4: chat_template_kwargs + reasoning_effort ОБЯЗАТЕЛЬНЫ.
 * Другие thinking-модели: chat_template_kwargs опционально.
 */
function buildThinkingParams(nimModel) {
  if (DEEPSEEK_V4_MODELS.has(nimModel)) {
    return {
      chat_template_kwargs: { thinking: true },
      reasoning_effort: DEEPSEEK_REASONING_EFFORT,
    };
  }
  if (OPTIONAL_THINKING_MODELS.has(nimModel)) {
    return { chat_template_kwargs: { thinking: true } };
  }
  return {};
}

/**
 * Объединяет reasoning_content и content в одну строку.
 * Если SHOW_REASONING=false — возвращает только content.
 */
function mergeContent(content, reasoningContent) {
  if (SHOW_REASONING && reasoningContent) {
    return `<think>\n${reasoningContent}\n</think>\n\n${content || ''}`;
  }
  return content || '';
}

// ─────────────────────────────────────────────
// 🌐 ENDPOINTS
// ─────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    show_reasoning: SHOW_REASONING,
    deepseek_reasoning_effort: DEEPSEEK_REASONING_EFFORT,
    available_models: Object.keys(MODEL_MAPPING),
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy',
  }));
  res.json({ object: 'list', data: models });
});

// ─────────────────────────────────────────────
// 💬 CHAT COMPLETIONS
// ─────────────────────────────────────────────

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // 1. Определяем реальную модель NIM
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      // Пробуем использовать переданное имя напрямую
      try {
        const testRes = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          { model, messages: [{ role: 'user', content: 'hi' }], max_tokens: 1 },
          {
            headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
            validateStatus: () => true,
            timeout: 8000,
          }
        );
        if (testRes.status >= 200 && testRes.status < 300) nimModel = model;
      } catch (_) { /* ignore */ }

      // Fallback по ключевым словам
      if (!nimModel) {
        const m = model.toLowerCase();
        if (m.includes('deepseek-v4') || m.includes('deepseek_v4')) {
          nimModel = 'deepseek-ai/deepseek-v4-pro';
        } else if (m.includes('gpt-4') || m.includes('405b') || m.includes('opus')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // 2. Параметры thinking/reasoning для данной модели
    const thinkingParams = buildThinkingParams(nimModel);
    const isDeepSeek = DEEPSEEK_V4_MODELS.has(nimModel);

    // 3. Собираем запрос к NIM
    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? (isDeepSeek ? 0.6 : 0.7),
      max_tokens: max_tokens || (isDeepSeek ? 16384 : 4096),
      stream: stream || false,
      ...thinkingParams,
    };

    console.log(
      `[Proxy] "${model}" → "${nimModel}" | effort=${thinkingParams.reasoning_effort || 'n/a'} | stream=${nimRequest.stream}`
    );

    // 4. Запрос к NIM
    // DeepSeek V4 может долго думать → таймаут 5 минут
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
        },
        responseType: stream ? 'stream' : 'json',
        timeout: isDeepSeek ? 300_000 : 120_000,
      }
    );

    // ─────────────────────────────────────────
    // STREAMING
    // ─────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningBuffer = '';
      let reasoningClosed = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (!delta) { res.write(`data: ${JSON.stringify(data)}\n\n`); continue; }

            const reasoning = delta.reasoning_content || '';
            const content   = delta.content || '';

            if (SHOW_REASONING) {
              let out = '';
              if (reasoning) {
                if (reasoningBuffer === '') out += '<think>\n';
                reasoningBuffer += reasoning;
                out += reasoning;
              }
              if (content) {
                if (reasoningBuffer && !reasoningClosed) {
                  out += '\n</think>\n\n';
                  reasoningClosed = true;
                }
                out += content;
              }
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

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('[Stream error]', err.message);
        res.end();
      });

    // ─────────────────────────────────────────
    // NON-STREAMING
    // ─────────────────────────────────────────
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map((choice) => ({
          index: choice.index,
          message: {
            role: choice.message.role,
            content: mergeContent(
              choice.message?.content,
              choice.message?.reasoning_content
            ),
          },
          finish_reason: choice.finish_reason,
        })),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
        },
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    const nimError = error.response?.data;
    console.error('[Proxy error]', error.message);
    if (nimError) console.error('[NIM response]', JSON.stringify(nimError, null, 2));

    res.status(error.response?.status || 500).json({
      error: {
        message: nimError?.detail || nimError?.message || error.message || 'Internal server error',
        type: 'proxy_error',
        code: error.response?.status || 500,
        nim_error: nimError,
      },
    });
  }
});

// 404 catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'not_found', code: 404 },
  });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy`);
  console.log(`   Port:              ${PORT}`);
  console.log(`   NIM base:          ${NIM_API_BASE}`);
  console.log(`   Show reasoning:    ${SHOW_REASONING}`);
  console.log(`   DeepSeek effort:   ${DEEPSEEK_REASONING_EFFORT}`);
  console.log(`   Health:            http://localhost:${PORT}/health\n`);
});
