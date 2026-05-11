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

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
// ⚠️ Only set to true for models that actually support chat_template_kwargs (e.g. nemotron, qwen3-thinking)
const ENABLE_THINKING_MODE = false; // ← ИСПРАВЛЕНО: было true, вызывало 400 у большинства моделей

// Models that support thinking mode (chat_template_kwargs)
const THINKING_CAPABLE_MODELS = [
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-next-80b-a3b-thinking',
  'qwen/qwen3-coder-480b-a35b-instruct',
];

// Model mapping — только актуальные модели NIM
const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'meta/llama-3.1-8b-instruct',
  'gpt-4':          'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo':    'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4o':         'meta/llama-3.3-70b-instruct',
  'claude-3-opus':  'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet':'meta/llama-3.1-70b-instruct',
  'gemini-pro':     'mistralai/mistral-large-2-instruct',
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      // Try the model name as-is against NIM
      try {
        const testRes = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: () => true
        });
        if (testRes.status >= 200 && testRes.status < 300) {
          nimModel = model;
        }
      } catch (e) { /* ignore */ }

      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // Determine if thinking mode should be applied for this specific model
    const useThinking = ENABLE_THINKING_MODE && THINKING_CAPABLE_MODELS.includes(nimModel);

    // Build NIM request
    // ✅ ИСПРАВЛЕНИЕ: chat_template_kwargs передаётся напрямую в тело запроса,
    //    а НЕ через extra_body (extra_body — это концепция Python SDK, не HTTP).
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false,
      ...(useThinking ? { chat_template_kwargs: { thinking: true } } : {})
    };

    console.log(`[Proxy] ${model} → ${nimModel} | thinking=${useThinking} | stream=${nimRequest.stream}`);

    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content;
              const content = data.choices[0].delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';
                if (reasoning && !reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }
                if (content && reasoningStarted) {
                  combinedContent += '\n</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }
                if (combinedContent) {
                  data.choices[0].delta.content = combinedContent;
                }
              } else {
                data.choices[0].delta.content = content || '';
              }
              delete data.choices[0].delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    // ✅ Улучшенное логирование: теперь видно реальный ответ NIM при ошибке
    const nimError = error.response?.data;
    console.error('Proxy error:', error.message);
    console.error('NIM API response:', JSON.stringify(nimError, null, 2));

    res.status(error.response?.status || 500).json({
      error: {
        message: nimError?.detail || nimError?.message || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500,
        nim_error: nimError
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`NIM base: ${NIM_API_BASE}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
