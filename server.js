// server.js - OpenAI-compatible proxy for NVIDIA hosted NIM
// Supports:
//   POST /v1/chat/completions  -> NVIDIA /v1/chat/completions
//   POST /v1/completions       -> compatibility bridge over NVIDIA /v1/chat/completions
//
// Important: NVIDIA hosted API at integrate.api.nvidia.com exposes chat completions.
// /v1/completions below is therefore an OpenAI Text Completions compatibility layer.

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
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'Content-Type, Authorization, X-Requested-With, Accept, Origin'
  );
  res.setHeader('Access-Control-Max-Age', '86400');

  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// ─────────────────────────────────────────────
// Universal JSON body parser
// ─────────────────────────────────────────────
app.use((req, res, next) => {
  if (req.method === 'GET' || req.method === 'OPTIONS' || req.method === 'HEAD') {
    return next();
  }

  let raw = '';
  req.setEncoding('utf8');

  req.on('data', chunk => {
    raw += chunk;
  });

  req.on('end', () => {
    if (!raw) return next();

    try {
      req.body = JSON.parse(raw);
    } catch {
      req.body = raw;
    }

    next();
  });

  req.on('error', () => next());
});

// ─────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

// Chat reasoning is returned in reasoning_content; Text Completion uses <think> XML.
const DEEPSEEK_REASONING_EFFORT =
  process.env.DEEPSEEK_REASONING_EFFORT || 'low';

// GLM-5.3 reasoning and output limits are controlled by the client (SillyTavern).
// The proxy does not force an effort level or a minimum response length.

const FIRST_TOKEN_TIMEOUT =
  parseInt(process.env.FIRST_TOKEN_TIMEOUT || '480000', 10); // 8 min

const IDLE_TIMEOUT =
  parseInt(process.env.IDLE_TIMEOUT || '120000', 10); // 2 min

const KEEPALIVE_INTERVAL =
  parseInt(process.env.KEEPALIVE_INTERVAL || '8000', 10); // 8 sec

const MAX_RETRIES_SAME_MODEL =
  parseInt(process.env.MAX_RETRIES_SAME_MODEL || '2', 10);

const MIN_RESPONSE_TOKENS =
  parseInt(process.env.MIN_RESPONSE_TOKENS || '2048', 10);

// RP prompt is intentionally disabled.
// SillyTavern is expected to build the prompt itself.
const ENABLE_RP_STYLE_PROMPT = false;
const RP_STYLE_INSTRUCTION   = '';

const httpAgent  = new http.Agent({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

// ─────────────────────────────────────────────
// MODELS
// ─────────────────────────────────────────────
const DEEPSEEK_V4_PRO_ID   = 'deepseek-ai/deepseek-v4-pro-0813';
const DEEPSEEK_V4_FLASH_ID = 'deepseek-ai/deepseek-v4-flash-0731';

function isDeepSeekV4(model) {
  return (
    model.includes('deepseek-v4-pro') ||
    model.includes('deepseek-v4-flash')
  );
}

function isGLM53(model) {
  const value = String(model || '').toLowerCase();
  return (
    value.includes('glm-5.3') ||
    value.includes('glm-5-3')
  );
}

function normalizeGLM53ReasoningEffort(value) {
  const effort = String(value ?? '').trim().toLowerCase();

  // No value / Auto: do not inject the parameter. NVIDIA decides the default.
  if (!effort || effort === 'auto') return undefined;

  // GLM-5.3 on NVIDIA accepts only low / high / max.
  // Map common SillyTavern/OpenAI-style levels to the nearest supported value.
  if (['none', 'off', 'disabled', 'minimal', 'minimum', 'low'].includes(effort)) {
    return 'low';
  }
  if (['medium', 'high'].includes(effort)) {
    return 'high';
  }
  if (['xhigh', 'max', 'maximum'].includes(effort)) {
    return 'max';
  }

  return undefined;
}

function getRequestedReasoningEffort(sourceBody = {}) {
  if (sourceBody.reasoning_effort !== undefined) {
    return sourceBody.reasoning_effort;
  }

  if (sourceBody.reasoning && sourceBody.reasoning.effort !== undefined) {
    return sourceBody.reasoning.effort;
  }

  if (sourceBody.output_config && sourceBody.output_config.effort !== undefined) {
    return sourceBody.output_config.effort;
  }

  if (
    sourceBody.chat_template_kwargs &&
    sourceBody.chat_template_kwargs.reasoning_effort !== undefined
  ) {
    return sourceBody.chat_template_kwargs.reasoning_effort;
  }

  // GLM-5.3 cannot fully disable reasoning on NVIDIA.
  // If a client explicitly requests disabled thinking, use the lightest budget.
  const thinkingType = String(sourceBody.thinking?.type ?? '').toLowerCase();
  if (['disabled', 'off', 'none'].includes(thinkingType)) {
    return 'low';
  }

  return undefined;
}

const OPTIONAL_THINKING_MODELS = new Set([
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-next-80b-a3b-thinking',
]);

const MODEL_MAPPING = {
  'deepseek-v4-pro':   DEEPSEEK_V4_PRO_ID,
  'deepseek-v4-flash': DEEPSEEK_V4_FLASH_ID,
  'glm-5.3':           'z-ai/glm-5.3',
  'glm-5.3-flash':     'z-ai/glm-5.3-flash',

  'gpt-4o':        DEEPSEEK_V4_PRO_ID,
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4':         'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo':   'meta/llama-3.1-405b-instruct',

  'claude-3-opus':   'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',

  'gemini-pro': 'mistralai/mistral-large-2-instruct',
};

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────
function resolveModel(model) {
  return MODEL_MAPPING[model] || model;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function copyDefinedParams(source, target, keys) {
  for (const key of keys) {
    if (source[key] !== undefined) target[key] = source[key];
  }
}

function buildThinkingParams(nimModel, sourceBody = {}) {
  if (isGLM53(nimModel)) {
    const requestedEffort = getRequestedReasoningEffort(sourceBody);
    const effort = normalizeGLM53ReasoningEffort(requestedEffort);

    const params = {
      // Recommended by NVIDIA for ordinary multi-turn chat/RP.
      // This does not set the reasoning budget; it only clears previous thinking.
      chat_template_kwargs: {
        ...(sourceBody.chat_template_kwargs || {}),
        enable_thinking: true,
        clear_thinking: sourceBody.chat_template_kwargs?.clear_thinking ?? true,
      },
    };

    // Only send reasoning_effort when SillyTavern/client actually supplied one.
    if (effort) {
      params.reasoning_effort = effort;
    }

    return params;
  }

  if (isDeepSeekV4(nimModel)) {
    const effort = (DEEPSEEK_REASONING_EFFORT || 'low').toLowerCase();

    if (effort === 'max') {
      return {
        chat_template_kwargs: {
          enable_thinking: true,
          thinking: true,
          reasoning_effort: 'max',
        },
      };
    }

    if (effort === 'high' || effort === 'medium') {
      return {
        chat_template_kwargs: {
          enable_thinking: true,
          thinking: true,
          reasoning_effort: 'high',
        },
      };
    }

    return {
      chat_template_kwargs: {
        enable_thinking: false,
        thinking: false,
      },
    };
  }

  if (OPTIONAL_THINKING_MODELS.has(nimModel)) {
    return {
      chat_template_kwargs: {
        thinking: true,
      },
    };
  }

  return {};
}

function applyStylePrompt(messages) {
  if (
    !ENABLE_RP_STYLE_PROMPT ||
    !Array.isArray(messages) ||
    messages.length === 0
  ) {
    return messages;
  }

  const out = messages.map(message => ({ ...message }));

  if (out[0]?.role === 'system') {
    out[0] = {
      ...out[0],
      content: `${out[0].content}\n\n${RP_STYLE_INSTRUCTION}`,
    };
  } else {
    out.unshift({
      role: 'system',
      content: RP_STYLE_INSTRUCTION,
    });
  }

  return out;
}

function mergeContent(content, reasoningContent) {
  if (SHOW_REASONING && reasoningContent) {
    return `<think>\n${reasoningContent}\n</think>\n\n${content || ''}`;
  }

  return content || '';
}

function withSmartTimeout(stream, label) {
  let first = false;

  let timer = setTimeout(() => {
    console.error(
      `[Timeout] ${label}: no first token for ${FIRST_TOKEN_TIMEOUT / 1000}s`
    );
    stream.destroy(new Error('first-token timeout'));
  }, FIRST_TOKEN_TIMEOUT);

  stream.on('data', () => {
    clearTimeout(timer);

    if (!first) {
      first = true;
      console.log(`[${label}] first token received`);
    }

    timer = setTimeout(() => {
      console.error(
        `[Idle] ${label}: no data for ${IDLE_TIMEOUT / 1000}s`
      );
      stream.destroy(new Error('idle timeout'));
    }, IDLE_TIMEOUT);
  });

  stream.on('end', () => clearTimeout(timer));
  stream.on('error', () => clearTimeout(timer));
}

async function readErrorStream(stream) {
  return new Promise(resolve => {
    let text = '';

    stream.on('data', chunk => {
      text += chunk.toString();
    });

    stream.on('end', () => resolve(text));
    stream.on('error', () => resolve(text));
  });
}

// ─────────────────────────────────────────────
// NVIDIA HOSTED CHAT REQUEST
// ─────────────────────────────────────────────
function buildNimChatPayload(nimModel, sourceBody, messages) {
  const isDeepSeek = isDeepSeekV4(nimModel);
  const isGLM = isGLM53(nimModel);

  const rawMaxTokens =
    sourceBody.max_tokens ??
    sourceBody.max_completion_tokens;

  const parsedMaxTokens = Number(rawMaxTokens);

  const requestedMaxTokens =
    Number.isFinite(parsedMaxTokens) && parsedMaxTokens > 0
      ? Math.floor(parsedMaxTokens)
      : 4096;

  const payload = {
    model: nimModel,
    messages: applyStylePrompt(messages),

    temperature:
      sourceBody.temperature ?? (isDeepSeek ? 1.0 : (isGLM ? 0.5 : 0.7)),

    top_p:
      sourceBody.top_p ?? (isDeepSeek ? 0.95 : (isGLM ? 1.0 : undefined)),

    // Respect the response length selected in SillyTavern.
    max_tokens: requestedMaxTokens,

    // Internally we always request a stream.
    // It gives us early data and allows Render/client keepalive handling.
    stream: true,

    ...buildThinkingParams(nimModel, sourceBody),
  };

  // Parameters that are common to NVIDIA hosted Chat Completions.
  copyDefinedParams(sourceBody, payload, [
    'stop',
    'frequency_penalty',
    'presence_penalty',
    'seed',
  ]);

  return payload;
}

async function nimChatFetch(nimModel, sourceBody, messages, label = 'chat') {
  const payload = buildNimChatPayload(
    nimModel,
    sourceBody,
    messages
  );

  const response = await axios.post(
    `${NIM_API_BASE}/chat/completions`,
    payload,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },

      responseType: 'stream',
      timeout: 0,

      httpAgent,
      httpsAgent,

      validateStatus: null,
    }
  );

  if (response.status >= 400) {
    const raw = await readErrorStream(response.data);

    const err = new Error(
      `NIM ${response.status}: ${raw || response.statusText || 'request failed'}`
    );

    err.status = response.status;
    throw err;
  }

  withSmartTimeout(
    response.data,
    `${nimModel} [${label}]`
  );

  return response.data;
}

async function nimChatFetchWithRetry(
  nimModel,
  sourceBody,
  messages,
  label = 'chat'
) {
  let lastError;

  for (
    let attempt = 1;
    attempt <= MAX_RETRIES_SAME_MODEL;
    attempt++
  ) {
    const isLast =
      attempt === MAX_RETRIES_SAME_MODEL;

    try {
      console.log(
        `[NIM ${label}] attempt ${attempt}/${MAX_RETRIES_SAME_MODEL}: ${nimModel}`
      );

      return await nimChatFetch(
        nimModel,
        sourceBody,
        messages,
        label
      );
    } catch (err) {
      lastError = err;

      if (isLast) {
        throw err;
      }

      console.warn(
        `[NIM ${label}] ${nimModel} -> ${err.status || err.message}; retry in 2s`
      );

      await sleep(2000);
    }
  }

  throw lastError;
}

// ─────────────────────────────────────────────
// KEEPALIVE
// ─────────────────────────────────────────────
function makeChatKeepaliveChunk(model) {
  return `data: ${JSON.stringify({
    id: 'chatcmpl-keepalive',
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: null,
      },
    ],
  })}\n\n`;
}

function makeTextKeepaliveChunk(model) {
  return `data: ${JSON.stringify({
    id: 'cmpl-keepalive',
    object: 'text_completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        text: '',
        logprobs: null,
        finish_reason: null,
      },
    ],
  })}\n\n`;
}

// ─────────────────────────────────────────────
// CHAT STREAM COLLECTION
// ─────────────────────────────────────────────
function collectChatStream(nimStream) {
  return new Promise((resolve, reject) => {
    let buffer = '';
    let usage = null;

    const choices = new Map();

    function consumeLine(line) {
      if (
        !line.startsWith('data: ') ||
        line.includes('[DONE]')
      ) {
        return;
      }

      try {
        const data = JSON.parse(line.slice(6));

        if (data.usage) {
          usage = data.usage;
        }

        for (const choice of data.choices || []) {
          const index =
            Number.isInteger(choice.index)
              ? choice.index
              : 0;

          const state =
            choices.get(index) || {
              content: '',
              reasoning: '',
              finish_reason: null,
            };

          state.content +=
            choice.delta?.content || '';

          state.reasoning +=
            choice.delta?.reasoning_content || '';

          if (choice.finish_reason !== undefined) {
            state.finish_reason =
              choice.finish_reason;
          }

          choices.set(index, state);
        }
      } catch {
        // Ignore malformed/non-JSON SSE lines.
      }
    }

    nimStream.on('data', chunk => {
      buffer += chunk.toString();

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        consumeLine(line);
      }
    });

    nimStream.on('end', () => {
      if (buffer) {
        consumeLine(buffer);
      }

      resolve({
        choices: [...choices.entries()]
          .sort((a, b) => a[0] - b[0])
          .map(([index, state]) => ({
            index,
            content: state.content,
            reasoning: state.reasoning,
            finish_reason:
              state.finish_reason || 'stop',
          })),

        usage,
      });
    });

    nimStream.on('error', reject);
  });
}

// ─────────────────────────────────────────────
// TEXT COMPLETION BRIDGE HELPERS
// ─────────────────────────────────────────────
function promptToMessages(prompt) {
  // NVIDIA hosted API does not expose native /v1/completions.
  // We keep SillyTavern's complete formatted prompt intact and place
  // the whole prompt inside a single user message.
  return [
    {
      role: 'user',
      content: prompt,
    },
  ];
}

function createReasoningState() {
  return {
    started: false,
    closed: false,
  };
}

function chatDeltaToTextCompletionText(delta, state) {
  const reasoning =
    delta?.reasoning_content ??
    delta?.reasoning ??
    '';

  const content =
    delta?.content || '';

  let out = '';

  // Text Completion has no standard reasoning_content channel.
  // Emit Think XML so SillyTavern Reasoning Auto-Parse can split it
  // into a dedicated collapsible reasoning block.
  if (reasoning) {
    if (!state.started) {
      out += '<think>\n';
      state.started = true;
    }

    out += reasoning;
  }

  if (
    content &&
    state.started &&
    !state.closed
  ) {
    out += '\n</think>\n\n';
    state.closed = true;
  }

  if (content) {
    out += content;
  }

  return out;
}

function convertChatChunkToTextChunk(
  data,
  requestedModel,
  reasoningStates
) {
  const textChoices = [];

  for (const choice of data.choices || []) {
    const index =
      Number.isInteger(choice.index)
        ? choice.index
        : 0;

    if (!reasoningStates.has(index)) {
      reasoningStates.set(
        index,
        createReasoningState()
      );
    }

    const state = reasoningStates.get(index);

    let text =
      chatDeltaToTextCompletionText(
        choice.delta || {},
        state
      );

    // If generation ended while still inside reasoning, close the tag so
    // SillyTavern can parse the block cleanly.
    if (
      choice.finish_reason !== null &&
      choice.finish_reason !== undefined &&
      state.started &&
      !state.closed
    ) {
      text += '\n</think>\n\n';
      state.closed = true;
    }

    // Emit a choice if it contains text OR carries a finish_reason.
    if (
      text ||
      choice.finish_reason !== null &&
      choice.finish_reason !== undefined
    ) {
      textChoices.push({
        index,
        text,
        logprobs: null,
        finish_reason:
          choice.finish_reason ?? null,
      });
    }
  }

  if (textChoices.length === 0) {
    return null;
  }

  return {
    id:
      typeof data.id === 'string'
        ? data.id.replace(/^chatcmpl-/, 'cmpl-')
        : `cmpl-${Date.now()}`,

    object: 'text_completion',

    created:
      data.created ||
      Math.floor(Date.now() / 1000),

    model: requestedModel,

    choices: textChoices,

    ...(data.usage
      ? { usage: data.usage }
      : {}),
  };
}

// ─────────────────────────────────────────────
// BASIC ENDPOINTS
// ─────────────────────────────────────────────
app.get('/ping', (_, res) => {
  res.send('pong');
});

app.get('/health', (_, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI -> NVIDIA hosted NIM proxy',

    nim_key_set:
      !!NIM_API_KEY,

    upstream_api:
      `${NIM_API_BASE}/chat/completions`,

    text_completion_mode:
      'compatibility bridge over chat/completions',

    native_nvidia_text_completions:
      false,

    supported_endpoints: [
      '/v1/chat/completions',
      '/v1/completions',
    ],

    deepseek_mode:
      DEEPSEEK_REASONING_EFFORT,

    glm53_reasoning_control:
      'client/SillyTavern passthrough (low/high/max)',

    response_length_control:
      'client/SillyTavern max_tokens passthrough',

    min_response_tokens:
      MIN_RESPONSE_TOKENS,

    max_retries_same_model:
      MAX_RETRIES_SAME_MODEL,

    first_token_timeout_ms:
      FIRST_TOKEN_TIMEOUT,

    idle_timeout_ms:
      IDLE_TIMEOUT,

    keepalive_interval_ms:
      KEEPALIVE_INTERVAL,

    available_models:
      Object.keys(MODEL_MAPPING),
  });
});

app.get('/v1/models', (_, res) => {
  const created =
    Math.floor(Date.now() / 1000);

  res.json({
    object: 'list',

    data: [
      ...Object.entries(MODEL_MAPPING)
        .map(([alias, real]) => ({
          id: alias,
          object: 'model',
          created,
          owned_by: 'alias',
          description: `-> ${real}`,
        })),

      {
        id: DEEPSEEK_V4_PRO_ID,
        object: 'model',
        created,
        owned_by: 'nvidia-nim',
      },

      {
        id: DEEPSEEK_V4_FLASH_ID,
        object: 'model',
        created,
        owned_by: 'nvidia-nim',
      },

      {
        id: 'meta/llama-3.1-70b-instruct',
        object: 'model',
        created,
        owned_by: 'nvidia-nim',
      },

      {
        id: 'meta/llama-3.1-405b-instruct',
        object: 'model',
        created,
        owned_by: 'nvidia-nim',
      },

      {
        id: 'mistralai/mistral-large-2-instruct',
        object: 'model',
        created,
        owned_by: 'nvidia-nim',
      },
    ],
  });
});

// ─────────────────────────────────────────────
// POST /v1/chat/completions
// ─────────────────────────────────────────────
app.post(
  '/v1/chat/completions',
  async (req, res) => {
    const body = req.body || {};

    console.log(
      '[Chat request] model:',
      body.model,
      '| messages:',
      Array.isArray(body.messages)
        ? body.messages.length
        : typeof body.messages,
      '| stream:',
      body.stream,
      '| max_tokens:',
      body.max_tokens ?? body.max_completion_tokens,
      '| reasoning_effort:',
      body.reasoning_effort ?? body.reasoning?.effort ?? body.output_config?.effort ?? 'not-sent'
    );

    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'NIM_API_KEY is not set',
          code: 500,
        },
      });
    }

    if (
      !Array.isArray(body.messages) ||
      body.messages.length === 0
    ) {
      return res.status(400).json({
        error: {
          message:
            'messages must be a non-empty array',
          code: 400,
        },
      });
    }

    const requestedModel =
      body.model || DEEPSEEK_V4_PRO_ID;

    const nimModel =
      resolveModel(requestedModel);

    const clientWantsStream =
      body.stream === true;

    if (clientWantsStream) {
      res.setHeader(
        'Content-Type',
        'text/event-stream'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      res.flushHeaders();

      const keepalive =
        setInterval(() => {
          if (!res.writableEnded) {
            res.write(
              makeChatKeepaliveChunk(
                requestedModel
              )
            );
          }
        }, KEEPALIVE_INTERVAL);

      let nimStream;

      try {
        nimStream =
          await nimChatFetchWithRetry(
            nimModel,
            body,
            body.messages,
            'chat'
          );
      } catch (err) {
        clearInterval(keepalive);

        console.error(
          '[Chat proxy error]',
          err.message
        );

        if (!res.writableEnded) {
          const errorChunk = {
            id: 'chatcmpl-error',
            object:
              'chat.completion.chunk',
            created:
              Math.floor(Date.now() / 1000),
            model: requestedModel,

            choices: [
              {
                index: 0,

                delta: {
                  content:
                    `\n\n[Error: ${err.message}]`,
                },

                finish_reason: 'stop',
              },
            ],
          };

          res.write(
            `data: ${JSON.stringify(errorChunk)}\n\n`
          );

          res.write(
            'data: [DONE]\n\n'
          );

          res.end();
        }

        return;
      }

      clearInterval(keepalive);

      let buffer = '';

      const reasoningStates =
        new Map();

      nimStream.on('data', chunk => {
        buffer += chunk.toString();

        const lines =
          buffer.split('\n');

        buffer =
          lines.pop() || '';

        for (const line of lines) {
          if (
            !line.startsWith('data: ')
          ) {
            continue;
          }

          if (
            line.includes('[DONE]')
          ) {
            if (!res.writableEnded) {
              res.write(
                'data: [DONE]\n\n'
              );
            }
            continue;
          }

          try {
            const data =
              JSON.parse(line.slice(6));

            for (
              const choice of
              data.choices || []
            ) {
              const index =
                Number.isInteger(
                  choice.index
                )
                  ? choice.index
                  : 0;

              if (
                !reasoningStates.has(index)
              ) {
                reasoningStates.set(
                  index,
                  createReasoningState()
                );
              }

              if (choice.delta) {
                // Preserve NVIDIA's reasoning channel for SillyTavern.
                // Custom (OpenAI-compatible) can display delta.reasoning_content
                // when "Request model reasoning" is enabled.
                if (
                  choice.delta.reasoning_content === undefined &&
                  choice.delta.reasoning !== undefined
                ) {
                  choice.delta.reasoning_content =
                    choice.delta.reasoning;
                }

                choice.delta.content =
                  choice.delta.content || '';
              }
            }

            data.model =
              requestedModel;

            if (!res.writableEnded) {
              res.write(
                `data: ${JSON.stringify(data)}\n\n`
              );
            }
          } catch {
            if (!res.writableEnded) {
              res.write(
                `${line}\n\n`
              );
            }
          }
        }
      });

      nimStream.on('end', () => {
        if (!res.writableEnded) {
          res.end();
        }
      });

      nimStream.on('error', err => {
        console.error(
          '[Chat stream error]',
          err.message
        );

        if (!res.writableEnded) {
          res.end();
        }
      });

      return;
    }

    // Non-stream to client.
    // Internally NVIDIA is still streamed.
    res.setHeader(
      'Content-Type',
      'application/json'
    );

    res.setHeader(
      'Transfer-Encoding',
      'chunked'
    );

    res.setHeader(
      'Connection',
      'keep-alive'
    );

    res.flushHeaders();

    const keepalive =
      setInterval(() => {
        if (!res.writableEnded) {
          // Whitespace is valid before JSON.
          res.write(' ');
        }
      }, KEEPALIVE_INTERVAL);

    let nimStream;

    try {
      nimStream =
        await nimChatFetchWithRetry(
          nimModel,
          body,
          body.messages,
          'chat'
        );
    } catch (err) {
      clearInterval(keepalive);

      console.error(
        '[Chat proxy error]',
        err.message
      );

      if (!res.writableEnded) {
        res.end(
          JSON.stringify({
            error: {
              message: err.message,
              code: err.status || 500,
            },
          })
        );
      }

      return;
    }

    clearInterval(keepalive);

    let collected;

    try {
      collected =
        await collectChatStream(
          nimStream
        );
    } catch (err) {
      console.error(
        '[Chat collect error]',
        err.message
      );

      if (!res.writableEnded) {
        res.end(
          JSON.stringify({
            error: {
              message: err.message,
              code: 500,
            },
          })
        );
      }

      return;
    }

    const choices =
      collected.choices.length
        ? collected.choices.map(
            choice => ({
              index: choice.index,

              message: {
                role: 'assistant',
                content: choice.content || '',
                ...(choice.reasoning
                  ? { reasoning_content: choice.reasoning }
                  : {}),
              },

              finish_reason:
                choice.finish_reason,
            })
          )
        : [
            {
              index: 0,

              message: {
                role: 'assistant',
                content: '',
              },

              finish_reason: 'stop',
            },
          ];

    if (!res.writableEnded) {
      res.end(
        JSON.stringify({
          id:
            `chatcmpl-${Date.now()}`,

          object:
            'chat.completion',

          created:
            Math.floor(Date.now() / 1000),

          model:
            requestedModel,

          choices,

          usage:
            collected.usage || {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
            },
        })
      );
    }
  }
);

// ─────────────────────────────────────────────
// POST /v1/completions
//
// OpenAI Text Completions compatibility endpoint.
//
// NVIDIA hosted NIM does NOT receive /v1/completions.
// The complete prompt is wrapped into one user message,
// sent to NVIDIA /v1/chat/completions, then converted back
// into OpenAI text_completion format.
// ─────────────────────────────────────────────
app.post(
  '/v1/completions',
  async (req, res) => {
    const body = req.body || {};

    console.log(
      '[Text compatibility request] model:',
      body.model,
      '| prompt:',
      typeof body.prompt,
      '| stream:',
      body.stream,
      '| max_tokens:',
      body.max_tokens ?? body.max_completion_tokens,
      '| reasoning_effort:',
      body.reasoning_effort ?? body.reasoning?.effort ?? body.output_config?.effort ?? 'not-sent'
    );

    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message:
            'NIM_API_KEY is not set',
          code: 500,
        },
      });
    }

    // SillyTavern normally sends a string prompt.
    // Native OpenAI completions can also accept arrays,
    // but NVIDIA hosted chat has no equivalent multi-prompt call.
    if (
      typeof body.prompt !== 'string' ||
      body.prompt.length === 0
    ) {
      return res.status(400).json({
        error: {
          message:
            'This NVIDIA hosted compatibility bridge currently requires prompt to be a non-empty string',
          code: 400,
        },
      });
    }

    const requestedModel =
      body.model || DEEPSEEK_V4_PRO_ID;

    const nimModel =
      resolveModel(requestedModel);

    const messages =
      promptToMessages(body.prompt);

    const clientWantsStream =
      body.stream === true;

    if (clientWantsStream) {
      res.setHeader(
        'Content-Type',
        'text/event-stream'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      res.flushHeaders();

      const keepalive =
        setInterval(() => {
          if (!res.writableEnded) {
            res.write(
              makeTextKeepaliveChunk(
                requestedModel
              )
            );
          }
        }, KEEPALIVE_INTERVAL);

      let nimStream;

      try {
        nimStream =
          await nimChatFetchWithRetry(
            nimModel,
            body,
            messages,
            'text-bridge'
          );
      } catch (err) {
        clearInterval(keepalive);

        console.error(
          '[Text bridge proxy error]',
          err.message
        );

        if (!res.writableEnded) {
          const errorChunk = {
            id: 'cmpl-error',
            object:
              'text_completion',

            created:
              Math.floor(Date.now() / 1000),

            model:
              requestedModel,

            choices: [
              {
                index: 0,
                text:
                  `\n\n[Error: ${err.message}]`,
                logprobs: null,
                finish_reason: 'stop',
              },
            ],
          };

          res.write(
            `data: ${JSON.stringify(errorChunk)}\n\n`
          );

          res.write(
            'data: [DONE]\n\n'
          );

          res.end();
        }

        return;
      }

      clearInterval(keepalive);

      let buffer = '';

      const reasoningStates =
        new Map();

      let doneSent = false;

      function sendDone() {
        if (
          !doneSent &&
          !res.writableEnded
        ) {
          doneSent = true;

          res.write(
            'data: [DONE]\n\n'
          );
        }
      }

      function consumeSseLine(line) {
        if (
          !line.startsWith('data: ')
        ) {
          return;
        }

        if (
          line.includes('[DONE]')
        ) {
          sendDone();
          return;
        }

        try {
          const data =
            JSON.parse(line.slice(6));

          const converted =
            convertChatChunkToTextChunk(
              data,
              requestedModel,
              reasoningStates
            );

          if (
            converted &&
            !res.writableEnded
          ) {
            res.write(
              `data: ${JSON.stringify(converted)}\n\n`
            );
          }
        } catch {
          // Do not forward raw chat chunks to a text-completion client.
          // Invalid/non-JSON upstream lines are ignored.
        }
      }

      nimStream.on('data', chunk => {
        buffer += chunk.toString();

        const lines =
          buffer.split('\n');

        buffer =
          lines.pop() || '';

        for (const line of lines) {
          consumeSseLine(line);
        }
      });

      nimStream.on('end', () => {
        if (buffer.trim()) {
          consumeSseLine(
            buffer.trimEnd()
          );
        }

        // Some upstreams may close after the finish chunk without [DONE].
        sendDone();

        if (!res.writableEnded) {
          res.end();
        }
      });

      nimStream.on('error', err => {
        console.error(
          '[Text bridge stream error]',
          err.message
        );

        sendDone();

        if (!res.writableEnded) {
          res.end();
        }
      });

      return;
    }

    // Non-stream client response.
    // NVIDIA upstream still streams internally.
    res.setHeader(
      'Content-Type',
      'application/json'
    );

    res.setHeader(
      'Transfer-Encoding',
      'chunked'
    );

    res.setHeader(
      'Connection',
      'keep-alive'
    );

    res.flushHeaders();

    const keepalive =
      setInterval(() => {
        if (!res.writableEnded) {
          res.write(' ');
        }
      }, KEEPALIVE_INTERVAL);

    let nimStream;

    try {
      nimStream =
        await nimChatFetchWithRetry(
          nimModel,
          body,
          messages,
          'text-bridge'
        );
    } catch (err) {
      clearInterval(keepalive);

      console.error(
        '[Text bridge proxy error]',
        err.message
      );

      if (!res.writableEnded) {
        res.end(
          JSON.stringify({
            error: {
              message: err.message,
              code: err.status || 500,
            },
          })
        );
      }

      return;
    }

    clearInterval(keepalive);

    let collected;

    try {
      collected =
        await collectChatStream(
          nimStream
        );
    } catch (err) {
      console.error(
        '[Text bridge collect error]',
        err.message
      );

      if (!res.writableEnded) {
        res.end(
          JSON.stringify({
            error: {
              message: err.message,
              code: 500,
            },
          })
        );
      }

      return;
    }

    const choices =
      collected.choices.length
        ? collected.choices.map(
            choice => ({
              index:
                choice.index,

              text:
                choice.reasoning
                  ? `<think>\n${choice.reasoning}\n</think>\n\n${choice.content || ''}`
                  : (choice.content || ''),

              logprobs:
                null,

              finish_reason:
                choice.finish_reason,
            })
          )
        : [
            {
              index: 0,
              text: '',
              logprobs: null,
              finish_reason: 'stop',
            },
          ];

    if (!res.writableEnded) {
      res.end(
        JSON.stringify({
          id:
            `cmpl-${Date.now()}`,

          object:
            'text_completion',

          created:
            Math.floor(Date.now() / 1000),

          model:
            requestedModel,

          choices,

          usage:
            collected.usage || {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
            },
        })
      );
    }
  }
);

// ─────────────────────────────────────────────
// 404
// ─────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message:
        `${req.path} not found`,
      code: 404,
    },
  });
});

// ─────────────────────────────────────────────
// START
// ─────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(
    '\nOpenAI-compatible -> NVIDIA hosted NIM proxy'
  );

  console.log(
    `   Port:              ${PORT}`
  );

  console.log(
    `   Upstream:          ${NIM_API_BASE}/chat/completions`
  );

  console.log(
    `   Chat endpoint:     /v1/chat/completions`
  );

  console.log(
    `   Text endpoint:     /v1/completions (compatibility bridge)`
  );

  console.log(
    `   DeepSeek Pro ID:   ${DEEPSEEK_V4_PRO_ID}`
  );

  console.log(
    `   DeepSeek Flash ID: ${DEEPSEEK_V4_FLASH_ID}`
  );

  console.log(
    `   DeepSeek mode:     ${DEEPSEEK_REASONING_EFFORT}`
  );

  console.log(
    '   GLM-5.3 effort:    controlled by SillyTavern/client'
  );

  console.log(
    '   Response tokens:   controlled by SillyTavern/client'
  );

  console.log(
    `   First-token wait:  ${FIRST_TOKEN_TIMEOUT / 1000}s`
  );

  console.log(
    `   Idle timeout:      ${IDLE_TIMEOUT / 1000}s`
  );

  console.log(
    `   Keepalive every:   ${KEEPALIVE_INTERVAL / 1000}s`
  );

  console.log(
    '   Native NVIDIA /v1/completions: NO'
  );

  console.log(
    '   Keep-alive ping:   /ping\n'
  );
});
