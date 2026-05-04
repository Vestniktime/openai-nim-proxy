// server.js — Multi-model OpenAI-compatible proxy for NVIDIA NIM
// Works on Render, Railway, Fly.io and other platforms.
// Never calls process.exit() — all config errors reported via HTTP.
'use strict';

var express = require('express');
var cors    = require('cors');
var axios   = require('axios');

// ─────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────
var PORT          = process.env.PORT          || 3000;
var NIM_API_BASE  = process.env.NIM_API_BASE  || 'https://integrate.api.nvidia.com/v1';
var NIM_API_KEY   = process.env.NIM_API_KEY   || '';
var DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'deepseek-ai/deepseek-v4-pro';

var SHOW_REASONING  = process.env.SHOW_REASONING  !== 'false';
var STREAM_TIMEOUT  = parseInt(process.env.STREAM_TIMEOUT  || '600000', 10);
var CONNECT_TIMEOUT = parseInt(process.env.CONNECT_TIMEOUT || '120000', 10);
var MAX_RETRIES     = parseInt(process.env.MAX_RETRIES     || '3',      10);
var RETRY_DELAY_MS  = parseInt(process.env.RETRY_DELAY_MS  || '3000',   10);

var RETRYABLE_STATUSES = [429, 500, 502, 503, 504];

if (!NIM_API_KEY) {
  console.warn('[warn] NIM_API_KEY is not set — add it as an environment variable on your platform.');
}

// ─────────────────────────────────────────────
//  Model Registry
// ─────────────────────────────────────────────
var MODEL_REGISTRY = {
  'deepseek-ai/deepseek-v4-pro': {
    nimId: 'deepseek-ai/deepseek-v4-pro', thinking: true,
    temperature: 0.6, max_tokens: 8192, top_p: 0.9,
    aliases: ['deepseek-v4-pro', 'deepseek-v4', 'gpt-4o', 'gpt-4']
  },
  'deepseek-ai/deepseek-v4-flash': {
    nimId: 'deepseek-ai/deepseek-v4-flash', thinking: true,
    temperature: 0.6, max_tokens: 8192, top_p: 0.9,
    aliases: ['deepseek-v4-flash', 'deepseek-flash']
  },
  'deepseek-ai/deepseek-v3.2': {
    nimId: 'deepseek-ai/deepseek-v3.2', thinking: false,
    temperature: 0.6, max_tokens: 8192, top_p: 0.9,
    aliases: ['deepseek-v3', 'deepseek-v3.2']
  },
  'qwen/qwen3.5-397b-a17b': {
    nimId: 'qwen/qwen3.5-397b-a17b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3.5-397b', 'qwen3.5', 'qwen-large']
  },
  'qwen/qwen3.5-122b-a10b': {
    nimId: 'qwen/qwen3.5-122b-a10b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3.5-122b', 'qwen-mid']
  },
  'qwen/qwen3-coder-480b-a35b-instruct': {
    nimId: 'qwen/qwen3-coder-480b-a35b-instruct', thinking: false,
    temperature: 0.2, max_tokens: 16384, top_p: 0.95,
    aliases: ['qwen3-coder', 'qwen-coder', 'qwen3-coder-480b']
  },
  'qwen/qwen3-next-80b-a3b-thinking': {
    nimId: 'qwen/qwen3-next-80b-a3b-thinking', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3-thinking', 'qwen3-80b-thinking']
  },
  'qwen/qwen3-next-80b-a3b-instruct': {
    nimId: 'qwen/qwen3-next-80b-a3b-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['qwen3-80b', 'qwen3-instruct']
  },
  'meta/llama-3.3-70b-instruct': {
    nimId: 'meta/llama-3.3-70b-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['llama-3.3-70b', 'llama-70b', 'gpt-3.5-turbo']
  },
  'meta/llama-3.1-405b-instruct': {
    nimId: 'meta/llama-3.1-405b-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['llama-405b', 'llama-3.1-405b']
  },
  'meta/llama-4-maverick-17b-128e-instruct': {
    nimId: 'meta/llama-4-maverick-17b-128e-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['llama-4-maverick', 'llama4', 'gpt-3.5-turbo-16k']
  },
  'nvidia/llama-3.1-nemotron-ultra-253b-v1': {
    nimId: 'nvidia/llama-3.1-nemotron-ultra-253b-v1', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['nemotron-ultra', 'llama-nemotron-ultra']
  },
  'nvidia/llama-3.3-nemotron-super-49b-v1': {
    nimId: 'nvidia/llama-3.3-nemotron-super-49b-v1', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['nemotron-super', 'nemotron-49b']
  },
  'mistralai/mistral-large-3-675b-instruct-2512': {
    nimId: 'mistralai/mistral-large-3-675b-instruct-2512', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['mistral-large-3', 'mistral-large', 'claude-3-opus']
  },
  'mistralai/mistral-medium-3.5-128b': {
    nimId: 'mistralai/mistral-medium-3.5-128b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['mistral-medium', 'mistral-medium-3.5']
  },
  'moonshotai/kimi-k2-instruct': {
    nimId: 'moonshotai/kimi-k2-instruct', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['kimi-k2', 'moonshot-kimi']
  },
  'moonshotai/kimi-k2-thinking': {
    nimId: 'moonshotai/kimi-k2-thinking', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['kimi-k2-thinking', 'kimi-thinking']
  },
  'openai/gpt-oss-120b': {
    nimId: 'openai/gpt-oss-120b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['gpt-oss-120b']
  },
  'openai/gpt-oss-20b': {
    nimId: 'openai/gpt-oss-20b', thinking: false,
    temperature: 0.7, max_tokens: 8192, top_p: 0.9,
    aliases: ['gpt-oss-20b', 'claude-3-sonnet', 'claude-3-haiku']
  }
};

// ─────────────────────────────────────────────
//  Build alias lookup map
// ─────────────────────────────────────────────
var ALIAS_MAP = {};
var _allModels = Object.values(MODEL_REGISTRY);
for (var _mi = 0; _mi < _allModels.length; _mi++) {
  var _mc = _allModels[_mi];
  ALIAS_MAP[_mc.nimId.toLowerCase()] = _mc;
  if (_mc.aliases) {
    for (var _ai = 0; _ai < _mc.aliases.length; _ai++) {
      ALIAS_MAP[_mc.aliases[_ai].toLowerCase()] = _mc;
    }
  }
}
var DEFAULT_CONFIG = ALIAS_MAP[DEFAULT_MODEL.toLowerCase()] || _allModels[0];

// ─────────────────────────────────────────────
//  App setup
// ─────────────────────────────────────────────
var app = express();
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'x-api-key']
}));
app.options('*', cors());  // explicit preflight
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

function sleep(ms) {
  return new Promise(function(resolve) { setTimeout(resolve, ms); });
}

function resolveConfig(requested) {
  if (!requested) return DEFAULT_CONFIG;
  var found = ALIAS_MAP[requested.toLowerCase()];
  if (!found) {
    console.warn('[proxy] Unknown model "' + requested + '" — falling back to ' + DEFAULT_CONFIG.nimId);
    return DEFAULT_CONFIG;
  }
  return found;
}

function buildNimBody(reqBody) {
  var cfg = resolveConfig(reqBody.model);
  var body = {
    model:       cfg.nimId,
    messages:    reqBody.messages,
    temperature: reqBody.temperature !== undefined ? reqBody.temperature : cfg.temperature,
    max_tokens:  reqBody.max_tokens  !== undefined ? reqBody.max_tokens  : cfg.max_tokens,
    top_p:       reqBody.top_p       !== undefined ? reqBody.top_p       : cfg.top_p,
    stream:      true
  };
  if (cfg.thinking) {
    body.extra_body = { chat_template_kwargs: { thinking: true } };
  }
  return { body: body, cfg: cfg };
}

function mergeContent(reasoningContent, content) {
  if (!SHOW_REASONING || !reasoningContent) return content || '';
  return '<think>\n' + reasoningContent + '\n</think>\n\n' + (content || '');
}

function makeStreamChunk(model, content) {
  return {
    id: 'chatcmpl-inject-' + Date.now(),
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{ index: 0, delta: { content: content }, finish_reason: null }]
  };
}

function getNimHeaders() {
  return {
    'Authorization': 'Bearer ' + NIM_API_KEY,
    'Content-Type':  'application/json'
  };
}

function describeError(err) {
  return {
    message:     err.message,
    code:        err.code,
    http_status: err.response ? err.response.status : undefined,
    nim_error:   err.response ? err.response.data   : undefined
  };
}

function withRetry(fn, label) {
  var attempt = 0;
  function tryOnce() {
    return fn(attempt).catch(function(err) {
      var status = err.response ? err.response.status : undefined;
      var retryable = !status || RETRYABLE_STATUSES.indexOf(status) !== -1;
      if (!retryable || attempt >= MAX_RETRIES) throw err;
      var delay = Math.min(RETRY_DELAY_MS * Math.pow(2, attempt), 30000);
      if (status === 429 && err.response.headers && err.response.headers['retry-after']) {
        var ra = parseInt(err.response.headers['retry-after'], 10);
        if (ra > 0) delay = ra * 1000;
      }
      console.warn('[retry] ' + label + ' attempt ' + (attempt + 1) + '/' + MAX_RETRIES + ' — ' + (status || err.code) + '. Waiting ' + (delay / 1000) + 's');
      attempt++;
      return sleep(delay).then(tryOnce);
    });
  }
  return tryOnce();
}

function consumeStream(nimStream) {
  return new Promise(function(resolve, reject) {
    var buffer       = '';
    var contentAcc   = '';
    var reasoningAcc = '';
    var lastData     = null;

    nimStream.on('data', function(raw) {
      buffer += raw.toString();
      var lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        if (line.indexOf('data: ') !== 0) continue;
        var payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;
        try {
          var parsed = JSON.parse(payload);
          lastData   = parsed;
          var delta  = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
          if (delta) {
            if (delta.reasoning_content) reasoningAcc += delta.reasoning_content;
            if (delta.content)           contentAcc   += delta.content;
          }
        } catch (e) { /* skip malformed chunk */ }
      }
    });

    nimStream.on('end', function() {
      resolve({
        id:                (lastData && lastData.id)      || ('chatcmpl-' + Date.now()),
        created:           (lastData && lastData.created) || Math.floor(Date.now() / 1000),
        model:             lastData && lastData.model,
        usage:             (lastData && lastData.usage)   || null,
        role:              'assistant',
        content:           contentAcc,
        reasoning_content: reasoningAcc || null,
        finish_reason:     (lastData && lastData.choices && lastData.choices[0] && lastData.choices[0].finish_reason) || 'stop'
      });
    });

    nimStream.on('error', reject);
  });
}

function sendError(res, err, model) {
  var detail    = describeError(err);
  var status    = detail.http_status;
  var isTimeout = err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT' || err.code === 'STREAM_TIMEOUT' || status === 504;
  console.error('[proxy error]', JSON.stringify(detail));
  var message = isTimeout
    ? 'Model ' + model + ' did not respond in time. It may be cold-starting — please retry.'
    : ((detail.nim_error && (detail.nim_error.detail || (detail.nim_error.error && detail.nim_error.error.message))) || err.message || 'Proxy error');
  if (res.headersSent) { res.end(); return; }
  var type = status === 429 ? 'rate_limit_error' : isTimeout ? 'timeout_error' : status >= 500 ? 'api_error' : 'invalid_request_error';
  res.status(status || (isTimeout ? 504 : 500)).json({
    error: { message: message, type: type, code: status || 500, model: model, nim_detail: detail.nim_error, tip: 'Visit /api/test to diagnose.' }
  });
}

function requireApiKey(req, res, next) {
  if (!NIM_API_KEY) {
    return res.status(401).json({
      error: { message: 'NIM_API_KEY is not configured. Add it as an environment variable.', type: 'auth_error', code: 401 }
    });
  }
  next();
}

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────

app.get('/health', function(_req, res) {
  res.json({
    status:            'ok',
    service:           'NVIDIA NIM Multi-Model Proxy',
    node_version:      process.version,
    api_key_set:       !!NIM_API_KEY,
    api_key_prefix:    NIM_API_KEY ? NIM_API_KEY.slice(0, 8) + '...' : null,
    default_model:     DEFAULT_CONFIG.nimId,
    show_reasoning:    SHOW_REASONING,
    models_loaded:     Object.keys(MODEL_REGISTRY).length,
    stream_timeout_s:  STREAM_TIMEOUT / 1000,
    connect_timeout_s: CONNECT_TIMEOUT / 1000,
    diagnostic_url:    '/api/test'
  });
});

app.get('/v1/models', function(_req, res) {
  var seen = {};
  var data = [];
  var cfgs = Object.values(MODEL_REGISTRY);
  for (var i = 0; i < cfgs.length; i++) {
    var cfg = cfgs[i];
    var ids = [cfg.nimId].concat(cfg.aliases || []);
    for (var j = 0; j < ids.length; j++) {
      if (seen[ids[j]]) continue;
      seen[ids[j]] = true;
      data.push({ id: ids[j], object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'nvidia-nim-proxy', root: cfg.nimId });
    }
  }
  res.json({ object: 'list', data: data });
});

app.get('/api/test', function(req, res) {
  var modelToTest = req.query.model || DEFAULT_CONFIG.nimId;
  var report = {
    timestamp:    new Date().toISOString(),
    node_version: process.version,
    proxy_config: {
      nim_api_base:   NIM_API_BASE,
      api_key_set:    !!NIM_API_KEY,
      api_key_prefix: NIM_API_KEY ? NIM_API_KEY.slice(0, 8) + '...' : '(not set)',
      default_model:  DEFAULT_CONFIG.nimId
    },
    steps: []
  };

  if (!NIM_API_KEY) {
    report.verdict = 'NIM_API_KEY is not set. Add it as an environment variable on your platform.';
    return res.json(report);
  }

  var t0 = Date.now();
  axios.get(NIM_API_BASE + '/models', { headers: getNimHeaders(), timeout: CONNECT_TIMEOUT })
    .then(function() {
      report.steps.push({ step: '1_connectivity', status: 'ok', ms: Date.now() - t0 });
      var t1 = Date.now();
      return axios.get(NIM_API_BASE + '/models', { headers: getNimHeaders(), timeout: CONNECT_TIMEOUT });
    })
    .then(function(r) {
      var availableIds = (r.data && r.data.data ? r.data.data : []).map(function(m) { return m.id; });
      report.steps.push({ step: '2_auth_and_models', status: 'ok', available_model_count: availableIds.length });
      if (availableIds.length > 0 && availableIds.indexOf(modelToTest) === -1) {
        report.steps.push({ step: '3_model_availability', status: 'warn', detail: '"' + modelToTest + '" not found in your account.' });
      } else {
        report.steps.push({ step: '3_model_availability', status: 'ok', model: modelToTest });
      }
      var t2  = Date.now();
      var cfg = resolveConfig(modelToTest);
      var reqBody = { model: cfg.nimId, messages: [{ role: 'user', content: 'Reply with one word: OK' }], max_tokens: 10, temperature: 0, stream: true };
      if (cfg.thinking) reqBody.extra_body = { chat_template_kwargs: { thinking: true } };
      return axios.post(NIM_API_BASE + '/chat/completions', reqBody, { headers: getNimHeaders(), responseType: 'stream', timeout: CONNECT_TIMEOUT })
        .then(function(upstream) {
          return Promise.race([
            consumeStream(upstream.data),
            new Promise(function(_, rej) { setTimeout(function() { rej(new Error('stream_timeout')); }, 60000); })
          ]);
        })
        .then(function(assembled) {
          report.steps.push({ step: '4_inference_stream', status: 'ok', ms: Date.now() - t2, response: assembled.content });
          report.verdict = 'Model "' + modelToTest + '" is working. Response in ~' + (Date.now() - t2) + 'ms.';
          res.json(report);
        });
    })
    .catch(function(err) {
      var d = describeError(err);
      report.steps.push({ step: 'failed', detail: d });
      report.verdict = err.message === 'stream_timeout'
        ? 'Model "' + modelToTest + '" timed out (60s). Try again in 30s.'
        : 'Failed: HTTP ' + (d.http_status || err.code || err.message);
      res.json(report);
    });
});

app.post('/v1/chat/completions', requireApiKey, function(req, res) {
  var clientWantsStream = !!req.body.stream;
  var built   = buildNimBody(req.body);
  var nimBody = built.body;
  var cfg     = built.cfg;

  console.log('[proxy] ' + (req.body.model || '(none)') + ' -> ' + nimBody.model + ' | stream=' + clientWantsStream + ' | thinking=' + cfg.thinking);

  withRetry(function() {
    return axios.post(NIM_API_BASE + '/chat/completions', nimBody, {
      headers: getNimHeaders(), responseType: 'stream', timeout: CONNECT_TIMEOUT
    });
  }, nimBody.model)
  .then(function(upstream) {
    if (clientWantsStream) {
      res.setHeader('Content-Type',  'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');
      res.flushHeaders();

      var buffer    = '';
      var thinkOpen = false;

      upstream.data.on('data', function(raw) {
        buffer += raw.toString();
        var lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (line.indexOf('data: ') !== 0) continue;
          var payload = line.slice(6).trim();
          if (payload === '[DONE]') { res.write('data: [DONE]\n\n'); continue; }
          var data;
          try { data = JSON.parse(payload); } catch (e) { res.write(line + '\n'); continue; }
          var delta = data.choices && data.choices[0] && data.choices[0].delta;
          if (!delta) { res.write('data: ' + JSON.stringify(data) + '\n\n'); continue; }
          var reasoning = delta.reasoning_content || '';
          var content   = delta.content           || '';
          delete delta.reasoning_content;
          if (SHOW_REASONING && cfg.thinking) {
            var out = '';
            if (reasoning) { if (!thinkOpen) { out += '<think>\n'; thinkOpen = true; } out += reasoning; }
            if (content)   { if (thinkOpen)  { out += '\n</think>\n\n'; thinkOpen = false; } out += content; }
            delta.content = out;
          } else {
            delta.content = content;
          }
          res.write('data: ' + JSON.stringify(data) + '\n\n');
        }
      });

      upstream.data.on('end', function() {
        if (thinkOpen && SHOW_REASONING) {
          res.write('data: ' + JSON.stringify(makeStreamChunk(nimBody.model, '\n</think>\n\n')) + '\n\n');
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });

      upstream.data.on('error', function(err) {
        console.error('[stream error]', describeError(err));
        if (!res.headersSent) res.end();
      });

      res.on('close', function() { if (upstream.data.destroy) upstream.data.destroy(); });

    } else {
      var th;
      var timeoutErr = new Error('stream_timeout');
      timeoutErr.code = 'STREAM_TIMEOUT';
      Promise.race([
        consumeStream(upstream.data),
        new Promise(function(_, rej) { th = setTimeout(function() { rej(timeoutErr); }, STREAM_TIMEOUT); })
      ])
      .then(function(assembled) {
        clearTimeout(th);
        res.json({
          id:      assembled.id,
          object:  'chat.completion',
          created: assembled.created,
          model:   req.body.model || assembled.model || nimBody.model,
          choices: [{ index: 0, message: { role: assembled.role, content: mergeContent(cfg.thinking ? assembled.reasoning_content : null, assembled.content) }, finish_reason: assembled.finish_reason }],
          usage:   assembled.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
        });
      })
      .catch(function(err) {
        clearTimeout(th);
        if (upstream.data.destroy) upstream.data.destroy();
        sendError(res, err, nimBody.model);
      });
    }
  })
  .catch(function(err) {
    sendError(res, err, nimBody.model);
  });
});

app.all('*', function(req, res) {
  res.status(404).json({ error: { message: 'Endpoint ' + req.method + ' ' + req.path + ' not found', type: 'invalid_request_error', code: 404 } });
});

// ─────────────────────────────────────────────
//  Start
// ─────────────────────────────────────────────
app.listen(PORT, function() {
  console.log('\n🚀  NVIDIA NIM Multi-Model Proxy — port ' + PORT);
  console.log('    Node.js        : ' + process.version);
  console.log('    API key set    : ' + (NIM_API_KEY ? 'YES (' + NIM_API_KEY.slice(0, 8) + '...)' : 'NO — set NIM_API_KEY env var'));
  console.log('    Default model  : ' + DEFAULT_CONFIG.nimId);
  console.log('    Show reasoning : ' + (SHOW_REASONING ? 'ON' : 'OFF'));
  console.log('    Stream timeout : ' + (STREAM_TIMEOUT / 1000) + 's');
  console.log('    Models loaded  : ' + Object.keys(MODEL_REGISTRY).length);
  console.log('    Health         : http://localhost:' + PORT + '/health');
  console.log('    Diagnostic     : http://localhost:' + PORT + '/api/test');
  console.log('');
});

// ─────────────────────────────────────────────
//  Global safety nets — keep the process alive
//  and log what caused a "failed to fetch"
// ─────────────────────────────────────────────
process.on('uncaughtException', function(err) {
  console.error('[uncaughtException]', err.message, err.stack);
  // Do NOT exit — let Render/Railway restart if needed via health check
});

process.on('unhandledRejection', function(reason) {
  console.error('[unhandledRejection]', reason && reason.message ? reason.message : reason);
});
