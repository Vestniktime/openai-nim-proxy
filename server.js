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
var MAX_RETRIES     = parseInt(process.env.MAX_RETRIES     || '5',      10);
var RETRY_DELAY_MS  = parseInt(process.env.RETRY_DELAY_MS  || '10000',  10); // 10s base — 429s need time to clear
var MAX_CONCURRENT  = parseInt(process.env.MAX_CONCURRENT  || '3',      10); // max parallel NIM requests

var RETRYABLE_STATUSES = [429, 500, 502, 503, 504];

// ─────────────────────────────────────────────
//  Request queue — prevents hammering NIM API
//  and spreading 429s to all parallel requests
// ─────────────────────────────────────────────
var activeRequests = 0;
var requestQueue   = []; // [{fn, resolve, reject}]

function enqueue(fn) {
  return new Promise(function(resolve, reject) {
    requestQueue.push({ fn: fn, resolve: resolve, reject: reject });
    drainQueue();
  });
}

function drainQueue() {
  if (activeRequests >= MAX_CONCURRENT || requestQueue.length === 0) return;
  var item = requestQueue.shift();
  activeRequests++;
  item.fn()
    .then(function(result) {
      activeRequests--;
      item.resolve(result);
      drainQueue();
    })
    .catch(function(err) {
      activeRequests--;
      item.reject(err);
      drainQueue();
    });
}

if (!NIM_API_KEY) {
  console.warn('[warn] NIM_API_KEY is not set — add it as an environment variable on your platform.');
}

// ─────────────────────────────────────────────
//  Model Registry
// ─────────────────────────────────────────────
var MODEL_REGISTRY = {
  'deepseek-ai/deepseek-v4-pro': {
    nimId: 'deepseek-ai/deepseek-v4-pro', thinking: true,
    temperature: 0.6, max_tokens: 300000, top_p: 0.9,
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
//  Per-model rate limiter
//  Enforces a minimum gap between requests to
//  the same NIM model to stay under RPM limits.
//
//  NIM free tier: ~5 RPM on large models = 1 req / 12s
//  Set NIM_RPM=10 in env if you have a paid plan.
// ─────────────────────────────────────────────
var NIM_RPM           = parseInt(process.env.NIM_RPM || '5', 10);
var MIN_GAP_MS        = Math.ceil(60000 / NIM_RPM);   // e.g. 5 RPM → 12 000 ms
var lastRequestTime   = {};  // nimId → timestamp of last dispatched request
var modelQueues       = {};  // nimId → array of {fn, resolve, reject}
var modelTimers       = {};  // nimId → setTimeout handle

function rateLimitedEnqueue(nimId, fn) {
  return new Promise(function(resolve, reject) {
    if (!modelQueues[nimId]) modelQueues[nimId] = [];
    modelQueues[nimId].push({ fn: fn, resolve: resolve, reject: reject });
    scheduleNext(nimId);
  });
}

function scheduleNext(nimId) {
  if (modelTimers[nimId]) return;           // already scheduled
  if (!modelQueues[nimId] || modelQueues[nimId].length === 0) return;
  if (activeRequests >= MAX_CONCURRENT) {   // global cap still applies
    setTimeout(function() { scheduleNext(nimId); }, 500);
    return;
  }

  var now      = Date.now();
  var lastTime = lastRequestTime[nimId] || 0;
  var wait     = Math.max(0, lastTime + MIN_GAP_MS - now);

  modelTimers[nimId] = setTimeout(function() {
    modelTimers[nimId] = null;

    var item = modelQueues[nimId] && modelQueues[nimId].shift();
    if (!item) return;

    activeRequests++;
    lastRequestTime[nimId] = Date.now();

    if (modelQueues[nimId] && modelQueues[nimId].length > 0) {
      console.log('[ratelimit] ' + nimId + ': ' + modelQueues[nimId].length + ' request(s) still queued, next in ' + (MIN_GAP_MS / 1000) + 's');
    }

    item.fn()
      .then(function(result) {
        activeRequests--;
        item.resolve(result);
        scheduleNext(nimId);
      })
      .catch(function(err) {
        activeRequests--;
        // On 429: back off one extra gap before next attempt
        if (err.response && err.response.status === 429) {
          lastRequestTime[nimId] = Date.now() + MIN_GAP_MS;
        }
        item.reject(err);
        scheduleNext(nimId);
      });
  }, wait);

  if (wait > 0) {
    console.log('[ratelimit] ' + nimId + ': waiting ' + (wait / 1000).toFixed(1) + 's before next request (NIM_RPM=' + NIM_RPM + ')');
  }
}

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

// Safe JSON stringify — skips circular refs and non-serialisable values (e.g. axios agent)
function safeStringify(obj) {
  var seen = [];
  try {
    return JSON.stringify(obj, function(_key, value) {
      if (typeof value === 'object' && value !== null) {
        if (seen.indexOf(value) !== -1) return '[Circular]';
        seen.push(value);
      }
      if (typeof value === 'function') return '[Function]';
      return value;
    });
  } catch (e) {
    return '"[unserializable: ' + e.message + ']"';
  }
}

// Extract only the safe, useful fields from an axios error
function describeError(err) {
  // axios with responseType:'stream' puts a Readable stream in err.response.data.
  // We pull the status code and construct a helpful message directly.
  var status  = err.response ? err.response.status : undefined;
  var nimData = undefined;
  if (err.response && err.response.data) {
    var d = err.response.data;
    if (typeof d === 'string') {
      try { nimData = JSON.parse(d); } catch (e) { nimData = { raw: d }; }
    } else if (typeof d === 'object' && !d.pipe) {
      try { nimData = JSON.parse(safeStringify(d)); } catch (e) { nimData = { raw: String(d) }; }
    }
    // d.pipe = it's a stream; body already drained by axios internally on error
  }
  return {
    message:     err.message,
    code:        err.code,
    http_status: status,
    nim_error:   nimData
  };
}

function withRetry(fn, label) {
  var attempt = 0;
  function tryOnce() {
    return fn(attempt).catch(function(err) {
      var status = err.response ? err.response.status : undefined;
      var retryable = !status || RETRYABLE_STATUSES.indexOf(status) !== -1;
      if (!retryable || attempt >= MAX_RETRIES) throw err;
      // For 429: use Retry-After header if present, otherwise use a long fixed wait.
      // Do NOT do exponential backoff on 429 — it can make things worse by queuing retries.
      var delay;
      if (status === 429) {
        var ra = 0;
        if (err.response && err.response.headers && err.response.headers['retry-after']) {
          ra = parseInt(err.response.headers['retry-after'], 10);
        }
        delay = ra > 0 ? ra * 1000 : 60000; // 60s fixed — NIM rate limit windows are typically 60s
      } else {
        delay = Math.min(RETRY_DELAY_MS * Math.pow(2, attempt), 60000);
      }
      console.warn('[retry] ' + label + ' attempt ' + (attempt + 1) + '/' + MAX_RETRIES + ' — ' + (status || err.code) + '. Waiting ' + (delay / 1000) + 's. Queue: active=' + activeRequests + ' waiting=' + requestQueue.length);
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
  console.error('[proxy error]', safeStringify(detail));
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
    diagnostic_url:    '/api/test',
    rate_limiter: {
      nim_rpm:       NIM_RPM,
      min_gap_s:     MIN_GAP_MS / 1000,
      queued_per_model: (function() {
        var out = {};
        var keys = Object.keys(modelQueues);
        for (var i = 0; i < keys.length; i++) { if (modelQueues[keys[i]].length > 0) out[keys[i]] = modelQueues[keys[i]].length; }
        return out;
      })()
    },
    concurrency: { active: activeRequests, max: MAX_CONCURRENT }
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

  console.log('[proxy] ' + (req.body.model || '(none)') + ' -> ' + nimBody.model +
    ' | nim_stream=true | client_stream=' + clientWantsStream + ' | thinking=' + cfg.thinking);

  // For streaming clients: open SSE immediately + send heartbeats so
  //   Render/Railway don't kill the connection during NIM cold-start.
  // For non-streaming clients: buffer the NIM stream and return plain JSON.
  var heartbeatInterval = null;

  function cleanup() {
    if (heartbeatInterval) { clearInterval(heartbeatInterval); heartbeatInterval = null; }
  }

  if (clientWantsStream) {
    res.setHeader('Content-Type',  'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection',    'keep-alive');
    res.flushHeaders();
    heartbeatInterval = setInterval(function() {
      if (!res.writableEnded) res.write(': heartbeat\n\n');
    }, 5000);
  }

  res.on('close', cleanup);

  rateLimitedEnqueue(nimBody.model, function() {
    return withRetry(function() {
      return axios.post(NIM_API_BASE + '/chat/completions', nimBody, {
        headers: getNimHeaders(), responseType: 'stream', timeout: CONNECT_TIMEOUT
      });
    }, nimBody.model);
  })
  .then(function(upstream) {
    // Stop heartbeat — real data is coming.
    cleanup();

    // Abort upstream if client disconnects.
    res.on('close', function() {
      if (upstream.data.destroy) upstream.data.destroy();
    });

    var buffer    = '';
    var thinkOpen = false;

    // For non-streaming clients we accumulate chunks, then send one JSON blob.
    var assembledContent   = '';
    var assembledReasoning = '';
    var lastChunkData      = null;

    upstream.data.on('data', function(raw) {
      buffer += raw.toString();
      var lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        if (line.indexOf('data: ') !== 0) continue;
        var payload = line.slice(6).trim();
        if (payload === '[DONE]') {
          if (clientWantsStream) res.write('data: [DONE]\n\n');
          continue;
        }

        var data;
        try { data = JSON.parse(payload); } catch (e) {
          if (clientWantsStream) res.write(line + '\n');
          continue;
        }

        lastChunkData = data;
        var delta = data.choices && data.choices[0] && data.choices[0].delta;
        if (!delta) {
          if (clientWantsStream) res.write('data: ' + JSON.stringify(data) + '\n\n');
          continue;
        }

        var reasoning = delta.reasoning_content || '';
        var content   = delta.content           || '';
        delete delta.reasoning_content;

        // Accumulate for non-stream clients regardless
        if (reasoning) assembledReasoning += reasoning;
        if (content)   assembledContent   += content;

        if (clientWantsStream) {
          if (SHOW_REASONING && cfg.thinking) {
            var out = '';
            if (reasoning) { if (!thinkOpen) { out += '<think>\n'; thinkOpen = true; } out += reasoning; }
            if (content)   { if (thinkOpen)  { out += '\n</think>\n\n'; thinkOpen = false; } out += content; }
            delta.content = out;
          } else {
            delta.content = content;
          }
          if (!res.writableEnded) res.write('data: ' + JSON.stringify(data) + '\n\n');
        }
      }
    });

    upstream.data.on('end', function() {
      if (res.writableEnded) return;

      if (clientWantsStream) {
        if (thinkOpen && SHOW_REASONING) {
          res.write('data: ' + JSON.stringify(makeStreamChunk(nimBody.model, '\n</think>\n\n')) + '\n\n');
        }
        res.write('data: [DONE]\n\n');
        res.end();
      } else {
        // Non-stream client: return a standard application/json response.
        var fullContent = mergeContent(cfg.thinking ? assembledReasoning : null, assembledContent);
        res.setHeader('Content-Type', 'application/json');
        res.json({
          id:      (lastChunkData && lastChunkData.id) || ('chatcmpl-' + Date.now()),
          object:  'chat.completion',
          created: (lastChunkData && lastChunkData.created) || Math.floor(Date.now() / 1000),
          model:   req.body.model || nimBody.model,
          choices: [{ index: 0, message: { role: 'assistant', content: fullContent },
                      finish_reason: (lastChunkData && lastChunkData.choices && lastChunkData.choices[0] && lastChunkData.choices[0].finish_reason) || 'stop' }],
          usage:   (lastChunkData && lastChunkData.usage) || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
        });
      }
    });

    upstream.data.on('error', function(err) {
      console.error('[upstream error]', describeError(err));
      if (!res.writableEnded) res.end();
    });
  })
  .catch(function(err) {
    cleanup();
    console.error('[proxy error]', safeStringify(describeError(err)));
    if (res.writableEnded) return;

    var detail    = describeError(err);
    var status    = detail.http_status;
    var isTimeout = err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT' || err.code === 'STREAM_TIMEOUT' || status === 504;
    var is429     = status === 429;
    var message   = isTimeout
      ? 'Model ' + nimBody.model + ' did not respond in time after retries. Please try again in a moment.'
      : is429
        ? 'NIM API rate limit reached for model ' + nimBody.model + '. Wait 60s and retry, or switch to a less-loaded model (e.g. meta/llama-3.3-70b-instruct).'
        : ((detail.nim_error && (detail.nim_error.detail || (detail.nim_error.error && detail.nim_error.error.message))) || err.message || 'Proxy error');

    // Send error as SSE event (headers already sent)
    var errType    = is429 ? 'rate_limit_error' : isTimeout ? 'timeout_error' : 'api_error';
    var errPayload = { error: { message: message, type: errType, code: status || 500, model: nimBody.model } };
    if (clientWantsStream) {
      res.write('data: ' + JSON.stringify(errPayload) + '\n\n');
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      res.status(status || (isTimeout ? 504 : 500))
         .setHeader('Content-Type', 'application/json');
      res.json(errPayload);
    }
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
