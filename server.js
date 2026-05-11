import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Middleware ────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());

// ─── Config ───────────────────────────────────────────────────────────────────
const NIM_API_KEY  = process.env.NIM_API_KEY  || "";   // nvapi-xxxx
const NIM_BASE_URL = process.env.NIM_BASE_URL || "https://integrate.api.nvidia.com/v1";
const MODEL        = process.env.MODEL        || "deepseek-ai/deepseek-r1";

if (!NIM_API_KEY) {
  console.error("⛔  NIM_API_KEY is not set. Create a .env file or set the env variable.");
  process.exit(1);
}

// ─── Helper: forward to NIM ───────────────────────────────────────────────────
async function callNim(body) {
  const response = await fetch(`${NIM_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type":  "application/json",
      Authorization:   `Bearer ${NIM_API_KEY}`,
    },
    body: JSON.stringify({ model: MODEL, ...body }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw Object.assign(new Error(err), { status: response.status });
  }

  return response;
}

// ─── Routes ───────────────────────────────────────────────────────────────────

/**
 * Health check
 * GET /health
 */
app.get("/health", (_req, res) => {
  res.json({ status: "ok", model: MODEL, base: NIM_BASE_URL });
});

/**
 * Chat completions proxy  (non-streaming)
 * POST /v1/chat/completions
 *
 * Body: standard OpenAI-compatible chat completions request
 * { messages: [...], temperature?, max_tokens?, ... }
 */
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const upstream = await callNim({ ...req.body, stream: false });
    const data     = await upstream.json();
    res.json(data);
  } catch (err) {
    const status = err.status ?? 500;
    res.status(status).json({ error: err.message });
  }
});

/**
 * Streaming chat completions proxy
 * POST /v1/chat/completions/stream
 *
 * Returns SSE chunks forwarded directly from NIM.
 */
app.post("/v1/chat/completions/stream", async (req, res) => {
  try {
    const upstream = await callNim({ ...req.body, stream: true });

    res.setHeader("Content-Type",  "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection",    "keep-alive");

    // Pipe the readable stream from NIM → client
    const reader  = upstream.body.getReader();
    const decoder = new TextDecoder();

    const pump = async () => {
      const { done, value } = await reader.read();
      if (done) { res.end(); return; }
      res.write(decoder.decode(value, { stream: true }));
      await pump();
    };

    await pump();
  } catch (err) {
    const status = err.status ?? 500;
    if (!res.headersSent) {
      res.status(status).json({ error: err.message });
    } else {
      res.write(`data: {"error":"${err.message}"}\n\n`);
      res.end();
    }
  }
});

/**
 * Models list  (returns a static entry for the configured model)
 * GET /v1/models
 */
app.get("/v1/models", (_req, res) => {
  res.json({
    object: "list",
    data: [
      {
        id:       MODEL,
        object:   "model",
        owned_by: "nvidia-nim-proxy",
      },
    ],
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🟢  NIM Proxy running on http://localhost:${PORT}`);
  console.log(`    Model  : ${MODEL}`);
  console.log(`    NIM    : ${NIM_BASE_URL}\n`);
});
