import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// ─── Конфигурация ────────────────────────────────────────────────────────────

const GITHUB_TOKEN = process.env.GITHUB_TOKEN; // ваш GitHub PAT
const MODEL        = process.env.MODEL ?? "deepseek-ai/deepseek-v4-pro"; // замените при необходимости
const NIM_ENDPOINT = "https://models.github.ai/inference";             // GitHub Models → Nvidia NIM

if (!GITHUB_TOKEN) {
  console.error("❌  Переменная окружения GITHUB_TOKEN не задана. Создайте .env файл.");
  process.exit(1);
}

// ─── Вспомогательная функция: запрос к NIM ───────────────────────────────────

async function callNim(body) {
  const response = await fetch(`${NIM_ENDPOINT}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type":  "application/json",
      Authorization:   `Bearer ${GITHUB_TOKEN}`,
    },
    body: JSON.stringify({
      model: MODEL,
      ...body,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw Object.assign(new Error(`NIM вернул ${response.status}`), {
      status: response.status,
      upstream: text,
    });
  }

  return response; // возвращаем Response, чтобы поддерживать стриминг
}

// ─── POST /v1/chat/completions ────────────────────────────────────────────────

app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { stream = false, ...rest } = req.body;
    const upstream = await callNim({ ...rest, stream });

    if (stream) {
      // ── Стриминг ────────────────────────────────────────────────────────────
      res.setHeader("Content-Type",  "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection",    "keep-alive");

      const reader = upstream.body.getReader();
      const decoder = new TextDecoder();

      const pump = async () => {
        const { done, value } = await reader.read();
        if (done) { res.end(); return; }
        res.write(decoder.decode(value, { stream: true }));
        pump();
      };
      pump();
    } else {
      // ── Обычный ответ ────────────────────────────────────────────────────────
      const data = await upstream.json();
      res.json(data);
    }
  } catch (err) {
    console.error("Ошибка при обращении к NIM:", err.message);
    res.status(err.status ?? 500).json({
      error: {
        message:  err.message,
        upstream: err.upstream ?? null,
      },
    });
  }
});

// ─── GET /v1/models ───────────────────────────────────────────────────────────

app.get("/v1/models", async (_req, res) => {
  try {
    const upstream = await fetch(`${NIM_ENDPOINT}/models`, {
      headers: { Authorization: `Bearer ${GITHUB_TOKEN}` },
    });
    const data = await upstream.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── GET /health ──────────────────────────────────────────────────────────────

app.get("/health", (_req, res) =>
  res.json({ status: "ok", model: MODEL, endpoint: NIM_ENDPOINT })
);

// ─── Запуск ───────────────────────────────────────────────────────────────────

const PORT = process.env.PORT ?? 3000;
app.listen(PORT, () => {
  console.log(`✅  Прокси запущен: http://localhost:${PORT}`);
  console.log(`   Модель  : ${MODEL}`);
  console.log(`   Endpoint: ${NIM_ENDPOINT}`);
});
