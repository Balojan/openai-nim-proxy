// server.js - NVIDIA NIM Proxy for Janitor AI
// Plugin-enabled: control thinking/reasoning from the system prompt
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ──────────────────────────────────────────────
// GLOBAL DEFAULTS (used when no plugin tags found)
// ──────────────────────────────────────────────
const SHOW_REASONING = false;       // global fallback
const ENABLE_THINKING_MODE = false; // global fallback

// ──────────────────────────────────────────────
// MODEL MAPPING — Janitor AI model name → NVIDIA NIM model ID
// These are the ACTUAL NIM model IDs, not OpenAI names.
// Janitor AI sends model names like "gpt-4", "claude-3-opus", etc.
// We map those to real NIM models available on NVIDIA's API.
// ──────────────────────────────────────────────
const MODEL_MAPPING = {
  // OpenAI-style aliases (Janitor AI sends these)
  'gpt-3.5-turbo':       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4':               'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo':         'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o':              'deepseek-ai/deepseek-v3.1',
  'gpt-4o-mini':         'deepseek-ai/deepseek-v4-flash',

  // Claude-style aliases
  'claude-3-opus':       'openai/gpt-oss-120b',
  'claude-3-sonnet':     'openai/gpt-oss-20b',
  'claude-3-haiku':      'meta/llama-3.1-8b-instruct',

  // Gemini-style aliases
  'gemini-pro':          'qwen/qwen3-next-80b-a3b-thinking',
  'gemini-1.5-pro':      'qwen/qwen3-next-80b-a3b-thinking',

  // DeepSeek aliases
  'deepseek-v4':         'deepseek-ai/deepseek-v4',
  'deepseek-v4-flash':   'deepseek-ai/deepseek-v4-flash',
  'deepseek-v3.1':      'deepseek-ai/deepseek-v3.1',
  'deepseek-chat':       'deepseek-ai/deepseek-v3.1',

  // GLM aliases
  'glm-5.1':             'z-ai/glm-5.1',
  'glm-4':               'z-ai/glm-5.1',

  // Moonshot / Kimi aliases
  'kimi-k2':             'moonshotai/kimi-k2-instruct-0905',
  'kimi-k2.6':           'moonshotai/kimi-k2.6',

  // Meta Llama direct aliases
  'llama-3.1-8b':        'meta/llama-3.1-8b-instruct',
  'llama-3.1-70b':       'meta/llama-3.1-70b-instruct',
  'llama-3.1-405b':      'meta/llama-3.1-405b-instruct',
  'llama-3.3-70b':       'meta/llama-3.3-70b-instruct',

  // NVIDIA Nemotron aliases
  'nemotron-ultra':      'nvidia/llama-3.1-nemotron-ultra-253b-v1',

  // Qwen aliases
  'qwen3-coder':         'qwen/qwen3-coder-480b-a35b-instruct',
  'qwen3-next':          'qwen/qwen3-next-80b-a3b-thinking',
};

// Reverse mapping for logging/debugging (NIM ID → friendly name)
const REVERSE_MODEL_MAP = Object.fromEntries(
  Object.entries(MODEL_MAPPING).map(([k, v]) => [v, k])
);

// ──────────────────────────────────────────────
// PLUGIN PARSER – reads //SHOW_THINKING// and
// //SHOW_REASONING// from system messages, then
// strips them before forwarding to the model.
// ──────────────────────────────────────────────
function parsePlugins(messages) {
  let showReasoning = SHOW_REASONING;
  let enableThinking = ENABLE_THINKING_MODE;

  if (!messages || !Array.isArray(messages)) {
    return { messages, showReasoning, enableThinking };
  }

  // Deep-clone to avoid mutating the original request
  const cleanedMessages = JSON.parse(JSON.stringify(messages));

  for (let i = 0; i < cleanedMessages.length; i++) {
    const msg = cleanedMessages[i];
    if (msg.role === 'system' && typeof msg.content === 'string') {
      // Check for SHOW_REASONING tag
      if (msg.content.includes('//SHOW_REASONING//')) {
        showReasoning = true;
        msg.content = msg.content.replace(/\/\/SHOW_REASONING\/\//g, '');
      }
      // Check for SHOW_THINKING tag
      if (msg.content.includes('//SHOW_THINKING//')) {
        enableThinking = true;
        msg.content = msg.content.replace(/\/\/SHOW_THINKING\/\//g, '');
      }
      // Clean up leftover whitespace/newlines
      msg.content = msg.content.replace(/\n{3,}/g, '\n\n').trim();
    }
  }

  return { messages: cleanedMessages, showReasoning, enableThinking };
}

// ──────────────────────────────────────────────
// FALLBACK MODEL RESOLVER
// Tries to find the best NIM model when mapping fails.
// ──────────────────────────────────────────────
async function resolveModel(model, apiKey, apiBase) {
  // 1. Direct mapping hit
  if (MODEL_MAPPING[model]) {
    return { modelId: MODEL_MAPPING[model], source: 'mapped' };
  }

  // 2. If the user already sent a valid NIM model ID (contains "/"), test it directly
  if (model.includes('/')) {
    try {
      const test = await axios.post(
        `${apiBase}/chat/completions`,
        {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        },
        {
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
          },
          validateStatus: (status) => status < 500,
          timeout: 10000
        }
      );
      if (test.status >= 200 && test.status < 300) {
        return { modelId: model, source: 'direct-nim' };
      }
    } catch (e) {
      // Direct test failed, fall through to heuristic
    }
  }

  // 3. Heuristic fallback based on model name hints
  const lower = model.toLowerCase();

  if (lower.includes('405b') || lower.includes('opus') || lower.includes('ultra') || lower.includes('large')) {
    return { modelId: 'meta/llama-3.1-405b-instruct', source: 'heuristic-large' };
  }
  if (lower.includes('70b') || lower.includes('sonnet') || lower.includes('pro')) {
    return { modelId: 'meta/llama-3.1-70b-instruct', source: 'heuristic-medium' };
  }
  if (lower.includes('8b') || lower.includes('haiku') || lower.includes('mini') || lower.includes('flash')) {
    return { modelId: 'meta/llama-3.1-8b-instruct', source: 'heuristic-small' };
  }

  // 4. Ultimate fallback
  return { modelId: 'meta/llama-3.1-8b-instruct', source: 'fallback' };
}

// ──────────────────────────────────────────────
// ROUTES
// ──────────────────────────────────────────────

app.get('/', (req, res) => {
  res.send('NVIDIA NIM Proxy is running. Health check at /health');
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'NVIDIA NIM Proxy for Janitor AI (plugin-enabled)',
    version: '2.0.0',
    timestamp: new Date().toISOString()
  });
});

// GET fallback for Janitor AI's pre-check / uptime robot
app.get('/v1/chat/completions', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Completions endpoint. Use POST to chat.',
    available: true
  });
});

// ──────────────────────────────────────────────
// MAIN PROXY – POST /v1/chat/completions
// ──────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const requestStart = Date.now();

  try {
    const { model, messages, temperature, max_tokens, stream, top_p, frequency_penalty, presence_penalty } = req.body;

    // ── Plugin injection ──────────────────────
    const pluginResult = parsePlugins(messages);
    const showReasoning = pluginResult.showReasoning;
    const enableThinking = pluginResult.enableThinking;
    const cleanedMessages = pluginResult.messages;
    // ──────────────────────────────────────────

    // ── Model resolution ──────────────────────
    const { modelId: nimModel, source: modelSource } = await resolveModel(model, NIM_API_KEY, NIM_API_BASE);
    console.log(`[${new Date().toISOString()}] Model resolved: "${model}" → "${nimModel}" (source: ${modelSource})`);
    // ──────────────────────────────────────────

    // ── Build NIM request ─────────────────────
    const nimRequest = {
      model: nimModel,
      messages: cleanedMessages,
      temperature: temperature !== undefined ? temperature : 0.6,
      max_tokens: max_tokens !== undefined ? max_tokens : 9024,
      stream: stream || false
    };

    // Optional OpenAI-compatible parameters (pass through if provided)
    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;

    // ── Thinking / Reasoning configuration ──────
    // CORRECT FORMAT per your requirement:
    // extra_body:
    //   thinking:
    //     type: disabled
    //   chat_template_kwargs:
    //     thinking: false
    // ──────────────────────────────────────────
    if (enableThinking) {
      // User explicitly wants thinking enabled via //SHOW_THINKING//
      nimRequest.extra_body = {
        thinking: { type: "enabled" },
        chat_template_kwargs: { thinking: true }
      };
    } else {
      // Default: explicitly DISABLE thinking to prevent unwanted reasoning tokens
      nimRequest.extra_body = {
        thinking: { type: "disabled" },
        chat_template_kwargs: { thinking: false }
      };
    }
    // ──────────────────────────────────────────

    // ── Send to NVIDIA NIM ────────────────────
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 300000 // 5 minute timeout for long generations
      }
    );

    // ── STREAMING RESPONSE ────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering

      let buffer = '';
      let reasoningStarted = false;
      let hasError = false;

      response.data.on('data', (chunk) => {
        if (hasError) return;
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));

            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content;
              const content = data.choices[0].delta.content;

              if (showReasoning) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent = '\u003cthink\u003e\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += '\n\u003c/think\u003e\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                if (combinedContent) {
                  data.choices[0].delta.content = combinedContent;
                  delete data.choices[0].delta.reasoning_content;
                }
              } else {
                // Don't show reasoning — only pass through content
                if (content) {
                  data.choices[0].delta.content = content;
                } else {
                  data.choices[0].delta.content = '';
                }
                delete data.choices[0].delta.reasoning_content;
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            // If JSON parse fails, forward raw line to avoid breaking stream
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => {
        res.write('data: [DONE]\n\n');
        res.end();
        const duration = Date.now() - requestStart;
        console.log(`[${new Date().toISOString()}] Stream completed in ${duration}ms`);
      });

      response.data.on('error', (err) => {
        hasError = true;
        console.error('Stream error:', err.message);
        res.write(`data: ${JSON.stringify({ error: { message: 'Stream error: ' + err.message } })}\n\n`);
        res.end();
      });

      // Handle client disconnect
      req.on('close', () => {
        if (!res.writableEnded) {
          console.log(`[${new Date().toISOString()}] Client disconnected, ending stream`);
          res.end();
        }
      });

    // ── NON-STREAMING RESPONSE ────────────────
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model, // Return the ORIGINAL model name Janitor AI sent
        choices: response.data.choices.map(choice => {
          let content = choice.message?.content || '';

          if (showReasoning && choice.message?.reasoning_content) {
            content = '\u003cthink\u003e\n' + choice.message.reasoning_content + '\n\u003c/think\u003e\n\n' + content;
          }

          return {
            index: choice.index,
            message: { role: choice.message.role, content },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
      const duration = Date.now() - requestStart;
      console.log(`[${new Date().toISOString()}] Request completed in ${duration}ms`);
    }

  } catch (error) {
    const duration = Date.now() - requestStart;
    console.error(`[${new Date().toISOString()}] Proxy error after ${duration}ms:`, error.message);

    // Don't crash — return a proper OpenAI-style error
    const statusCode = error.response?.status || 500;
    const errorMessage = error.response?.data?.error?.message 
      || error.response?.data?.message 
      || error.message 
      || 'Internal server error';

    res.status(statusCode).json({
      error: {
        message: errorMessage,
        type: statusCode >= 500 ? 'internal_server_error' : 'invalid_request_error',
        code: statusCode
      }
    });
  }
});

// ──────────────────────────────────────────────
// MODEL LIST ENDPOINT (for debugging)
// ──────────────────────────────────────────────
app.get('/v1/models', (req, res) => {
  const models = Object.entries(MODEL_MAPPING).map(([id, nimId]) => ({
    id,
    object: 'model',
    owned_by: nimId.split('/')[0],
    nim_model_id: nimId
  }));
  res.json({ object: 'list', data: models });
});

// Catch-all 404
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.method} ${req.path} not found` }
  });
});

app.listen(PORT, () => {
  console.log(`═══════════════════════════════════════════════`);
  console.log(`  NVIDIA NIM Proxy for Janitor AI`);
  console.log(`  Version: 2.0.0`);
  console.log(`  Port: ${PORT}`);
  console.log(`  API Base: ${NIM_API_BASE}`);
  console.log(`  API Key: ${NIM_API_KEY ? '✓ Set' : '✗ NOT SET — proxy will fail!'}`);
  console.log(`───────────────────────────────────────────────`);
  console.log(`  Plugin tags: //SHOW_THINKING//  //SHOW_REASONING//`);
  console.log(`  Thinking default: disabled (explicitly set)`);
  console.log(`═══════════════════════════════════════════════`);
});
