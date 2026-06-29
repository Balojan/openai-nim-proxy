// server.js - NVIDIA NIM Proxy for Janitor AI
// v2.3.1 - SSE streaming bugfix + reasoning fix
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
// MODEL CONFIGURATION
// ──────────────────────────────────────────────
const MODEL_CONFIG = {
  'deepseek-v3.1': {
    nimId: 'deepseek-ai/deepseek-v3.1',
    extraBodyOn: { chat_template_kwargs: { thinking: true } },
    extraBodyOff: { chat_template_kwargs: { thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning']
  },
  'deepseek-v4': {
    nimId: 'deepseek-ai/deepseek-v4',
    extraBodyOn: { chat_template_kwargs: { thinking: true, reasoning_effort: 'high' } },
    extraBodyOff: { chat_template_kwargs: { thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning']
  },
  'deepseek-v4-flash': {
    nimId: 'deepseek-ai/deepseek-v4-flash',
    extraBodyOn: { chat_template_kwargs: { thinking: true } },
    extraBodyOff: { chat_template_kwargs: { thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning']
  },
  'glm-5.1': {
    nimId: 'z-ai/glm-5.1',
    extraBodyOn: { chat_template_kwargs: { enable_thinking: true, clear_thinking: false } },
    extraBodyOff: { chat_template_kwargs: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning']
  },
  'qwen3-coder': {
    nimId: 'qwen/qwen3-coder-480b-a35b-instruct',
    extraBodyOn: { chat_template_kwargs: { enable_thinking: true } },
    extraBodyOff: { chat_template_kwargs: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning']
  },
  'qwen3-next': {
    nimId: 'qwen/qwen3-next-80b-a3b-thinking',
    extraBodyOn: { chat_template_kwargs: { enable_thinking: true } },
    extraBodyOff: { chat_template_kwargs: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning']
  },
  'kimi-k2': {
    nimId: 'moonshotai/kimi-k2-instruct-0905',
    extraBodyOn: { chat_template_kwargs: { thinking: true } },
    extraBodyOff: { chat_template_kwargs: { thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning'],
    contentNullWhenThinking: true
  },
  'kimi-k2.6': {
    nimId: 'moonshotai/kimi-k2.6',
    extraBodyOn: { chat_template_kwargs: { thinking: true } },
    extraBodyOff: { chat_template_kwargs: { thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning'],
    contentNullWhenThinking: true
  },
  'gpt-oss-120b': {
    nimId: 'openai/gpt-oss-120b',
    extraBodyOn: { reasoning_effort: 'high' },
    extraBodyOff: { reasoning_effort: 'low' },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning'],
    alwaysReasons: true
  },
  'gpt-oss-20b': {
    nimId: 'openai/gpt-oss-20b',
    extraBodyOn: { reasoning_effort: 'high' },
    extraBodyOff: { reasoning_effort: 'low' },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning'],
    alwaysReasons: true
  },
  'nemotron-ultra': {
    nimId: 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    extraBodyOn: { chat_template_kwargs: { enable_thinking: true } },
    extraBodyOff: { chat_template_kwargs: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningFields: ['reasoning_content', 'reasoning']
  },
  'llama-3.1-8b': { nimId: 'meta/llama-3.1-8b-instruct', supportsReasoning: false },
  'llama-3.1-70b': { nimId: 'meta/llama-3.1-70b-instruct', supportsReasoning: false },
  'llama-3.1-405b': { nimId: 'meta/llama-3.1-405b-instruct', supportsReasoning: false },
  'llama-3.3-70b': { nimId: 'meta/llama-3.3-70b-instruct', supportsReasoning: false },
};

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nemotron-ultra',
  'gpt-4': 'qwen3-coder',
  'gpt-4-turbo': 'kimi-k2',
  'gpt-4o': 'deepseek-v3.1',
  'gpt-4o-mini': 'deepseek-v4-flash',
  'claude-3-opus': 'gpt-oss-120b',
  'claude-3-sonnet': 'gpt-oss-20b',
  'claude-3-haiku': 'llama-3.1-8b',
  'gemini-pro': 'qwen3-next',
  'gemini-1.5-pro': 'qwen3-next',
  'deepseek-v4': 'deepseek-v4',
  'deepseek-v4-flash': 'deepseek-v4-flash',
  'deepseek-v3.1': 'deepseek-v3.1',
  'deepseek-chat': 'deepseek-v3.1',
  'glm-5.1': 'glm-5.1',
  'glm-4': 'glm-5.1',
  'kimi-k2': 'kimi-k2',
  'kimi-k2.6': 'kimi-k2.6',
  'llama-3.1-8b': 'llama-3.1-8b',
  'llama-3.1-70b': 'llama-3.1-70b',
  'llama-3.1-405b': 'llama-3.1-405b',
  'llama-3.3-70b': 'llama-3.3-70b',
  'nemotron-ultra': 'nemotron-ultra',
  'qwen3-coder': 'qwen3-coder',
  'qwen3-next': 'qwen3-next',
};

// ──────────────────────────────────────────────
// PLUGIN PARSER
// ──────────────────────────────────────────────
function parsePlugins(messages) {
  let showReasoning = false;
  let enableThinking = false;

  if (!messages || !Array.isArray(messages)) {
    return { messages, showReasoning, enableThinking };
  }

  const cleanedMessages = JSON.parse(JSON.stringify(messages));

  for (let i = 0; i < cleanedMessages.length; i++) {
    const msg = cleanedMessages[i];
    if (msg.role === 'system' && typeof msg.content === 'string') {
      if (msg.content.includes('//SHOW_REASONING//')) {
        showReasoning = true;
        msg.content = msg.content.replace(/\/\/SHOW_REASONING\//g, '');
      }
      if (msg.content.includes('//SHOW_THINKING//')) {
        enableThinking = true;
        msg.content = msg.content.replace(/\/\/SHOW_THINKING\//g, '');
      }
      msg.content = msg.content.replace(/\n{3,}/g, '\n\n').trim();
    }
  }

  return { messages: cleanedMessages, showReasoning, enableThinking };
}

// ──────────────────────────────────────────────
// FALLBACK MODEL RESOLVER
// ──────────────────────────────────────────────
async function resolveModel(model, apiKey, apiBase) {
  const configKey = MODEL_MAPPING[model];
  if (configKey && MODEL_CONFIG[configKey]) {
    return { configKey, config: MODEL_CONFIG[configKey], source: 'mapped' };
  }

  if (model.includes('/')) {
    try {
      const test = await axios.post(
        `${apiBase}/chat/completions`,
        { model: model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
        {
          headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500,
          timeout: 10000
        }
      );
      if (test.status >= 200 && test.status < 300) {
        return {
          configKey: model,
          config: { nimId: model, supportsReasoning: false },
          source: 'direct-nim'
        };
      }
    } catch (e) {}
  }

  const lower = model.toLowerCase();
  if (lower.includes('405b') || lower.includes('ultra') || lower.includes('large')) {
    return { configKey: 'llama-3.1-405b', config: MODEL_CONFIG['llama-3.1-405b'], source: 'heuristic-large' };
  }
  if (lower.includes('70b') || lower.includes('pro')) {
    return { configKey: 'llama-3.1-70b', config: MODEL_CONFIG['llama-3.1-70b'], source: 'heuristic-medium' };
  }
  return { configKey: 'llama-3.1-8b', config: MODEL_CONFIG['llama-3.1-8b'], source: 'fallback' };
}

// ──────────────────────────────────────────────
// GET REASONING FROM MESSAGE (checks multiple fields)
// ──────────────────────────────────────────────
function getReasoning(msg, config) {
  if (!msg || !config.supportsReasoning) return '';
  const fields = config.reasoningFields || ['reasoning_content', 'reasoning'];
  for (const field of fields) {
    if (msg[field] && typeof msg[field] === 'string' && msg[field].trim()) {
      return msg[field];
    }
  }
  return '';
}

// ──────────────────────────────────────────────
// GET CONTENT FROM MESSAGE
// ──────────────────────────────────────────────
function getContent(msg) {
  if (!msg) return '';
  if (msg.content === null || msg.content === undefined) return '';
  if (typeof msg.content !== 'string') return String(msg.content);
  return msg.content;
}

// ──────────────────────────────────────────────
// CLEAN LEAKED REASONING IN CONTENT
// ──────────────────────────────────────────────
function cleanLeakedReasoning(content) {
  if (!content || typeof content !== 'string') return content;
  let cleaned = content;
  cleaned = cleaned.replace(/([a-zA-Z])—\s*([a-zA-Z])/g, '$1, $2');
  cleaned = cleaned.replace(/([a-zA-Z])—\s*$/gm, '$1.');
  cleaned = cleaned.replace(/([a-zA-Z])—\s+(?=[A-Z])/g, '$1. ');
  const sentences = cleaned.split(/(?<=[.!?])\s+/);
  const fixed = sentences.map(sent => {
    const wordCount = sent.split(/\s+/).length;
    if (wordCount > 150 && !sent.includes('\n')) {
      return sent.replace(/\s+(?=(?:probably|perhaps|maybe|likely|apparently|obviously|clearly|evidently|presumably)\s)/gi, '. $1');
    }
    return sent;
  });
  return fixed.join(' ');
}

// ──────────────────────────────────────────────
// FORMAT REASONING FOR DISPLAY
// ──────────────────────────────────────────────
function formatWithReasoning(reasoning, content) {
  let result = '';
  if (reasoning) {
    result += ' think\n' + reasoning + '\n/think';
  }
  if (content && content.trim()) {
    if (result) result += '\n\n';
    result += content;
  }
  return result;
}

// ──────────────────────────────────────────────
// SSE WRITE HELPER — always writes with \\n\\n
// ──────────────────────────────────────────────
function writeSSE(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function writeSSERaw(res, line) {
  res.write(`${line}\n\n`);
}

// ──────────────────────────────────────────────
// ROUTES
// ──────────────────────────────────────────────

app.get('/', (req, res) => {
  res.send('NVIDIA NIM Proxy is running. Health check at /health');
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'NVIDIA NIM Proxy for Janitor AI', version: '2.3.1', timestamp: new Date().toISOString() });
});

app.get('/v1/chat/completions', (req, res) => {
  res.json({ status: 'ok', message: 'Use POST to chat.', available: true });
});

// ──────────────────────────────────────────────
// MAIN PROXY
// ──────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const requestStart = Date.now();

  try {
    const { model, messages, temperature, max_tokens, stream, top_p, top_k, frequency_penalty, presence_penalty } = req.body;

    // Parse plugins
    const pluginResult = parsePlugins(messages);
    const showReasoning = pluginResult.showReasoning;
    const enableThinking = pluginResult.enableThinking;
    const cleanedMessages = pluginResult.messages;

    // Resolve model
    const { configKey, config, source } = await resolveModel(model, NIM_API_KEY, NIM_API_BASE);
    const nimModel = config.nimId;
    console.log(`[${new Date().toISOString()}] "${model}" -> "${nimModel}" (source: ${source}, thinking=${enableThinking}, reasoning=${showReasoning})`);

    // Build NIM request
    const nimRequest = {
      model: nimModel,
      messages: cleanedMessages,
      temperature: temperature !== undefined ? temperature : 0.7,
      max_tokens: max_tokens !== undefined ? max_tokens : 9024,
      stream: stream || false
    };

    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (top_k !== undefined) nimRequest.top_k = top_k;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;

    // Apply thinking config inside extra_body
    if (config.supportsReasoning) {
      const extraBody = enableThinking ? config.extraBodyOn : config.extraBodyOff;
      if (extraBody) {
        nimRequest.extra_body = extraBody;
      }
    }

    console.log(`[DEBUG] extra_body: ${JSON.stringify(nimRequest.extra_body || {})}`);

    // Send to NVIDIA
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 300000
      }
    );

    // ── STREAMING ────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      let buffer = '';
      let reasoningBuffer = '';
      let contentBuffer = '';
      let reasoningStarted = false;
      let hasError = false;
      let firstChunkLogged = false;

      response.data.on('data', (chunk) => {
        if (hasError) return;
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.trim()) return; // Skip empty lines

          // [DONE] signal
          if (line.includes('[DONE]')) {
            writeSSERaw(res, line);
            return;
          }

          // Must start with "data: "
          if (!line.startsWith('data: ')) {
            // Unknown line format, forward as-is
            writeSSERaw(res, line);
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            if (!data.choices?.[0]?.delta) {
              // No delta — might be a heartbeat or error, forward as-is
              writeSSE(res, data);
              return;
            }

            const delta = data.choices[0].delta;

            // Check ALL possible reasoning field names
            let reasoningChunk = '';
            if (delta.reasoning_content && typeof delta.reasoning_content === 'string') reasoningChunk = delta.reasoning_content;
            else if (delta.reasoning && typeof delta.reasoning === 'string') reasoningChunk = delta.reasoning;

            let contentChunk = getContent(delta);

            // Log first chunk for debugging
            if (!firstChunkLogged) {
              console.log(`[DEBUG] First chunk: reasoning=${reasoningChunk.length > 0}, content=${contentChunk.length > 0}, contentNull=${delta.content === null}`);
              firstChunkLogged = true;
            }

            if (showReasoning && config.supportsReasoning) {
              // ── SHOW REASONING MODE ──
              if (reasoningChunk) {
                if (!reasoningStarted) {
                  reasoningBuffer = ' think\n' + reasoningChunk;
                  reasoningStarted = true;
                } else {
                  reasoningBuffer += reasoningChunk;
                }
              }

              if (contentChunk) {
                // Content arrived — flush reasoning first, then content
                let combined = '';
                if (reasoningStarted) {
                  combined = reasoningBuffer + '\n/think\n\n' + contentChunk;
                  reasoningBuffer = '';
                  reasoningStarted = false;
                } else {
                  combined = contentChunk;
                }
                data.choices[0].delta.content = combined;
                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
                writeSSE(res, data);
              } else if (reasoningChunk && reasoningStarted) {
                // Still buffering reasoning, don't emit yet
                return;
              } else if (!reasoningChunk && !contentChunk) {
                // Empty delta
                data.choices[0].delta.content = '';
                writeSSE(res, data);
              }
            } else {
              // ── HIDE REASONING MODE ──
              if (contentChunk) {
                data.choices[0].delta.content = cleanLeakedReasoning(contentChunk);
              } else if (reasoningChunk && !contentChunk) {
                // content is null but reasoning exists — use reasoning as content
                data.choices[0].delta.content = reasoningChunk;
              } else {
                data.choices[0].delta.content = '';
              }
              delete data.choices[0].delta.reasoning_content;
              delete data.choices[0].delta.reasoning;
              writeSSE(res, data);
            }
          } catch (e) {
            // JSON parse failed — write raw line as SSE event
            // This prevents partial data from being concatenated with next event
            writeSSERaw(res, line);
          }
        });
      });

      response.data.on('end', () => {
        // Flush remaining buffered reasoning
        if (showReasoning && reasoningBuffer && reasoningStarted) {
          const flushData = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: model,
            choices: [{
              index: 0,
              delta: { content: '\n/think\n\n' + (contentBuffer || '') },
              finish_reason: 'stop'
            }]
          };
          writeSSE(res, flushData);
        }
        writeSSERaw(res, 'data: [DONE]');
        res.end();
        console.log(`[${new Date().toISOString()}] Stream done in ${Date.now() - requestStart}ms`);
      });

      response.data.on('error', (err) => {
        hasError = true;
        console.error('Stream error:', err.message);
        const errorData = { error: { message: 'Stream error: ' + err.message } };
        writeSSE(res, errorData);
        res.end();
      });

      req.on('close', () => {
        if (!res.writableEnded) res.end();
      });

    // ── NON-STREAMING ────────────────────────
    } else {
      const choice = response.data.choices?.[0];
      const msg = choice?.message;

      const reasoning = getReasoning(msg, config);
      let content = getContent(msg);

      console.log(`[DEBUG] Non-streaming: reasoning=${reasoning.length > 0}, content=${content.length > 0}, contentNull=${msg?.content === null}`);

      // If content is null/empty but reasoning exists, use reasoning
      if ((!content || content === 'null') && reasoning) {
        content = reasoning;
      }

      let finalContent = content;
      if (showReasoning && reasoning) {
        finalContent = formatWithReasoning(reasoning, content);
      }

      if (!finalContent || finalContent === 'null') {
        finalContent = reasoning || '[No response content]';
      }

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: { role: msg?.role || 'assistant', content: finalContent },
          finish_reason: choice?.finish_reason || 'stop'
        }],
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
      console.log(`[${new Date().toISOString()}] Done in ${Date.now() - requestStart}ms`);
    }

  } catch (error) {
    console.error(`[${new Date().toISOString()}] Error:`, error.message);
    if (error.response?.data) {
      console.error(`[DEBUG] NIM error:`, JSON.stringify(error.response.data, null, 2));
    }
    const statusCode = error.response?.status || 500;
    const errorMsg = error.response?.data?.error?.message
      || error.response?.data?.message
      || error.message
      || 'Internal server error';

    res.status(statusCode).json({
      error: {
        message: errorMsg,
        type: statusCode >= 500 ? 'internal_server_error' : 'invalid_request_error',
        code: statusCode
      }
    });
  }
});

app.get('/v1/models', (req, res) => {
  const models = Object.entries(MODEL_MAPPING).map(([id, configKey]) => {
    const cfg = MODEL_CONFIG[configKey];
    return {
      id,
      nim_model_id: cfg.nimId,
      supports_reasoning: cfg.supportsReasoning,
      thinking_config: cfg.supportsReasoning ? cfg.extraBodyOn : 'n/a'
    };
  });
  res.json({ object: 'list', data: models });
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.method} ${req.path} not found` } });
});

app.listen(PORT, () => {
  console.log(`═══════════════════════════════════════════════`);
  console.log(`  NVIDIA NIM Proxy for Janitor AI`);
  console.log(`  Version: 2.3.1`);
  console.log(`  Port: ${PORT}`);
  console.log(`  API Base: ${NIM_API_BASE}`);
  console.log(`  API Key: ${NIM_API_KEY ? 'Set' : 'NOT SET!'}`);
  console.log(`───────────────────────────────────────────────`);
  console.log(`  Plugin tags: //SHOW_THINKING//  //SHOW_REASONING//`);
  console.log(`  Models loaded: ${Object.keys(MODEL_CONFIG).length}`);
  console.log(`═══════════════════════════════════════════════`);
});
