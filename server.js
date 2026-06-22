// server.js - NVIDIA NIM Proxy for Janitor AI
// v2.1.0 - Per-model thinking config + gibberish fix
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
// GLOBAL DEFAULTS
// ──────────────────────────────────────────────
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// ──────────────────────────────────────────────
// MODEL CONFIGURATION
// Each model has its NIM ID + how to enable/disable thinking
// ──────────────────────────────────────────────
const MODEL_CONFIG = {
  // ── DeepSeek ───────────────────────────────
  'deepseek-v3.1': {
    nimId: 'deepseek-ai/deepseek-v3.1',
    thinking: { param: 'chat_template_kwargs', value: { thinking: true } },
    noThinking: { param: 'chat_template_kwargs', value: { thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },
  'deepseek-v4': {
    nimId: 'deepseek-ai/deepseek-v4',
    thinking: { param: 'chat_template_kwargs', value: { thinking: true, reasoning_effort: 'high' } },
    noThinking: { param: 'chat_template_kwargs', value: { thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },
  'deepseek-v4-flash': {
    nimId: 'deepseek-ai/deepseek-v4-flash',
    thinking: { param: 'chat_template_kwargs', value: { thinking: true } },
    noThinking: { param: 'chat_template_kwargs', value: { thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },
  'deepseek-chat': {
    nimId: 'deepseek-ai/deepseek-v3.1',
    thinking: { param: 'chat_template_kwargs', value: { thinking: true } },
    noThinking: { param: 'chat_template_kwargs', value: { thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },

  // ── GLM ────────────────────────────────────
  'glm-5.1': {
    nimId: 'z-ai/glm-5.1',
    thinking: { param: 'chat_template_kwargs', value: { enable_thinking: true, clear_thinking: false } },
    noThinking: { param: 'chat_template_kwargs', value: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },
  'glm-4': {
    nimId: 'z-ai/glm-5.1',
    thinking: { param: 'chat_template_kwargs', value: { enable_thinking: true, clear_thinking: false } },
    noThinking: { param: 'chat_template_kwargs', value: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },

  // ── Qwen ───────────────────────────────────
  'qwen3-coder': {
    nimId: 'qwen/qwen3-coder-480b-a35b-instruct',
    thinking: { param: 'chat_template_kwargs', value: { enable_thinking: true } },
    noThinking: { param: 'chat_template_kwargs', value: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },
  'qwen3-next': {
    nimId: 'qwen/qwen3-next-80b-a3b-thinking',
    thinking: { param: 'chat_template_kwargs', value: { enable_thinking: true } },
    noThinking: { param: 'chat_template_kwargs', value: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },

  // ── Kimi / Moonshot ────────────────────────
  // WARNING: Kimi K2.6 returns content: null when thinking is enabled!
  // Only enable thinking if you also use //SHOW_REASONING//
  'kimi-k2': {
    nimId: 'moonshotai/kimi-k2-instruct-0905',
    thinking: { param: 'chat_template_kwargs', value: { thinking: true } },
    noThinking: { param: 'chat_template_kwargs', value: { thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content',
    contentNullWhenThinking: true
  },
  'kimi-k2.6': {
    nimId: 'moonshotai/kimi-k2.6',
    thinking: { param: 'chat_template_kwargs', value: { thinking: true } },
    noThinking: { param: 'chat_template_kwargs', value: { thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content',
    contentNullWhenThinking: true
  },

  // ── GPT-OSS ────────────────────────────────
  // Always reasons, uses reasoning_effort to control depth
  'gpt-oss-120b': {
    nimId: 'openai/gpt-oss-120b',
    thinking: { param: 'reasoning_effort', value: 'high' },
    noThinking: { param: 'reasoning_effort', value: 'low' },
    supportsReasoning: true,
    reasoningField: 'reasoning_content',
    alwaysReasons: true
  },
  'gpt-oss-20b': {
    nimId: 'openai/gpt-oss-20b',
    thinking: { param: 'reasoning_effort', value: 'high' },
    noThinking: { param: 'reasoning_effort', value: 'low' },
    supportsReasoning: true,
    reasoningField: 'reasoning_content',
    alwaysReasons: true
  },

  // ── Nemotron ───────────────────────────────
  'nemotron-ultra': {
    nimId: 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    thinking: { param: 'chat_template_kwargs', value: { enable_thinking: true } },
    noThinking: { param: 'chat_template_kwargs', value: { enable_thinking: false } },
    supportsReasoning: true,
    reasoningField: 'reasoning_content'
  },

  // ── Llama (no native reasoning) ──────────────
  'llama-3.1-8b': {
    nimId: 'meta/llama-3.1-8b-instruct',
    supportsReasoning: false
  },
  'llama-3.1-70b': {
    nimId: 'meta/llama-3.1-70b-instruct',
    supportsReasoning: false
  },
  'llama-3.1-405b': {
    nimId: 'meta/llama-3.1-405b-instruct',
    supportsReasoning: false
  },
  'llama-3.3-70b': {
    nimId: 'meta/llama-3.3-70b-instruct',
    supportsReasoning: false
  },
};

// Janitor AI friendly names → config keys
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
  'deepseek-chat': 'deepseek-chat',
  'glm-5.1': 'glm-5.1',
  'glm-4': 'glm-4',
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
  let showReasoning = SHOW_REASONING;
  let enableThinking = ENABLE_THINKING_MODE;

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

  // Direct NIM ID test
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

  // Heuristic fallback
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
// APPLY THINKING CONFIG
// Returns the correct parameter structure for the model
// ──────────────────────────────────────────────
function applyThinkingConfig(nimRequest, config, enableThinking) {
  if (!config.supportsReasoning) {
    // Model doesn't support reasoning at all — nothing to do
    return;
  }

  const thinkingCfg = enableThinking ? config.thinking : config.noThinking;
  if (!thinkingCfg) return;

  if (thinkingCfg.param === 'chat_template_kwargs') {
    nimRequest.chat_template_kwargs = thinkingCfg.value;
  } else if (thinkingCfg.param === 'reasoning_effort') {
    nimRequest.reasoning_effort = thinkingCfg.value;
  }
}

// ──────────────────────────────────────────────
// EXTRACT REASONING + CONTENT SAFELY
// Handles models that return content: null with only reasoning_content
// ──────────────────────────────────────────────
function extractMessageParts(message, config, showReasoning) {
  const reasoningField = config.reasoningField || 'reasoning_content';
  const reasoning = message?.[reasoningField] || message?.reasoning || '';
  let content = message?.content || '';

  // CRITICAL FIX: Some models (Kimi) return content: null when thinking is on
  // If content is null/empty but reasoning exists, use reasoning as content
  // unless showReasoning is true (then we format it properly)
  if ((!content || content === 'null') && reasoning) {
    if (showReasoning) {
      // Format: <think>reasoning</think>\n\n(content was null)
      content = '';
    } else {
      // No reasoning display requested, but content is null
      // Fall back to reasoning text so user isn't left with nothing
      content = reasoning;
    }
  }

  return { reasoning, content };
}

// ──────────────────────────────────────────────
// FORMAT REASONING FOR DISPLAY
// ──────────────────────────────────────────────
function formatWithReasoning(reasoning, content) {
  // Use actual XML-like tags that render cleanly
  let result = '';
  if (reasoning) {
    result += '<think>\n' + reasoning + '\n</think>';
  }
  if (content && content.trim()) {
    if (result) result += '\n\n';
    result += content;
  }
  return result;
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
    service: 'NVIDIA NIM Proxy for Janitor AI',
    version: '2.1.0',
    timestamp: new Date().toISOString()
  });
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
    const { model, messages, temperature, max_tokens, stream, top_p, frequency_penalty, presence_penalty } = req.body;

    // Parse plugins
    const pluginResult = parsePlugins(messages);
    const showReasoning = pluginResult.showReasoning;
    const enableThinking = pluginResult.enableThinking;
    const cleanedMessages = pluginResult.messages;

    // Resolve model
    const { configKey, config, source } = await resolveModel(model, NIM_API_KEY, NIM_API_BASE);
    const nimModel = config.nimId;
    console.log(`[${new Date().toISOString()}] "${model}" -> "${nimModel}" (via ${source}, thinking=${enableThinking}, reasoning=${showReasoning})`);

    // Build NIM request
    const nimRequest = {
      model: nimModel,
      messages: cleanedMessages,
      temperature: temperature !== undefined ? temperature : 0.6,
      max_tokens: max_tokens !== undefined ? max_tokens : 9024,
      stream: stream || false
    };

    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;

    // Apply per-model thinking config
    applyThinkingConfig(nimRequest, config, enableThinking);

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
      let reasoningSent = false;
      let hasError = false;

      response.data.on('data', (chunk) => {
        if (hasError) return;
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) {
            // Flush any remaining reasoning before DONE
            if (showReasoning && reasoningBuffer && !reasoningSent) {
              const flushData = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model: model,
                choices: [{
                  index: 0,
                  delta: { content: '\n</think>\n\n' + contentBuffer },
                  finish_reason: null
                }]
              };
              res.write(`data: ${JSON.stringify(flushData)}\n\n`);
              reasoningBuffer = '';
              contentBuffer = '';
            }
            res.write(line + '\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            if (!data.choices?.[0]?.delta) {
              res.write(line + '\n');
              return;
            }

            const delta = data.choices[0].delta;
            const reasoningChunk = delta.reasoning_content || delta.reasoning || '';
            const contentChunk = delta.content || '';

            if (showReasoning && config.supportsReasoning) {
              // Buffer reasoning until we see content, then emit formatted block
              if (reasoningChunk) {
                if (!reasoningSent) {
                  reasoningBuffer = '<think>\n' + reasoningChunk;
                  reasoningSent = true;
                } else {
                  reasoningBuffer += reasoningChunk;
                }
              }

              if (contentChunk) {
                if (reasoningSent) {
                  // Emit reasoning + transition + content
                  data.choices[0].delta.content = reasoningBuffer + '\n</think>\n\n' + contentChunk;
                  reasoningBuffer = '';
                  reasoningSent = false;
                } else {
                  data.choices[0].delta.content = contentChunk;
                }
                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              } else if (reasoningChunk && reasoningSent) {
                // Still receiving reasoning, don't emit yet
                return;
              }
            } else {
              // Don't show reasoning — only forward content
              // CRITICAL: If content is null/empty but reasoning exists,
              // some models leak reasoning into content. Filter it.
              if (contentChunk && contentChunk.trim()) {
                data.choices[0].delta.content = contentChunk;
              } else {
                data.choices[0].delta.content = '';
              }
              delete data.choices[0].delta.reasoning_content;
              delete data.choices[0].delta.reasoning;
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            }
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => {
        // Flush remaining buffered reasoning
        if (showReasoning && reasoningBuffer) {
          const flushData = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: model,
            choices: [{
              index: 0,
              delta: { content: '\n</think>\n\n' + (contentBuffer || '') },
              finish_reason: 'stop'
            }]
          };
          res.write(`data: ${JSON.stringify(flushData)}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
        console.log(`[${new Date().toISOString()}] Stream done in ${Date.now() - requestStart}ms`);
      });

      response.data.on('error', (err) => {
        hasError = true;
        console.error('Stream error:', err.message);
        res.write(`data: ${JSON.stringify({ error: { message: 'Stream error' } })}\n\n`);
        res.end();
      });

      req.on('close', () => {
        if (!res.writableEnded) res.end();
      });

    // ── NON-STREAMING ────────────────────────
    } else {
      const choice = response.data.choices?.[0];
      const msg = choice?.message;

      const { reasoning, content } = extractMessageParts(msg, config, showReasoning);

      let finalContent = content;
      if (showReasoning && reasoning) {
        finalContent = formatWithReasoning(reasoning, content);
      }

      // If content is still empty after all handling, use reasoning as fallback
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

// Model list endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.entries(MODEL_MAPPING).map(([id, configKey]) => {
    const cfg = MODEL_CONFIG[configKey];
    return {
      id,
      object: 'model',
      owned_by: cfg.nimId.split('/')[0],
      nim_model_id: cfg.nimId,
      supports_reasoning: cfg.supportsReasoning,
      thinking_param: cfg.supportsReasoning ? (cfg.thinking?.param || 'none') : 'n/a'
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
  console.log(`  Version: 2.1.0`);
  console.log(`  Port: ${PORT}`);
  console.log(`  API Base: ${NIM_API_BASE}`);
  console.log(`  API Key: ${NIM_API_KEY ? 'Set' : 'NOT SET!'}`);
  console.log(`───────────────────────────────────────────────`);
  console.log(`  Plugin tags: //SHOW_THINKING//  //SHOW_REASONING//`);
  console.log(`  Models loaded: ${Object.keys(MODEL_CONFIG).length}`);
  console.log(`═══════════════════════════════════════════════`);
});
