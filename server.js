// server.js - NVIDIA NIM Proxy for Janitor AI
// Plugin-enabled: control thinking/reasoning from the system prompt
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ──────────────────────────────────────────────
// GLOBAL DEFAULTS (used when no plugin tags found)
// ──────────────────────────────────────────────
const SHOW_REASONING = false;      // global fallback
const ENABLE_THINKING_MODE = false; // global fallback

// ──────────────────────────────────────────────
// MODEL MAPPING
// ──────────────────────────────────────────────
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm-5.1': 'z-ai/glm-5.1',
  'kimi-k2.6': 'moonshotai/kimi-k2.6'
};

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

  // Deep-clone to avoid mutating the original
  const cleanedMessages = JSON.parse(JSON.stringify(messages));

  for (let i = 0; i < cleanedMessages.length; i++) {
    const msg = cleanedMessages[i];
    if (msg.role === 'system' && typeof msg.content === 'string') {
      if (msg.content.includes('//SHOW_REASONING//')) {
        showReasoning = true;
        msg.content = msg.content.replace(/\/\/SHOW_REASONING\/\//g, '');
      }
      if (msg.content.includes('//SHOW_THINKING//')) {
        enableThinking = true;
        msg.content = msg.content.replace(/\/\/SHOW_THINKING\/\//g, '');
      }
      // Clean up leftover whitespace
      msg.content = msg.content.replace(/\n{3,}/g, '\n\n').trim();
    }
  }

  return { messages: cleanedMessages, showReasoning, enableThinking };
}

// ──────────────────────────────────────────────
// ROUTES
// ──────────────────────────────────────────────

app.get('/', (req, res) => {
  res.send('NVIDIA NIM Proxy is running. Health check at /health');
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'NVIDIA NIM Proxy for Janitor AI (plugin-enabled)' });
});

// GET fallback for Janitor AI's pre-check
app.get('/v1/chat/completions', (req, res) => {
  res.json({ status: 'ok', message: 'Completions endpoint. Use POST to chat.' });
});

// ──────────────────────────────────────────────
// MAIN PROXY – POST /v1/chat/completions
// ──────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // ── Plugin injection ──────────────────────
    const pluginResult = parsePlugins(messages);
    const showReasoning = pluginResult.showReasoning;
    const enableThinking = pluginResult.enableThinking;
    const cleanedMessages = pluginResult.messages;
    // ──────────────────────────────────────────

    // ── Model resolution ──────────────────────
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        const test = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        });
        if (test.status >= 200 && test.status < 300) nimModel = model;
      } catch (e) {}

      if (!nimModel) {
        const lower = model.toLowerCase();
        if (lower.includes('gpt-4') || lower.includes('claude-opus') || lower.includes('405b'))
          nimModel = 'meta/llama-3.1-405b-instruct';
        else if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b'))
          nimModel = 'meta/llama-3.1-70b-instruct';
        else
          nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }

    // ── Build NIM request ─────────────────────
    const nimRequest = {
      model: nimModel,
      messages: cleanedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    if (enableThinking) {
      nimRequest.extra_body = {
        thinking: { type: "enabled" }
      };
    }

    // ── Send to NVIDIA ────────────────────────
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    // ── STREAMING ─────────────────────────────
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
          if (line.startsWith('data: ')) {
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
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
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
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => {
        res.write('data: [DONE]\n\n');
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    // ── NON-STREAMING ─────────────────────────
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let content = choice.message?.content || '';
          if (showReasoning && choice.message?.reasoning_content) {
            content = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + content;
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
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all 404
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found` }
  });
});

app.listen(PORT, () => {
  console.log(`NVIDIA NIM Proxy running on port ${PORT}`);
  console.log('Plugin tags: //SHOW_THINKING// //SHOW_REASONING//');
});
