//! OpenAI-compatible streaming chat client for LFM2.5-Audio.

use base64::Engine;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;

const B64: base64::engine::general_purpose::GeneralPurpose = base64::engine::general_purpose::STANDARD;

#[derive(Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_array: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize)]
struct ChatRequestBody {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    max_tokens: u32,
    #[serde(rename = "reset_context", skip_serializing_if = "Option::is_none")]
    reset_context: Option<bool>,
}

#[derive(Deserialize)]
pub struct StreamChunk {
    pub choices: Option<Vec<StreamChoice>>,
}

#[derive(Deserialize)]
pub struct StreamChoice {
    pub delta: Option<StreamDelta>,
    #[serde(rename = "finish_reason")]
    pub finish_reason: Option<String>,
}

#[derive(Deserialize)]
pub struct StreamDelta {
    pub content: Option<String>,
    #[serde(rename = "audio_chunk")]
    pub audio_chunk: Option<AudioChunk>,
}

#[derive(Deserialize)]
pub struct AudioChunk {
    pub data: String,
}

pub struct StreamStats {
    pub ttft_secs: Option<f64>,
    pub total_secs: f64,
    pub text_chunk_count: usize,
    pub text_duration_secs: f64,
    pub total_audio_samples: usize,
    pub audio_duration_secs: f64,
    pub completed: bool,
}

/// Single-shot ASR or TTS request (resets context).
pub async fn stream_single_shot(
    client: &Client,
    base_url: &str,
    mode: &str,
    text: Option<&str>,
    wav_data: Option<&[u8]>,
    max_tokens: u32,
) -> Result<reqwest::Response, String> {
    let system = match mode {
        "asr" => "Perform ASR.",
        "tts" => "Perform TTS. Use the UK female voice.",
        _ => return Err("invalid mode".into()),
    };
    let mut messages = vec![ChatMessage {
        role: "system".into(),
        content: Some(system.into()),
        content_array: None,
    }];
    if mode == "asr" {
        if let Some(wav) = wav_data {
            messages.push(create_audio_message(wav));
        }
    } else if mode == "tts" {
        if let Some(t) = text {
            messages.push(ChatMessage {
                role: "user".into(),
                content: Some(t.to_string()),
                content_array: None,
            });
        }
    }
    let body = ChatRequestBody {
        model: String::new(),
        messages,
        stream: true,
        max_tokens,
        reset_context: Some(true),
    };
    post_stream(client, base_url, &body).await
}

/// Chat request for interleaved mode.
pub async fn stream_chat(
    client: &Client,
    base_url: &str,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    reset_context: bool,
) -> Result<reqwest::Response, String> {
    let body = ChatRequestBody {
        model: String::new(),
        messages,
        stream: true,
        max_tokens,
        reset_context: Some(reset_context),
    };
    post_stream(client, base_url, &body).await
}

pub fn create_audio_message(wav_data: &[u8]) -> ChatMessage {
    let encoded = B64.encode(wav_data);
    let content_array = vec![serde_json::json!({
        "type": "input_audio",
        "input_audio": { "data": encoded, "format": "wav" }
    })];
    ChatMessage {
        role: "user".into(),
        content: None,
        content_array: Some(content_array),
    }
}

async fn post_stream(
    client: &Client,
    base_url: &str,
    body: &ChatRequestBody,
) -> Result<reqwest::Response, String> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let res = client
        .post(&url)
        .json(body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        return Err(format!("{}: {}", status, text));
    }
    Ok(res)
}

/// Process streaming response: parse NDJSON/SSE, call on_text/on_audio, return stats.
pub async fn process_stream<F, G>(
    res: reqwest::Response,
    mut on_text: F,
    mut on_audio: G,
) -> Result<(String, StreamStats), String>
where
    F: FnMut(&str),
    G: FnMut(&[f32]),
{
    let t0 = Instant::now();
    let mut ttft = None::<f64>;
    let mut text_chunks: Vec<(f64, String)> = Vec::new();
    let mut audio_chunks: Vec<(f64, usize)> = Vec::new();
    let mut total_samples = 0usize;
    let mut completed = false;
    let mut buffer = String::new();

    let mut stream = res.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        if let Ok(s) = std::str::from_utf8(&chunk) {
            buffer.push_str(s);
        }
        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer = buffer[line_end + 1..].to_string();
            let data = line.strip_prefix("data: ").unwrap_or("");
            if data == "[DONE]" || data.is_empty() {
                continue;
            }
            let chunk: StreamChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let choices = match &chunk.choices {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };
            let choice = &choices[0];
            if choice.finish_reason.as_deref() == Some("stop") {
                completed = true;
                break;
            }
            let delta = match &choice.delta {
                Some(d) => d,
                None => continue,
            };
            let now = t0.elapsed().as_secs_f64();
            if ttft.is_none() {
                ttft = Some(now);
            }
            if let Some(ref text) = delta.content {
                if !text.is_empty() {
                    text_chunks.push((now, text.clone()));
                    on_text(text);
                }
            }
            if let Some(ref ac) = delta.audio_chunk {
                let decoded = B64.decode(ac.data.as_bytes()).unwrap_or_default();
                let samples: Vec<f32> = decoded
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let n = samples.len();
                if n > 0 {
                    total_samples += n;
                    audio_chunks.push((now, n));
                    on_text("♪");
                    on_audio(&samples);
                }
            }
        }
        if completed {
            break;
        }
    }

    let total_secs = t0.elapsed().as_secs_f64();
    let full_text = text_chunks.iter().map(|(_, t)| t.as_str()).collect::<String>();
    let text_duration_secs = if text_chunks.len() > 1 {
        text_chunks.last().map(|(t, _)| *t).unwrap_or(0.0)
            - text_chunks.first().map(|(t, _)| *t).unwrap_or(0.0)
    } else {
        0.0
    };
    let audio_duration_secs = if audio_chunks.is_empty() {
        0.0
    } else {
        let first = audio_chunks.first().map(|(t, _)| *t).unwrap_or(0.0);
        let last = audio_chunks.last().map(|(t, _)| *t).unwrap_or(0.0);
        last - first
    };
    let stats = StreamStats {
        ttft_secs: ttft,
        total_secs,
        text_chunk_count: text_chunks.len(),
        text_duration_secs,
        total_audio_samples: total_samples,
        audio_duration_secs,
        completed,
    };
    Ok((full_text, stats))
}
