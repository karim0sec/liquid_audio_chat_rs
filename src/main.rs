//! Low-latency, low-resource LFM2.5-Audio chat client (Rust).

mod api;
mod audio;

use api::{
    stream_chat, stream_single_shot, process_stream, ChatMessage, StreamStats,
};
use audio::{AudioPlayer, AudioRecorder, PlaybackHandle};
use clap::Parser;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

const SYSTEM_INTERLEAVED: &str = "Respond with interleaved text and audio.";

fn print_help() {
    println!(
        r#"
Commands:
  /mode <asr|tts|interleaved>  - Switch mode
  /reset                       - Reset context (interleaved mode only)
  /record                      - Record and transcribe/process audio
  /wav <path>                  - Load and transcribe/process audio file
  /help                        - Show this help
  /quit or /exit               - Exit the program

Modes:
  ASR (single-shot):
    - Use /record or /wav to transcribe audio
    - Each request is independent

  TTS (single-shot):
    - Type text to synthesize audio
    - Each request is independent

  Interleaved (chat):
    - Type text or use /record or /wav
    - Context is maintained across requests
    - Use /reset to start fresh
"#
    );
}

fn print_stats(stats: &StreamStats) {
    let mut parts = Vec::new();
    if let Some(ttft) = stats.ttft_secs {
        parts.push(format!("ttft {:.3}s", ttft));
    }
    if stats.text_chunk_count > 1 && stats.text_duration_secs > 0.0 {
        let rate = stats.text_chunk_count as f64 / stats.text_duration_secs;
        parts.push(format!(
            "text {} tok @ {:.1} tok/s",
            stats.text_chunk_count, rate
        ));
    }
    if stats.total_audio_samples > 0 && stats.audio_duration_secs > 0.0 {
        let rate = stats.total_audio_samples as f64 / stats.audio_duration_secs;
        let secs = stats.total_audio_samples as f64 / 24000.0;
        parts.push(format!("audio {:.1}s @ {:.0} samples/s", secs, rate));
    }
    parts.push(format!("total {:.3}s", stats.total_secs));
    println!("\n[{}]", parts.join(" | "));
}

#[derive(Parser)]
#[command(name = "liquid-audio-chat")]
#[command(about = "Low-latency LFM2.5-Audio chat client")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080/v1")]
    base_url: String,
    #[arg(long, default_value = "interleaved")]
    mode: String,
    #[arg(long, default_value = "512")]
    max_tokens: u32,
    #[arg(long)]
    no_audio_playback: bool,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args = Args::parse();
    if args.mode != "asr" && args.mode != "tts" && args.mode != "interleaved" {
        eprintln!("Invalid mode. Use asr, tts, or interleaved.");
        std::process::exit(1);
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("http client");
    let recorder = AudioRecorder::new();
    let audio_input_ok = recorder.available();
    let enable_playback = !args.no_audio_playback;

    println!("==================================================");
    println!("LFM2.5-Audio Interactive Chat (Rust)");
    println!("==================================================");
    println!("Server: {}", args.base_url);
    println!("Audio output: cpal");
    println!(
        "Audio input:  {}",
        if audio_input_ok {
            "microphone"
        } else {
            "file only (/wav)"
        }
    );
    println!("Type /help for commands");
    println!("==================================================");
    println!(
        "Mode: {}",
        if args.mode == "asr" || args.mode == "tts" {
            format!("{} (single-shot)", args.mode)
        } else {
            args.mode.clone() + " (chat)"
        }
    );

    let mut mode = args.mode.clone();
    let mut wav_data: Option<Vec<u8>> = None;
    let mut is_first_message = true;
    let mut rl = rustyline::DefaultEditor::new().expect("readline");

    loop {
        let mode_indicator = match mode.as_str() {
            "asr" => "[ASR]",
            "tts" => "[TTS]",
            _ => "[INT]",
        };
        let audio_indicator = if wav_data.is_some() { " [audio]" } else { "" };
        let prompt = format!("{}{}> ", mode_indicator, audio_indicator);

        let line = match rl.readline(&prompt) {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\nGoodbye!");
                break;
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("\nUse /quit to exit");
                continue;
            }
            Err(e) => {
                eprintln!("Readline error: {}", e);
                continue;
            }
        };
        let _ = rl.add_history_entry(line.as_str());

        let user_input = line.trim();
        let mut user_input = user_input;

        if user_input.is_empty() {
            if mode != "asr" || wav_data.is_none() {
                continue;
            }
        }

        if user_input.starts_with('/') {
            let parts: Vec<&str> = user_input.splitn(2, char::is_whitespace).collect();
            let cmd = parts[0].to_lowercase();
            let arg = parts.get(1).copied().unwrap_or("");

            match cmd.as_str() {
                "/quit" | "/exit" => {
                    println!("Goodbye!");
                    break;
                }
                "/help" => {
                    print_help();
                    continue;
                }
                "/mode" => {
                    if arg == "asr" || arg == "tts" || arg == "interleaved" {
                        if arg != mode {
                            mode = arg.to_string();
                            is_first_message = true;
                            println!(
                                "Mode: {}",
                                if mode == "asr" || mode == "tts" {
                                    format!("{} (single-shot)", mode)
                                } else {
                                    mode.clone() + " (chat)"
                                }
                            );
                        } else {
                            println!("Already in {} mode", mode);
                        }
                    } else {
                        println!("Usage: /mode <asr|tts|interleaved>");
                    }
                    continue;
                }
                "/reset" => {
                    if mode != "interleaved" {
                        println!("Reset only available in interleaved mode");
                        continue;
                    }
                    is_first_message = true;
                    println!("Context reset");
                    continue;
                }
                "/record" => {
                    if mode == "tts" {
                        println!("Recording not available in TTS mode");
                        continue;
                    }
                    if !recorder.available() {
                        println!("[No microphone available. Use /wav to load audio files.]");
                        continue;
                    }
                    println!("Recording... (Press Enter to stop)");
                    let stop_flag = Arc::new(AtomicBool::new(false));
                    let stop_c = Arc::clone(&stop_flag);
                    let rec = AudioRecorder::new();
                    let handle = thread::spawn(move || rec.record_blocking(move || stop_c.load(Ordering::Relaxed)));
                    // Wait for Enter (already got one line; that was the /record line; need another)
                    let _ = rl.readline(">> ");
                    stop_flag.store(true, Ordering::Relaxed);
                    match handle.join().expect("record thread") {
                        Ok(bytes) => {
                            if bytes.is_empty() {
                                continue;
                            }
                            wav_data = Some(bytes);
                            user_input = "";
                        }
                        Err(e) => {
                            println!("Record error: {}", e);
                            continue;
                        }
                    }
                }
                "/wav" => {
                    if mode == "tts" {
                        println!("Audio input not available in TTS mode");
                        continue;
                    }
                    if arg.is_empty() {
                        println!("Usage: /wav <path>");
                        continue;
                    }
                    match std::fs::read(Path::new(arg)) {
                        Ok(data) => {
                            wav_data = Some(data);
                            user_input = "";
                        }
                        Err(e) => {
                            println!("Error loading file: {}", e);
                            continue;
                        }
                    }
                }
                _ => {
                    println!("Unknown command: {}", cmd);
                    continue;
                }
            }
        }

        let text_input = if user_input.is_empty() || user_input.starts_with('/') {
            None
        } else {
            Some(user_input.to_string())
        };

        if mode == "asr" {
            if wav_data.is_none() {
                println!("ASR mode requires audio. Use /record or /wav first.");
                continue;
            }
        } else if mode == "tts" {
            if text_input.is_none() {
                println!("TTS mode requires text input.");
                continue;
            }
        }

        let (player, playback_handle) = if enable_playback {
            match AudioPlayer::new() {
                Ok(p) => {
                    let handle = p.handle();
                    (Some(p), Some(handle))
                }
                Err(e) => {
                    eprintln!("Audio output init failed: {}", e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        println!();

        let result = run_request(
            &client,
            &args.base_url,
            &mode,
            args.max_tokens,
            text_input.as_deref(),
            wav_data.as_deref(),
            &mut is_first_message,
            playback_handle,
        )
        .await;

        if let Some(p) = player {
            p.stop();
        }

        match result {
            Ok(stats) => print_stats(&stats),
            Err(e) => println!("Error: {}", e),
        }

        wav_data = None;
    }
}

async fn run_request(
    client: &reqwest::Client,
    base_url: &str,
    mode: &str,
    max_tokens: u32,
    text_input: Option<&str>,
    wav_data: Option<&[u8]>,
    is_first_message: &mut bool,
    playback_handle: Option<PlaybackHandle>,
) -> Result<api::StreamStats, String> {
    let res = if mode == "asr" || mode == "tts" {
        stream_single_shot(client, base_url, mode, text_input, wav_data, max_tokens).await?
    } else {
        let mut messages = Vec::new();
        if *is_first_message {
            messages.push(ChatMessage {
                role: "system".into(),
                content: Some(SYSTEM_INTERLEAVED.into()),
                content_array: None,
            });
        }
        if let Some(t) = text_input {
            messages.push(ChatMessage {
                role: "user".into(),
                content: Some(t.to_string()),
                content_array: None,
            });
        }
        if let Some(wav) = wav_data {
            messages.push(api::create_audio_message(wav));
        }
        let reset = *is_first_message;
        *is_first_message = false;
        stream_chat(client, base_url, messages, max_tokens, reset).await?
    };

    let mut stdout = std::io::stdout();
    let on_text = |s: &str| {
        let _ = stdout.write_all(s.as_bytes());
        let _ = stdout.flush();
    };
    let on_audio: Box<dyn FnMut(&[f32]) + Send> = if let Some(h) = playback_handle {
        Box::new(move |samples: &[f32]| h.add_samples(samples))
    } else {
        Box::new(|_| {})
    };

    let (_, stats) = process_stream(res, on_text, on_audio).await?;
    if !stats.completed {
        println!("[Warning: Server disconnected before completion]");
    }
    Ok(stats)
}
