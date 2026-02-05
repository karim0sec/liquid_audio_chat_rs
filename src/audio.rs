//! Low-latency audio I/O via cpal. Playback uses a lock-free channel fed by the stream.

use crossbeam_channel::{bounded, Sender};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use hound::{WavSpec, WavWriter};
use std::cell::RefCell;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

const PLAYBACK_SAMPLE_RATE: u32 = 24000;
const RECORD_SAMPLE_RATE: u32 = 16000;
const CHANNELS: u16 = 1;
const QUEUE_CAPACITY: usize = 64;

/// Send-safe handle to push samples from async/other threads.
#[derive(Clone)]
pub struct PlaybackHandle(Arc<Sender<Vec<f32>>>);

impl PlaybackHandle {
    #[inline]
    pub fn add_samples(&self, samples: &[f32]) {
        if !samples.is_empty() {
            let _ = self.0.try_send(samples.to_vec());
        }
    }
}

/// Non-blocking audio player. Streams f32 mono at 24 kHz. Not Send (cpal stream).
pub struct AudioPlayer {
    tx: Arc<Sender<Vec<f32>>>,
    running: Arc<AtomicBool>,
    _stream_guard: Arc<Mutex<Option<cpal::Stream>>>,
}

impl AudioPlayer {
    pub fn new() -> Result<Self, String> {
        let (tx, rx) = bounded::<Vec<f32>>(QUEUE_CAPACITY);
        let running = Arc::new(AtomicBool::new(true));

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("no default output device")?;

        let config = StreamConfig {
            channels: CHANNELS,
            sample_rate: cpal::SampleRate(PLAYBACK_SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Default,
        };

        let run = Arc::clone(&running);
        let leftover: RefCell<Option<(Vec<f32>, usize)>> = RefCell::new(None);
        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if !run.load(Ordering::Relaxed) {
                        return;
                    }
                    let mut written = 0;
                    // Drain leftover from previous callback
                    {
                        let mut left = leftover.borrow_mut();
                        if let Some((ref v, ref mut offset)) = *left {
                            let take = (data.len() - written).min(v.len() - *offset);
                            data[written..written + take].copy_from_slice(&v[*offset..*offset + take]);
                            written += take;
                            *offset += take;
                            if *offset >= v.len() {
                                *left = None;
                            }
                        }
                    }
                    while written < data.len() {
                        match rx.try_recv() {
                            Ok(mut chunk) => {
                                let need = data.len() - written;
                                let take = need.min(chunk.len());
                                data[written..written + take].copy_from_slice(&chunk[..take]);
                                written += take;
                                if take < chunk.len() {
                                    chunk.drain(..take);
                                    *leftover.borrow_mut() = Some((chunk, 0));
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    if written < data.len() {
                        data[written..].fill(0.0);
                    }
                },
                move |e| eprintln!("audio output error: {}", e),
                None,
            )
            .map_err(|e| e.to_string())?;

        stream.play().map_err(|e| e.to_string())?;

        Ok(Self {
            tx: Arc::new(tx),
            running,
            _stream_guard: Arc::new(Mutex::new(Some(stream))),
        })
    }

    /// Handle that can be sent to async tasks for feeding audio.
    #[inline]
    pub fn handle(&self) -> PlaybackHandle {
        PlaybackHandle(Arc::clone(&self.tx))
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        let _ = self._stream_guard.lock().unwrap().take();
    }
}

/// Record from default microphone, convert to WAV bytes.
pub struct AudioRecorder {
    sample_rate: u32,
    available: bool,
}

impl AudioRecorder {
    pub fn new() -> Self {
        let available = cpal::default_host()
            .default_input_device()
            .is_some();
        Self {
            sample_rate: RECORD_SAMPLE_RATE,
            available,
        }
    }

    pub fn available(&self) -> bool {
        self.available
    }

    /// Record until `stop` is signaled. Returns WAV file bytes (mono f32 → i16 for WAV).
    pub fn record_blocking(&self, stop: impl Fn() -> bool) -> Result<Vec<u8>, String> {
        if !self.available {
            return Err("no microphone".into());
        }

        let device = cpal::default_host()
            .default_input_device()
            .ok_or("no input device")?;

        let config = StreamConfig {
            channels: CHANNELS,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let samples: Arc<std::sync::Mutex<Vec<f32>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let samples_clone = Arc::clone(&samples);

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    samples_clone.lock().unwrap().extend_from_slice(data);
                },
                move |e| eprintln!("audio input error: {}", e),
                None,
            )
            .map_err(|e| e.to_string())?;

        stream.play().map_err(|e| e.to_string())?;

        while !stop() {
            thread::sleep(std::time::Duration::from_millis(50));
        }

        drop(stream);
        let recorded = samples.lock().unwrap().clone();
        if recorded.is_empty() {
            return Ok(Vec::new());
        }

        samples_to_wav_bytes(&recorded, self.sample_rate)
    }
}

/// Encode f32 samples (-1..1) to WAV bytes (16-bit PCM).
pub fn samples_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    let spec = WavSpec {
        channels: CHANNELS,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::new(&mut cursor, spec).map_err(|e| e.to_string())?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample = (clamped * 32767.0) as i16;
        writer.write_sample(sample).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

