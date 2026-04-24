/// Audio capture abstraction for the STT Tauri client.
///
/// Provides a trait-based audio source API with two implementations:
/// - `CpalAudioSource` for live microphone capture via the cpal crate
/// - `FileAudioSource` for streaming WAV files at real-time pace (testing/replay)
///
/// Both deliver audio as 512-sample f32 chunks at 16 kHz mono through a callback.
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavReader;
use rubato::{FastFixedIn, PolynomialDegree, Resampler};

use crate::error::AppError;

const CHUNK_SIZE: usize = 512;
const SAMPLE_RATE: u32 = 16_000;
/// 512 samples at 16 kHz
const CHUNK_DURATION: Duration = Duration::from_millis(32);

/// Trait for audio sources that deliver 512-sample f32 chunks via callback.
///
/// Implementations must be `Send + 'static` so they can be moved across threads.
pub trait AudioSource: Send + 'static {
    /// Start capturing audio, invoking `callback` with each 512-sample chunk.
    fn start(&mut self, callback: Box<dyn Fn(Vec<f32>) + Send + 'static>) -> Result<(), AppError>;

    /// Stop capturing audio.
    fn stop(&mut self) -> Result<(), AppError>;
}

/// Wrapper that asserts `Send` for `cpal::Stream`.
///
/// cpal marks `Stream` as `!Send` for cross-platform safety, but on Windows
/// (WASAPI) the underlying handle is safe to move between threads.
#[allow(dead_code)]
struct SendStream(cpal::Stream);

// SAFETY: WASAPI stream handles are thread-safe on Windows.
unsafe impl Send for SendStream {}

/// Captures audio from the default system input device using cpal.
///
/// Configures for 16 kHz mono f32. Buffers incoming samples from cpal's
/// arbitrary-sized callbacks and fires the user callback every 512 samples.
pub struct CpalAudioSource {
    running: Arc<AtomicBool>,
    stream: Option<SendStream>,
}

impl CpalAudioSource {
    /// Creates a new `CpalAudioSource` (does not start capturing).
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            stream: None,
        }
    }

    /// Finds a supported input config closest to 16 kHz mono f32.
    ///
    /// Algorithm:
    /// 1. Query all supported input configs from the device.
    /// 2. Filter to configs that support F32 sample format.
    /// 3. Prefer configs where 16000 Hz falls within the supported range.
    /// 4. Among candidates, prefer mono (1 channel).
    /// 5. If no config includes 16 kHz, pick the one whose max sample rate is closest.
    fn find_best_config(
        device: &cpal::Device,
    ) -> Result<cpal::SupportedStreamConfig, AppError> {
        let mut configs: Vec<cpal::SupportedStreamConfigRange> = device
            .supported_input_configs()
            .map_err(|e| AppError::AudioError(format!("failed to query configs: {e}")))?
            .collect();

        if configs.is_empty() {
            return Err(AppError::AudioError("no supported input configs".into()));
        }

        configs.sort_by_key(|c| {
            let supports_rate =
                c.min_sample_rate().0 <= SAMPLE_RATE && c.max_sample_rate().0 >= SAMPLE_RATE;
            let is_f32 = c.sample_format() == cpal::SampleFormat::F32;
            let is_mono = c.channels() == 1;
            // Lower score = better; sort ascending
            let score = (!is_f32 as u32) * 100 + (!supports_rate as u32) * 10 + (!is_mono as u32);
            score
        });

        let best = &configs[0];
        let rate = if best.min_sample_rate().0 <= SAMPLE_RATE
            && best.max_sample_rate().0 >= SAMPLE_RATE
        {
            cpal::SampleRate(SAMPLE_RATE)
        } else {
            best.max_sample_rate()
        };

        Ok(best.clone().with_sample_rate(rate))
    }
}

impl AudioSource for CpalAudioSource {
    /// Starts microphone capture, resampling to 16 kHz mono before chunking.
    ///
    /// Algorithm:
    /// 1. Select the best supported input config (preferring F32, 16 kHz, mono).
    /// 2. Downmix multi-channel frames to mono in the cpal callback.
    /// 3. If the device sample rate differs from 16 kHz, feed mono samples through
    ///    a `FastFixedIn` resampler before buffering.
    /// 4. Drain 512-sample chunks from the buffer and fire the callback.
    fn start(&mut self, callback: Box<dyn Fn(Vec<f32>) + Send + 'static>) -> Result<(), AppError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| AppError::AudioError("no default input device".into()))?;

        let config = Self::find_best_config(&device)?;
        let device_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        tracing::info!(
            "Audio config: {}ch {}Hz {:?}",
            channels,
            device_rate,
            config.sample_format()
        );

        let needs_resample = device_rate != SAMPLE_RATE;
        let resample_ratio = SAMPLE_RATE as f64 / device_rate as f64;

        // Input block size fed to the resampler: enough device samples to produce
        // at least one full 512-sample output chunk, rounded up to nearest 64.
        let resample_input_size = if needs_resample {
            let raw = (CHUNK_SIZE as f64 / resample_ratio).ceil() as usize;
            ((raw + 63) / 64) * 64
        } else {
            CHUNK_SIZE
        };

        let resampler: Arc<Mutex<Option<FastFixedIn<f32>>>> = Arc::new(Mutex::new(
            if needs_resample {
                Some(
                    FastFixedIn::<f32>::new(
                        resample_ratio,
                        1.0,
                        PolynomialDegree::Linear,
                        resample_input_size,
                        1,
                    )
                    .map_err(|e| AppError::AudioError(format!("resampler init: {e}")))?,
                )
            } else {
                None
            },
        ));

        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(CHUNK_SIZE * 4)));
        let buffer_clone = Arc::clone(&buffer);
        let resampler_clone = Arc::clone(&resampler);
        let running = Arc::clone(&self.running);
        self.running.store(true, Ordering::Relaxed);

        // Pre-input buffer: accumulates device-rate mono samples until we have
        // enough to feed one resampler block.
        let pre_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(resample_input_size * 2)));
        let pre_buf_clone = Arc::clone(&pre_buf);

        let stream = device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !running.load(Ordering::Relaxed) {
                        return;
                    }
                    let mono: Vec<f32> = if channels == 1 {
                        data.to_vec()
                    } else {
                        data.chunks(channels)
                            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                            .collect()
                    };

                    if needs_resample {
                        let mut pre = pre_buf_clone.lock().unwrap();
                        pre.extend_from_slice(&mono);

                        while pre.len() >= resample_input_size {
                            let block: Vec<f32> = pre.drain(..resample_input_size).collect();
                            if let Ok(mut rs_guard) = resampler_clone.lock() {
                                if let Some(rs) = rs_guard.as_mut() {
                                    match rs.process(&[&block], None) {
                                        Ok(out) => {
                                            let mut buf = buffer_clone.lock().unwrap();
                                            buf.extend_from_slice(&out[0]);
                                        }
                                        Err(e) => tracing::warn!("resample error: {e}"),
                                    }
                                }
                            }
                        }
                    } else {
                        let mut buf = buffer_clone.lock().unwrap();
                        buf.extend_from_slice(&mono);
                    }

                    let mut buf = buffer_clone.lock().unwrap();
                    while buf.len() >= CHUNK_SIZE {
                        let chunk: Vec<f32> = buf.drain(..CHUNK_SIZE).collect();
                        callback(chunk);
                    }
                },
                move |err| {
                    tracing::error!("cpal stream error: {err}");
                },
                None,
            )
            .map_err(|e| AppError::AudioError(format!("failed to build stream: {e}")))?;

        stream
            .play()
            .map_err(|e| AppError::AudioError(format!("failed to play stream: {e}")))?;

        self.stream = Some(SendStream(stream));
        Ok(())
    }

    fn stop(&mut self) -> Result<(), AppError> {
        self.running.store(false, Ordering::Relaxed);
        self.stream = None;
        Ok(())
    }
}

/// Streams a WAV file at real-time pace, delivering 512-sample chunks via callback.
///
/// Reads all samples upfront, then streams them in a background thread
/// with 32 ms sleeps between chunks to simulate real-time capture.
/// The last chunk is zero-padded to 512 samples if shorter. No empty
/// sentinel chunk is emitted.
pub struct FileAudioSource {
    path: PathBuf,
    realtime_pacing: bool,
    running: Arc<AtomicBool>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl FileAudioSource {
    /// Creates a new `FileAudioSource`.
    ///
    /// Args:
    ///   path: Path to the WAV file.
    ///   realtime_pacing: Whether to sleep 32 ms between chunks (true for realistic
    ///                    timing, false for fast test execution).
    pub fn new(path: PathBuf, realtime_pacing: bool) -> Self {
        Self {
            path,
            realtime_pacing,
            running: Arc::new(AtomicBool::new(false)),
            thread_handle: None,
        }
    }
}

impl AudioSource for FileAudioSource {
    fn start(&mut self, callback: Box<dyn Fn(Vec<f32>) + Send + 'static>) -> Result<(), AppError> {
        let reader = WavReader::open(&self.path)
            .map_err(|e| AppError::AudioError(format!("failed to open WAV: {e}")))?;

        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .map(|s| s.map_err(|e| AppError::AudioError(format!("read sample: {e}"))))
                .collect::<Result<Vec<f32>, AppError>>()?,
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1u32 << (bits - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .map(|s| {
                        s.map(|v| v as f32 / max_val)
                            .map_err(|e| AppError::AudioError(format!("read sample: {e}")))
                    })
                    .collect::<Result<Vec<f32>, AppError>>()?
            }
        };

        let mono: Vec<f32> = if spec.channels == 1 {
            samples
        } else {
            let ch = spec.channels as usize;
            samples
                .chunks(ch)
                .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                .collect()
        };

        self.running.store(true, Ordering::Relaxed);
        let running = Arc::clone(&self.running);
        let realtime = self.realtime_pacing;

        let handle = thread::spawn(move || {
            let mut offset = 0;
            while offset < mono.len() && running.load(Ordering::Relaxed) {
                let end = (offset + CHUNK_SIZE).min(mono.len());
                let mut chunk = mono[offset..end].to_vec();

                if chunk.len() < CHUNK_SIZE {
                    chunk.resize(CHUNK_SIZE, 0.0);
                }

                callback(chunk);
                offset += CHUNK_SIZE;

                if realtime && offset < mono.len() {
                    thread::sleep(CHUNK_DURATION);
                }
            }
            running.store(false, Ordering::Relaxed);
        });

        self.thread_handle = Some(handle);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), AppError> {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.thread_handle.take() {
            handle.join().map_err(|_| {
                AppError::AudioError("file audio thread panicked".into())
            })?;
        }
        Ok(())
    }
}
