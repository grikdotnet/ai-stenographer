use std::sync::{Arc, Mutex};

use hound::{SampleFormat, WavSpec, WavWriter};
use stt_tauri_client::audio::{AudioSource, CpalAudioSource, FileAudioSource};
use tempfile::NamedTempFile;

const CHUNK_SIZE: usize = 512;
const SAMPLE_RATE: u32 = 16_000;

/// Writes a synthetic mono 16kHz WAV file with the given number of f32 samples.
fn write_test_wav(num_samples: usize) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(tmp.path(), spec).expect("create wav writer");
    for i in 0..num_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        writer.write_sample((440.0 * t * std::f32::consts::TAU).sin()).unwrap();
    }
    writer.finalize().unwrap();
    tmp
}

#[test]
fn file_source_produces_512_sample_chunks() {
    let num_samples = CHUNK_SIZE * 3;
    let tmp = write_test_wav(num_samples);

    let collected: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_clone = Arc::clone(&collected);

    let mut source = FileAudioSource::new(tmp.path().to_path_buf(), false);
    source
        .start(Box::new(move |chunk| {
            collected_clone.lock().unwrap().push(chunk);
        }))
        .expect("start should succeed");

    std::thread::sleep(std::time::Duration::from_millis(500));

    let chunks = collected.lock().unwrap();
    assert_eq!(chunks.len(), 3, "expected exactly 3 full chunks");
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.len(), CHUNK_SIZE, "chunk {} should have 512 samples", i);
    }
}

#[test]
fn file_source_pads_last_short_chunk() {
    let num_samples = CHUNK_SIZE * 2 + 100;
    let tmp = write_test_wav(num_samples);

    let collected: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_clone = Arc::clone(&collected);

    let mut source = FileAudioSource::new(tmp.path().to_path_buf(), false);
    source
        .start(Box::new(move |chunk| {
            collected_clone.lock().unwrap().push(chunk);
        }))
        .expect("start should succeed");

    std::thread::sleep(std::time::Duration::from_millis(500));

    let chunks = collected.lock().unwrap();
    assert_eq!(chunks.len(), 3, "expected 2 full + 1 padded chunk");

    for chunk in chunks.iter() {
        assert_eq!(chunk.len(), CHUNK_SIZE);
    }

    let last = &chunks[2];
    assert!(
        last[100..].iter().all(|&s| s == 0.0),
        "trailing samples in padded chunk should be zero"
    );
}

#[test]
fn file_source_stops_after_exhaustion() {
    let num_samples = CHUNK_SIZE;
    let tmp = write_test_wav(num_samples);

    let collected: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_clone = Arc::clone(&collected);

    let mut source = FileAudioSource::new(tmp.path().to_path_buf(), false);
    source
        .start(Box::new(move |chunk| {
            collected_clone.lock().unwrap().push(chunk);
        }))
        .expect("start should succeed");

    std::thread::sleep(std::time::Duration::from_millis(300));

    let count_after_done = collected.lock().unwrap().len();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let count_later = collected.lock().unwrap().len();

    assert_eq!(count_after_done, count_later, "no new chunks after exhaustion");
    assert_eq!(count_later, 1, "exactly one chunk from 512-sample file");
}

#[test]
fn file_source_no_empty_trailing_chunk() {
    let num_samples = CHUNK_SIZE * 2;
    let tmp = write_test_wav(num_samples);

    let collected: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_clone = Arc::clone(&collected);

    let mut source = FileAudioSource::new(tmp.path().to_path_buf(), false);
    source
        .start(Box::new(move |chunk| {
            collected_clone.lock().unwrap().push(chunk);
        }))
        .expect("start should succeed");

    std::thread::sleep(std::time::Duration::from_millis(500));

    let chunks = collected.lock().unwrap();
    for chunk in chunks.iter() {
        assert!(!chunk.is_empty(), "no empty chunks should be produced");
    }
    assert_eq!(chunks.len(), 2);
}

#[test]
fn file_source_error_on_missing_file() {
    let mut source = FileAudioSource::new("definitely_not_a_real_file.wav".into(), false);
    let result = source.start(Box::new(|_| {}));
    assert!(result.is_err(), "should error on non-existent file");
}

#[test]
fn file_source_all_chunks_contain_signal_data() {
    let num_samples = CHUNK_SIZE * 2;
    let tmp = write_test_wav(num_samples);

    let collected: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_clone = Arc::clone(&collected);

    let mut source = FileAudioSource::new(tmp.path().to_path_buf(), false);
    source
        .start(Box::new(move |chunk| {
            collected_clone.lock().unwrap().push(chunk);
        }))
        .expect("start should succeed");

    std::thread::sleep(std::time::Duration::from_millis(500));

    let chunks = collected.lock().unwrap();
    for (i, chunk) in chunks.iter().enumerate() {
        let has_nonzero = chunk.iter().any(|&s| s != 0.0);
        assert!(has_nonzero, "chunk {} should contain actual signal", i);
    }
}

#[test]
fn file_source_stop_halts_streaming() {
    let num_samples = CHUNK_SIZE * 100;
    let tmp = write_test_wav(num_samples);

    let collected: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_clone = Arc::clone(&collected);

    let mut source = FileAudioSource::new(tmp.path().to_path_buf(), true);
    source
        .start(Box::new(move |chunk| {
            collected_clone.lock().unwrap().push(chunk);
        }))
        .expect("start should succeed");

    std::thread::sleep(std::time::Duration::from_millis(150));
    source.stop().expect("stop should succeed");

    let count_at_stop = collected.lock().unwrap().len();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let count_after = collected.lock().unwrap().len();

    assert!(count_at_stop > 0, "should have received some chunks");
    assert!(
        count_after <= count_at_stop + 1,
        "at most one extra chunk after stop (in-flight)"
    );
}

#[test]
#[ignore = "requires audio hardware"]
fn cpal_source_construction_does_not_panic() {
    let _source = CpalAudioSource::new();
}
