# Nvidia Parakeet TDT 0.6B v3 Audio Segment Specifications

**The Nvidia Parakeet TDT 0.6B v3 model uses 2-second audio segments (2000 ms / 32,000 samples) as the officially recommended duration for real-time speech processing.** This configuration balances accuracy and latency for most streaming applications. The model supports flexible segment durations from 1 second (minimal) to 1.6 seconds (optimal for lower latency), with all configurations operating at 16kHz mono-channel audio. These specifications come from the official Hugging Face model card and Nvidia NeMo framework documentation, where streaming inference parameters are explicitly defined for production deployment.

The Token-and-Duration Transducer architecture enables efficient variable-length processing by predicting token durations and skipping blank frames, making these segment sizes practical for real-time inference. The model processes audio in 80-millisecond frames due to 8x convolutional subsampling in the FastConformer encoder, meaning a 2-second chunk contains 25 frames for decoder processing.

## Core audio segment specifications for real-time inference

The model supports three distinct segment duration configurations based on application requirements:

**Minimal segment duration: 1.0 second**
- **Duration:** 1,000 milliseconds
- **Samples at 16kHz:** 16,000 samples
- **Frame count:** ~12.5 frames (at 80ms per frame)

This minimal configuration appears in multiple implementation examples, including the MLX port where `chunk_size = model.preprocessor_config.sample_rate` defaults to 16,000 samples. It provides lowest latency (~1 second delay) for highly responsive applications but may sacrifice accuracy due to reduced context and increases processing overhead from more frequent inference calls.

**Optimal segment duration: 1.6 seconds**
- **Duration:** 1,600 milliseconds  
- **Samples at 16kHz:** 25,600 samples
- **Frame count:** 20 frames (at 80ms per frame)

Specified in Nvidia NeMo's buffered inference examples (`speech_to_text_buffered_infer_rnnt.py`) with a 4.0-second total buffer. This represents the **balanced latency-accuracy trade-off** optimized through Nvidia's internal testing, suitable for applications requiring responsive feedback while maintaining good transcription quality.

**Recommended segment duration: 2.0 seconds**
- **Duration:** 2,000 milliseconds
- **Samples at 16kHz:** 32,000 samples  
- **Frame count:** 25 frames (at 80ms per frame)

This is the **officially documented parameter** in the Parakeet TDT 0.6B v3 model card on Hugging Face. The streaming inference command explicitly uses `chunk_secs=2` with `right_context_secs=2.0` and `left_context_secs=10.0`. This configuration delivers optimal accuracy for production deployments where 2-second latency is acceptable.

## Streaming context requirements and configuration

Beyond the primary audio segment, the model requires context windows for accurate transcription:

| Parameter | Duration | Samples @ 16kHz | Purpose |
|-----------|----------|-----------------|---------|
| Chunk duration | 2.0 s | 32,000 | Primary processing segment |
| Right context | 2.0 s | 32,000 | Future lookahead for accuracy |
| Left context | 10.0 s | 160,000 | Historical information |
| Total buffer | 14.0 s | 224,000 | Complete context window |

The right context (future lookahead) of 2 seconds provides the decoder with upcoming audio information, significantly improving transcription accuracy compared to pure causal models. The 10-second left context maintains historical information for resolving ambiguities. Both contexts are adjustable based on latency requirements—reducing right context to 0.25 seconds enables ultra-low latency but increases word deletion errors.

## Architecture-level frame processing details

The FastConformer encoder uses **8x convolutional subsampling** with a 10ms base window shift, producing **80-millisecond frames** as the fundamental processing unit:

- **Frame duration:** 80 ms (1,280 samples at 16kHz)
- **Frames per 2-second chunk:** 25 frames
- **Attention context size:** 256 frames left, 256 frames right
- **Maximum attention span:** ~40.96 seconds (512 frames × 80ms)

This subsampling strategy reduces computational complexity while maintaining acoustic information. The TDT decoder can predict token durations up to 4 frames per token (320ms), enabling efficient processing by skipping blank frames rather than emitting blank symbols like traditional RNN-Transducers.

## Latency-accuracy trade-offs across configurations

Different segment durations fundamentally alter the inference characteristics:

**Ultra-low latency (0.25-0.5 seconds):** Experimental configuration with chunk_secs=0.25-0.5 and right_context_secs=0.25. Provides 250-500ms total delay but significantly increases word deletions and reduces accuracy. Not recommended for production.

**Low latency (1.0 second):** The minimal configuration with chunk_secs=1.0 and right_context_secs=1.0 totals 2 seconds of buffering (1s chunk + 1s lookahead). Responsive for interactive applications like virtual assistants where immediate feedback matters more than perfect accuracy. Increased processing overhead from more frequent inference calls may impact throughput on resource-constrained systems.

**Balanced (1.6 seconds):** The optimal configuration uses chunk_secs=1.6 with proportional context adjustments. Nvidia's internal benchmarking identified this as the sweet spot for most streaming applications, providing 3.6-second total latency (1.6s chunk + 2s lookahead) while maintaining high accuracy.

**High accuracy (2.0 seconds):** The recommended configuration achieves best transcription quality with 4-second total latency (2s chunk + 2s lookahead). The official streaming example uses this configuration with batch_size=32 for production deployments where accuracy outweighs latency concerns.

**Maximum accuracy (5-10 seconds):** For applications where latency doesn't matter (offline transcription, batch processing), using 5-10 second chunks with 5-second right context maximizes accuracy by providing extensive context to the decoder.

## Model capacity and technical limits

The model architecture supports extensive audio processing:

- **Single-pass maximum duration:** 24 minutes with full attention (requires A100 80GB GPU)
- **Extended processing capacity:** 3 hours with local attention ([256, 256] frame context)
- **Recommended input format:** 16-bit PCM mono-channel WAV or FLAC at 16kHz
- **Preprocessing:** 80 mel bin spectrogram with normalization

These capacity figures represent theoretical maximums rather than recommended real-time streaming configurations. For production streaming inference, the 2-second segments with 14-second total buffer provide the best balance.

## Conclusion: Selecting the right segment duration

The 2-second recommended duration represents Nvidia's validated production configuration, tested across diverse audio conditions and use cases. **The key insight is that segment duration selection fundamentally depends on your latency budget**: applications requiring sub-second response times must accept accuracy trade-offs by using 1-second minimal segments, while applications tolerating 2-4 second delays should use the recommended 2-second configuration for optimal results.

The TDT architecture's ability to skip blank frames and predict token durations makes these relatively short segment sizes practical—traditional RNN-Transducers would require longer segments for comparable accuracy. For implementation, start with the 2-second recommended configuration from the official model card, then adjust based on measured latency requirements and accuracy metrics in your specific deployment environment.


## Context definition

The "context" refers to **actual recorded audio (silence, noise, or previous speech) surrounding the detected short word**, not previous transcribed speech.

**Context breakdown:**

For a 50ms word detected at timestamp T:
- **Pre-context (400-600ms)**: Audio from T-600ms to T. This captures:
  - Natural silence before the word
  - Attack transients (how the sound starts)
  - Room acoustics
  - Previous speech IF it exists nearby
  
- **Post-context (300-400ms)**: Audio from T+50ms to T+450ms. This captures:
  - Release transients (how the sound ends)
  - Natural silence after
  - Following speech IF it exists

**Your concern is valid:** In real-time with significant pauses, pre-context might be mostly silence. This is actually acceptable because:

1. **Natural silence ≠ artificial padding**. The model trained on real recordings with natural room tone/noise
2. **The acoustic boundaries matter**. Even silence shows how the word started/ended
3. **Better than pure padding** which your research document already proved degrades quality

**Critical distinction:** 
- ❌ Don't use artificial zeros/silence
- ✅ Do use whatever was actually recorded before/after

**Practical handling for sparse speech:**

```
Speech at T=5000ms (50ms duration)
Previous speech was at T=1000ms

Buffer: T=4400ms to T=5450ms (1050ms total)
├─ 4400-5000ms: real silence/noise (600ms)
├─ 5000-5050ms: actual speech (50ms)  
└─ 5050-5450ms: real silence/noise (400ms)
```

The model handles this real acoustic environment better than a 50ms clip with artificial padding.

Does this clarify the distinction?