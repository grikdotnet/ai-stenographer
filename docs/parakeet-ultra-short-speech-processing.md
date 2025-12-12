# Processing ultra-short speech with Parakeet TDT: architectural limits and practical solutions

The fundamental challenge you're facing stems from **Parakeet TDT's 8x downsampling architecture**, which mathematically prevents reliable transcription of 50-128ms segments. With standard 10ms frame stride, each encoded frame represents 80ms of audio—meaning your 50ms segments produce less than one frame and 128ms segments yield barely 1.6 frames after encoding. This architectural constraint, combined with the model's documented tendency to hallucinate common backchannels like "yeah" and "mm-hmm" on silence, creates a problem that cannot be fully solved through configuration alone but requires a strategic pipeline redesign combining intelligent VAD, buffering, and confidence filtering.

Research across academic papers, production implementations, and comparative ASR systems reveals that **no modern streaming ASR model reliably processes segments below 200-300ms** without significant accuracy degradation. However, several proven techniques can mitigate these constraints for real-time applications, particularly when you can buffer to 128ms with surrounding context. The most effective approach combines pre-filtering silence before ASR (preventing hallucinations), dynamic VAD configuration optimized for short word capture, temporal buffering strategies that accumulate context, and entropy-based confidence thresholding to reject unreliable predictions.

## Why Parakeet TDT fails below 1000ms: the downsampling bottleneck

The FastConformer encoder in Parakeet TDT implements aggressive 8x depthwise convolutional subsampling early in the processing pipeline. This architectural choice delivers substantial benefits—approximately 2.4-2.8x speedup over standard Conformer with 4x reduction in memory for attention layers—but creates hard constraints on minimum viable segment length. With each output frame representing 80ms of audio after downsampling, the model requires roughly 12-15 encoded frames (approximately 1000ms) to provide sufficient temporal context for the Token-Duration Transducer decoder.

The TDT decoder architecture compounds this challenge. Unlike standard RNN-Transducers, TDT jointly predicts both tokens and their durations, with frame-skipping capability of up to 4 frames based on predicted duration. **When processing 50-128ms audio that yields only 1-2 encoder frames, the decoder lacks minimal temporal context for reliable duration prediction**, leading the model to "fill in" with high-frequency training patterns—specifically common backchannels like "yeah" and "mm-hmm" that appeared frequently in the 1.7 million hours of training data.

NVIDIA's technical report (arXiv 2509.14128) confirms the company added non-speech audio during training explicitly to reduce hallucinations. However, this training-level mitigation cannot overcome architectural limitations. The self-attention mechanism with limited context attention still requires minimum frames for meaningful computation, and extremely short segments (under 200ms) remain underrepresented in the training distribution despite these improvements.

## VAD strategies that actually capture short words

Modern neural VAD systems can detect speech at remarkably fine temporal resolution, but configuration makes the critical difference between capturing genuine short words versus generating false positives that trigger hallucinations. **Silero VAD emerges as the optimal choice** from production implementations, offering 10ms frame-level detection with configurable minimum speech duration as low as 50ms—compared to typical defaults of 250ms.

The key to effective short-word capture lies in asymmetric threshold configuration. Set the activation threshold relatively low (0.4-0.5 versus the default 0.5) to ensure sensitivity to brief utterances, while maintaining a higher minimum silence duration (300-600ms) to prevent oversegmentation during natural pauses. This creates a system that detects short words reliably while avoiding the fragmentation that would send numerous tiny, context-free segments to the ASR model.

Research on VAD post-processing reveals that temporal smoothing and duration thresholds serve as the second line of defense. Apply median filtering to eliminate spurious detections, then merge segments separated by less than 300ms to maintain natural phrase boundaries. For segments that pass initial duration checks but fall below 500ms, require higher VAD confidence scores (0.9 versus typical 0.5-0.7) before sending to ASR. This progressive confidence adjustment prevents the pipeline from processing noise or breath sounds that might trigger hallucinations.

Production implementations consistently demonstrate that **pre-filtering silence before ASR prevents hallucinations more effectively than any post-processing approach**. The Modal Parakeet production example explicitly uses pydub's `detect_nonsilent()` function before transcription specifically because Parakeet hallucinates on silent inputs. This architectural pattern—silence detection as a gate before ASR rather than relying on the model to handle silence—proves critical for managing the documented "yeah" and "mm-hmm" hallucination issue.

## Buffering and accumulation techniques for context preservation

The tension between your 50-128ms segments and Parakeet's architectural requirements demands sophisticated buffering strategies that accumulate temporal context without introducing unacceptable latency. The most successful pattern from NeMo implementations uses overlapping buffers with symmetric context windows: maintain a 3-second rolling buffer, process 1-second chunks from the middle, and provide 1 second of context on each side. This approach ensures every transcription request includes sufficient encoder frames (approximately 12-13 frames from the 1-second chunk) while maintaining reasonable latency.

For your specific constraint of 50-128ms detected speech, **the optimal strategy involves delayed processing with context accumulation**. When VAD detects a very short segment, don't immediately send it to ASR. Instead, buffer it with 800-1200ms of surrounding audio context (400-600ms before and after), creating an effective processing window of 1.0-1.4 seconds. Research on dynamic chunk training demonstrates that models trained on varying chunk sizes (320-1280ms) can handle this approach effectively, with less than 1% WER degradation when the actual speech occupies only a small portion of the processing window.

The Temporal Alignment Buffer (TAB) technique from recent research (2025) offers a more sophisticated approach for extremely short segments. By introducing a controlled delay range (80ms proves optimal), the system can select the chunk with minimal divergence loss within the delay window. This overcomes cold-start limitations when initial context is insufficient, achieving **comparable performance to 320ms chunk models while operating at 40ms latency**—a 6-9% relative improvement over standard frame-by-frame approaches.

Context-sensitive chunking with simulated future frames represents the cutting edge. The CUSIDE framework uses a GRU-based simulation network to predict future contextual frames rather than waiting for them to arrive, maintaining streaming capability while improving accuracy. This technique, originally developed for Conformer-CTC models, could potentially be adapted to Parakeet's FastConformer architecture, though it would require custom implementation.

## Confidence thresholding to filter hallucinations and unreliable predictions

Entropy-based confidence estimation provides far more reliable quality assessment than raw probability scores from the model. Research from NVIDIA demonstrates that **Tsallis entropy with α=1/3 proves 4x better at detecting incorrect words** compared to maximum probability, addressing the overconfidence problem inherent in CTC and RNN-T models. This approach can filter approximately 40% of hallucinations at the cost of rejecting only 5% of correct predictions.

For word-level confidence aggregation from frame predictions, the minimum method consistently outperforms mean or product approaches. Taking the minimum confidence across all frames for a word ensures that any uncertain frame appropriately reduces the overall confidence score, minimizing false positives for incorrect transcriptions. Combined with optimized thresholds (typically 0.65-0.7 determined through F1-score optimization on validation data), this creates an effective filter for unreliable short-segment predictions.

Dynamic confidence thresholding based on segment characteristics adds another layer of quality control. Implement progressive adjustment where segments under 500ms require higher confidence (0.9 versus standard 0.7) before acceptance. For segments under 3 words in length with confidence below 0.7, automatically flag for review or rejection. **Maintain a "Bag of Hallucinations" blacklist** containing the known false outputs your system produces—minimally "yeah", "mm-hmm", "uh-huh", and any other backchannel tokens you observe—and automatically reject these when they appear as isolated predictions on short segments.

The compression ratio check provides an independent validation mechanism. Calculate words per second for each transcription, flagging any result exceeding 5 words per second as physically implausible and likely hallucinated. Similarly, examine word-level timestamps when available: any word with duration under 50ms (except articles "a", "I") indicates probable hallucination. These heuristics, combined with entropy-based confidence, create a robust multi-layer filtering system.

## Trade-offs between latency and accuracy for ultra-short speech

The fundamental tension between processing speed and transcription quality intensifies dramatically for sub-second segments. Industry benchmarks reveal that streaming ASR systems typically suffer a **20-25% accuracy penalty versus offline counterparts**, with this gap widening substantially for very short utterances. However, recent advances demonstrate that careful engineering can minimize this trade-off.

For segments in your 50-128ms range, attempting real-time processing (under 100ms latency) will likely yield WER above 25-30%, making transcriptions marginally useful. **The practical sweet spot balances 300-500ms total latency with WER in the 12-18% range**—achieved through buffering to accumulate 800-1200ms effective duration (including context), processing with optimized models, and applying confidence filtering. This represents acceptable accuracy for most conversational AI applications while maintaining responsiveness that feels natural in dialogue.

Chunk size selection directly impacts this trade-off. Research across multiple streaming ASR systems shows consistent patterns: 320ms chunks introduce +2-3% relative WER but enable 40-100ms latency; 640ms chunks add +1-2% relative WER with 100-200ms latency; 1280ms chunks approach baseline performance with 200-400ms latency. For your constraint of capturing 50-128ms speech, the optimal approach involves detecting these short segments with sensitive VAD but processing them with 640-800ms effective windows through context buffering.

The cold-start problem—processing speech without sufficient prior context—particularly affects very short first utterances in a conversation. Attention sink strategies mitigate this by allowing attention to the first few frames throughout processing, reducing computation while maintaining accuracy. Alternatively, warm-start techniques using lightweight pre-processing on the first 50-100ms to generate initial context can reduce cold-start penalty from 25% to 5-10% WER degradation.

## Batch processing and multi-pass strategies

For a real-time constraint with VAD-based segmentation, **the most practical approach involves pseudo-batch processing through segment accumulation**. Rather than processing each detected 50-128ms segment individually, accumulate segments until you reach 800-1200ms total duration (respecting natural pause boundaries from VAD), then process as a single batch. This maintains the responsiveness benefit of stream processing while providing the accuracy advantage of longer context. The Modal production implementation demonstrates this pattern, using pydub to detect non-silent chunks, then batching them for processing.

## Recommended architecture for your specific constraints

Given your requirements—50-128ms speech detection, 128ms maximum buffering, real-time processing with VAD—the optimal pipeline combines sensitive detection with strategic processing delays. **Start with Silero VAD configured for maximum sensitivity**: set `min_speech_duration=50ms`, `activation_threshold=0.4`, `min_silence_duration=300ms`, and `speech_pad_ms=30`. This captures genuine short words while filtering obvious non-speech.

When VAD detects a segment, don't immediately send to Parakeet. Instead, implement a two-tier buffering strategy. For segments under 300ms, automatically buffer with 500ms of preceding context and 300ms of following context (using your 128ms maximum for the following portion if that's a hard constraint), creating an 800-1100ms processing window. **Pre-filter this buffered audio with pydub's `detect_nonsilent()` to verify actual speech content**, preventing the hallucination issue. Only if speech is confirmed should you proceed to ASR.

For Parakeet TDT, use the buffered inference with overlapping chunks approach from the RNN-T streaming examples.
Extract not just the transcription but frame-level confidence scores. Calculate Tsallis entropy (α=1/3) for word-level confidence, aggregating with the minimum method. Apply your threshold: require 0.9 confidence for segments under 500ms effective duration, 0.7 for longer segments. Automatically reject isolated occurrences of "yeah", "mm-hmm", "uh-huh" when they appear alone with low confidence.

## Alternative approaches if architectural constraints prove insurmountable

If Parakeet TDT's 8x downsampling fundamentally limits your application, consider **FastConformer Hybrid models with 4x downsampling** instead of the standard 8x. This would double your effective temporal resolution, giving 50ms audio approximately 1.25 frames and 128ms audio approximately 3.2 frames—marginal but meaningful improvement. However, this requires either model retraining or switching to a different pre-trained variant, accepting potential accuracy trade-offs.

CTC-based models represent a more practical alternative for ultra-short segments. Conformer-CTC or FastConformer-CTC architectures handle brief utterances more gracefully than transducer models, with frame-independence preventing the duration prediction problems inherent in TDT. NVIDIA provides several CTC variants in the NeMo framework that could serve as drop-in replacements, accepting slightly higher WER on longer utterances in exchange for more stable short-segment performance.

For the most latency-critical applications, consider specialized models like CAIMAN-ASR or Deepgram Flux that achieve 260-300ms end-to-end latency through FPGA optimization or aggressive GPU optimization. These systems are explicitly engineered for the ultra-low-latency streaming use case you describe, accepting 3-5% relative WER increase compared to offline models in exchange for responsiveness. However, they may still implement internal minimum duration thresholds around 300ms.

## Production implementation priorities

Begin with the highest-impact, lowest-complexity changes. **First, implement silence pre-filtering before ASR**—this single change addresses the documented hallucination issue more effectively than any other intervention. Second, configure Silero VAD with parameters optimized for short word detection while maintaining high specificity. Third, establish the buffering pipeline that ensures Parakeet never processes segments with fewer than 10-12 encoded frames after downsampling.

Layer confidence filtering as your next priority. Integrate Tsallis entropy calculation into your post-processing pipeline, establish thresholds through validation data, and implement the dynamic adjustment based on segment duration. Maintain your hallucination blacklist and update it as you observe system behavior in production. These steps require minimal architectural changes while providing substantial quality improvement.

Monitor segment duration distribution continuously in production. Track WER by duration bucket (under 300ms, 300-500ms, 500-1000ms, over 1000ms) to understand where your system struggles. Similarly track hallucination rate, particularly the "yeah" and "mm-hmm" patterns you've already identified. Use these metrics to refine VAD parameters and confidence thresholds iteratively.

The fundamental reality you're confronting is that **Parakeet TDT's architecture was not designed for 50-128ms segments, and no amount of configuration will make this optimal**. However, the combination of intelligent VAD, strategic buffering, silence pre-filtering, and confidence thresholding can create a production-viable system that captures short words while maintaining acceptable accuracy and latency. The key lies in treating 50-128ms as a detection constraint rather than a processing constraint, accumulating sufficient context before invoking the ASR model itself.

___

## Context definition

The "context" here refers to **actual recorded audio (silence, noise, or previous speech) surrounding the detected short word**, not previous transcribed speech.

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

In real-time with significant pauses, pre-context might be mostly silence. This is actually acceptable because:

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

---

## Tsallis entropy

**Tsallis entropy** is used to measure prediction uncertainty in ASR models.

**Formula:**

```
H_α(P) = (1 - Σ(p_i^α)) / (α - 1)
```

Where:
- `P` = probability distribution over tokens at each frame
- `p_i` = probability of token i
- `α` = entropy parameter (NVIDIA found **α=1/3** optimal)

**Why it works better:**

Standard probability: Model outputs 0.95 confidence but is wrong 40% of the time (overconfident)

Tsallis entropy: Penalizes overconfident wrong predictions, detects errors **4x better** than raw probability

**Word-level aggregation with minimum method:**

For a word spanning frames 10-15:

```python
# Each frame produces entropy score
frame_entropies = [0.85, 0.92, 0.78, 0.88, 0.91, 0.82]

# Minimum method: take LOWEST confidence
word_confidence = min(frame_entropies)  # = 0.78
```

**Why minimum vs mean/product:**

- **Mean**: One bad frame gets averaged out (hides errors)
- **Product**: Multiple frames compound (too aggressive)
- **Minimum**: Any uncertain frame flags the whole word (conservative, accurate)

**Practical implementation:**

```python
import numpy as np

def tsallis_entropy(probs, alpha=1/3):
    """Calculate Tsallis entropy from frame probabilities"""
    return (1 - np.sum(probs ** alpha)) / (alpha - 1)

def word_confidence(frame_probs, method='min'):
    """Aggregate frame-level confidences to word level"""
    entropies = [tsallis_entropy(p) for p in frame_probs]
    
    if method == 'min':
        return min(entropies)  # Most conservative
    elif method == 'mean':
        return np.mean(entropies)
    else:
        return np.prod(entropies)
```

**Apply thresholds:**

```python
if segment_duration < 500:  # ms
    threshold = 0.9
else:
    threshold = 0.7

if word_confidence < threshold:
    reject_transcription()
```

**Getting frame probabilities from Parakeet:**

NeMo models can return frame-level logits/probabilities when configured - you'd need to enable this in your inference setup.

**Source:** [NVIDIA: Entropy-Based ASR Confidence Estimation](https://developer.nvidia.com/blog/entropy-based-methods-for-word-level-asr-confidence-estimation/)

---

## References

### Parakeet TDT & NVIDIA Documentation
- [NVIDIA Speech AI Models Performance](https://developer.nvidia.com/blog/nvidia-speech-ai-models-deliver-industry-leading-accuracy-and-performance/)
- [Parakeet-TDT Architecture Deep Dive](https://www.qed42.com/insights/nvidia-parakeet-tdt-0-6b-v2-a-deep-dive-into-state-of-the-art-speech-recognition-architecture)
- [Turbocharge ASR with Parakeet-TDT](https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt)
- [Parakeet TDT 1.1B Model Card](https://huggingface.co/nvidia/parakeet-tdt-1.1b)
- [Parakeet TDT 0.6B v2 Discussions](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/16)
- [Canary-1B-v2 & Parakeet-TDT-0.6B-v3 Paper](https://arxiv.org/abs/2509.14128)

### Streaming ASR & Implementation Examples
- [Modal: Streaming Parakeet Transcription](https://modal.com/docs/examples/streaming_parakeet)
- [Modal: Real-time Parakeet Transcription](https://modal.com/docs/examples/parakeet)
- [Building Local STT with Parakeet](https://towardsai.net/p/artificial-intelligence/%EF%B8%8F-building-a-local-speech-to-text-system-with-parakeet-tdt-0-6b-v2)
- [NeMo Streaming ASR Tutorial](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Streaming_ASR.ipynb)

### Voice Activity Detection (VAD)
- [SpeechBrain VAD Documentation](https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/voice-activity-detection.html)
- [VAD: A Lazy Data Science Guide](http://mohitmayank.com/a_lazy_data_science_guide/audio_intelligence/voice_activity_detection/)
- [One Voice Detector to Rule Them All](https://thegradient.pub/one-voice-detector-to-rule-them-all/)
- [Semantic VAD Paper](https://ar5iv.labs.arxiv.org/html/2305.12450)
- [Android VAD Library](https://github.com/gkonovalov/android-vad)
- [VAD Parameter Adjustment](https://pyvideotrans.com/en/vad)
- [NeMo Speaker Diarization Configs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html)

### Confidence Estimation & Hallucination
- [NVIDIA: Entropy-Based ASR Confidence](https://developer.nvidia.com/blog/entropy-based-methods-for-word-level-asr-confidence-estimation/)
- [ASR Confidence Scores Evaluation](https://arxiv.org/html/2503.15124v1)
- [Whisper Hallucinations Investigation](https://arxiv.org/html/2501.11378v1)
- [Whisper Hallucinations Dataset](https://huggingface.co/datasets/sachaarbonel/whisper-hallucinations)
- [Reducing Whisper Hallucinations](https://github.com/open-webui/open-webui/discussions/8118)

### Short Utterance Handling & ASR Architecture
- [Turning Whisper into Real-Time System](https://arxiv.org/html/2307.14743)
- [Whisper Streaming Repository](https://github.com/ufal/whisper_streaming)
- [Whisper vs Real-Time Alternatives](https://picovoice.ai/blog/whisper-speech-to-text-for-real-time-transcription/)
- [Wav2Vec2 Padding Problems](https://github.com/facebookresearch/fairseq/issues/3118)
- [Conformer Short Utterance Issues](https://github.com/espnet/espnet/issues/2893)
- [Context Requirements in ASR](https://arxiv.org/html/2310.15672v2)
- [Real-Time Transcription Evaluation](https://arxiv.org/html/2409.05674v1)

### Advanced Streaming Techniques
- [CUSIDE-T: Chunking & Simulating Future](https://arxiv.org/html/2407.10255v1)
- [CUSIDE-T Paper](https://arxiv.org/abs/2407.10255)
- [Hybrid Transducer and Attention](https://aclanthology.org/2023.acl-long.695.pdf)
- [End-to-End ASR Architectures Survey](https://www.assemblyai.com/blog/a-survey-on-end-to-end-speech-recognition-architectures-in-2021)
- [CAIMAN-ASR Low-Latency Streaming](https://myrtle.ai/resources/caiman-asr-pushing-the-boundaries-of-low-latency-streaming-speech-recognition/)

### NVIDIA NeMo Framework
- [NeMo ASR Models Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- [NVIDIA Riva ASR Customization](https://docs.nvidia.com/nim/riva/asr/latest/customization.html)
- [NVIDIA Riva ASR Overview](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html)

### Optimization & Production Systems
- [Gladia: Optimizing Whisper for Scale](https://www.gladia.io/blog/heres-how-we-optimized-whisper-asr-for-scale)
- [ASR Playground Repository](https://github.com/garbit/asr_playground)
- [WhisperX Repository](https://github.com/NicholasTheGreat/WhisperX)

### Academic Research
- [Hybrid CTC/Attention for Agglutinative Languages](https://pmc.ncbi.nlm.nih.gov/articles/PMC9571619/)
- [End-to-End ASR Introduction](https://blog.paperspace.com/end-to-end-automatic-speech-recognition/)
- [ACL: Hybrid Architecture Paper](https://aclanthology.org/2023.acl-industry.26.pdf)
