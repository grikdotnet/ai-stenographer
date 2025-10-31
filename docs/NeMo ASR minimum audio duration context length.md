## Segment Duration Resources Research

1. Single Words vs. Longer Utterances
Source: NeMo GitHub Issue #541 (2021) Finding:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html
"Reduced accuracy when inferring single word audio files with QuartzNet or Jasper compared to longer sentences"
Reason: "Context helps with recognition... Jasper and QuartzNet implicitly learn pretty good language models"
due to lack of linguistic context
Conclusion: Single-word accuracy IS lower, the reason is implicit language modeling.

2. Optimal Context Length for Transformer ASR
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html
Source: "How Much Context Does My Attention-Based ASR System Need?" (2023) Key Findings:
Baseline training: 10 seconds of context
Maximum benefit: Up to 21.8 minutes of context (14.5% relative improvement)
Practical sweet spot: 5.5 minutes yields "the majority of the WERR improvement"
Minimum benefit threshold: "No significant benefit to training on sequences longer than 20 seconds" (Rev16 dataset)
Academic datasets:
"Typically provided as short utterances (1-20 seconds)"
Previous work deals with "fairly short sequences of 20-30 seconds"
Conclusion: Optimal range is 1-20 seconds, with diminishing returns beyond 5.5 minutes.

3. Minimum Duration in NeMo Configuration
Source: NeMo ASR Configuration Files (2024) Finding:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html
Training configs typically set min_duration: 0.1 seconds (100ms)
No mention of <200ms being "very poor"
Focus is on batch processing, not accuracy thresholds
Conclusion: NeMo supports utterances as short as 100ms, no evidence of specific accuracy cliffs at 200ms.

4. Parakeet Model Capabilities
Source: NVIDIA Parakeet Blog Post (2024) Finding:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html
Designed for "audio segments up to 11 hours"
"Highly flexible, capable of processing anything from short audio clips to extremely long segments"
NO specific minimum duration mentioned
Conclusion: Parakeet is designed for long-form audio, but no minimum duration specified.

________________________________________________________________
Based on actual research:
Duration	Evidence-Based Assessment	Source
<100ms	Below NeMo's min_duration config	NeMo Config Docs
100ms-1s	Supported but lacks linguistic context	NeMo Issue #541
1-20s	Standard range for ASR datasets	ArXiv 2310.15672v2
1-5.5min	Optimal for transformer ASR (majority of improvement)	ArXiv 2310.15672v2
>5.5min	Diminishing returns, but up to 21.8min helps in noisy conditions	ArXiv 2310.15672v2
