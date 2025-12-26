# FP16 Model Conversion Guide

## Overview

This document describes how to convert and use FP16 (16-bit floating point) versions of the Parakeet STT models, reducing model size by approximately 50% while maintaining accuracy.

## Conversion Results

Successfully converted models:
- **encoder-model.fp16.onnx**: 1.2GB (originally 2.36GB)
- **decoder_joint-model.fp16.onnx**: 35MB (originally 69MB)
- **Total savings**: ~1.2GB

## Converting Models to FP16

### Prerequisites

Install required packages:
```bash
pip install onnx onnxconverter-common
```

These are already included in `requirements.txt`.

### Running the Conversion

**Step 1:** Convert all Parakeet models to FP16:

```bash
python convert_to_fp16.py
```

**Step 2:** Fix Cast operations for compatibility:

```bash
python fix_fp16_casts.py
```

This fixes internal Cast operations that cause type mismatches while preserving output types.

With verbose output:

```bash
python convert_to_fp16.py -v
python fix_fp16_casts.py -v
```

Custom input directory:

```bash
python convert_to_fp16.py --input-dir ./models/parakeet
python fix_fp16_casts.py --input-dir ./models/parakeet
```

### How the Conversion Works

The process involves two steps:

**Step 1: FP32 → FP16 Conversion** ([convert_to_fp16.py](convert_to_fp16.py))

Uses `onnxconverter-common` to convert model weights:

1. **Loads the original FP32 model** (including external data files for large models)
2. **Converts weights to FP16** using `float16.convert_float_to_float16()`
   - Uses `disable_shape_infer=True` to preserve weights in models with external data
   - Uses `keep_io_types=True` to keep I/O types as FP32
3. **Saves the FP16 model** as a single file

**Step 2: Fix Cast Operations** ([fix_fp16_casts.py](fix_fp16_casts.py))

Fixes type mismatches that cause ONNX Runtime errors:

1. **Identifies problematic Cast operations** that still cast to FP32 in FP16 graphs
2. **Converts internal Casts** from `Cast(to=FLOAT)` to `Cast(to=FLOAT16)`
3. **Preserves output Casts** to maintain FP32 outputs for compatibility
4. **Handles FP16 inputs** by converting them back to FP32 with internal Cast nodes

The fix resolves errors like:
```
Type parameter (T) of Optype (Add) bound to different types
(tensor(float) and tensor(float16))
```

## Using FP16 Models

### With onnx_asr quantization parameter

The `onnx_asr` library supports loading FP16 models directly via the `quantization` parameter:

```python
import onnx_asr

model = onnx_asr.load_model(
    'nemo-parakeet-tdt-0.6b-v3',
    './models/parakeet',
    quantization='fp16',  # Load FP16 models
    preprocessor_config={'device': 'gpu'},
    resampler_config={'device': 'gpu'}
)
```

This automatically loads `encoder-model.fp16.onnx` and `decoder_joint-model.fp16.onnx`.

### In the STT Pipeline

Update [pipeline.py](src/pipeline.py#L46) to use FP16:

```python
# Original (FP32):
self.model: Any = onnx_asr.load_model(
    "nemo-parakeet-tdt-0.6b-v3",
    model_path,
    providers=providers,
    sess_options=sess_options,
    preprocessor_config={'device': 'gpu'},
    resampler_config={'device': 'gpu'}
)

# For FP16:
self.model: Any = onnx_asr.load_model(
    "nemo-parakeet-tdt-0.6b-v3",
    model_path,
    quantization='fp16',  # Add this parameter
    providers=providers,
    sess_options=sess_options,
    preprocessor_config={'device': 'gpu'},
    resampler_config={'device': 'gpu'}
)
```

### Testing FP16 Models

Test that FP16 models work:

```bash
source venv/Scripts/activate
python -c "
import onnx_asr
import numpy as np

print('Loading FP16 models...')
model = onnx_asr.load_model(
    'nemo-parakeet-tdt-0.6b-v3',
    './models/parakeet',
    quantization='fp16',
    cpu_preprocessing=False
)
print('SUCCESS!')

# Test inference
audio = np.zeros(16000, dtype=np.float32)
result = model.recognize(audio)
print(f'Inference works: {result}')
"
```

## Performance Considerations

### Benefits

- **50% smaller model size**: Faster loading, less disk space
- **Potential GPU speedup**: FP16 operations can be faster on modern GPUs
- **Lower memory usage**: Reduced VRAM requirements

### Trade-offs

- **Minimal accuracy loss**: FP16 precision is usually sufficient for STT tasks
- **Quantization warnings**: Small values may be truncated (see conversion output)
- **Hardware support**: FP16 acceleration requires compatible GPU

## Testing

Run the FP16 conversion tests:

```bash
python -m pytest tests/test_fp16_conversion.py -v
```

Tests verify:
- Model conversion produces valid ONNX models
- FP16 models are significantly smaller than FP32
- Input/output types are preserved
- Converted models can be loaded by onnxruntime

## Implementation Details

### Key Technical Points

1. **`disable_shape_infer=True`** is critical for models with external data
   - Without this, the conversion loses all weight tensors
   - Shape inference can be problematic with large models

2. **`keep_io_types=True`** ensures compatibility
   - Input/output tensors remain FP32
   - Only internal weights are converted to FP16
   - Preprocessing code doesn't need changes

3. **External data handling**
   - Original encoder model uses external .data file (2.3GB)
   - FP16 version embeds all weights in single file (1.2GB)

### Conversion Parameters

```python
float16.convert_float_to_float16(
    model,
    min_positive_val=1e-07,      # Minimum positive value (default)
    max_finite_val=10000.0,       # Maximum finite value (default)
    keep_io_types=True,           # Keep I/O as FP32
    disable_shape_infer=True,     # Don't run shape inference
    op_block_list=None,           # No blocked operations
    node_block_list=None          # No blocked nodes
)
```

## Future Improvements

Potential enhancements:

1. **Config-based model selection**: Add FP16 option to `stt_config.json`
2. **Automatic fallback**: Try FP16 first, fall back to FP32 if not found
3. **Accuracy benchmarking**: Compare STT quality between FP32/FP16
4. **INT8 quantization**: Even smaller models with post-training quantization

## Quick Reference

### Complete FP16 Conversion Workflow

```bash
# 1. Install dependencies (if not already installed)
pip install onnx onnxconverter-common

# 2. Convert FP32 models to FP16
python convert_to_fp16.py

# 3. Fix Cast operations
python fix_fp16_casts.py

# 4. Test FP16 models
python -c "import onnx_asr; m = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', './models/parakeet', quantization='fp16'); print('Works!')"
```

### Common Issues

**Issue:** `Type parameter (T) of Optype (Add) bound to different types`
**Solution:** Run `python fix_fp16_casts.py` to fix internal Cast operations

**Issue:** `Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(float16))`
**Solution:** Run `python fix_fp16_casts.py` to add input Cast operations

**Issue:** Model loads but inference fails
**Solution:** Ensure you're using `quantization='fp16'` parameter in `onnx_asr.load_model()`

## References

- [ONNX Runtime FP16 Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- [onnxconverter-common on PyPI](https://pypi.org/project/onnxconverter-common/)
- [Parakeet Model on HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b)
