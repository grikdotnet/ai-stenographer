"""Execution Provider Manager for ONNX Runtime - DirectML/CPU selection.

This module manages execution provider selection with automatic fallback chain:
DirectML (GPU) → CPU (baseline)

Design Philosophy:
- Automatic provider selection based on hardware availability
- Graceful fallback to ensure application always runs
- Simple configuration interface (auto/directml/cpu)
- Smart GPU selection (prefers discrete over integrated GPUs)
"""

from typing import List, Dict, Union, Tuple, Optional
import onnxruntime as ort
import subprocess
import sys
import logging


class ExecutionProviderManager:
    """Manages execution provider selection with DirectML/CPU fallback.

    Responsibilities:
    - Detect available ONNX Runtime providers (DirectML, CPU)
    - Select optimal provider based on config and hardware
    - Build provider list for onnxruntime.InferenceSession()
    - Smart GPU detection (prefers discrete over integrated)

    Fallback chain: DirectML → CPU
    """

    _device_id_cache: Optional[int] = None
    _gpu_type_cache: Optional[str] = None

    def __init__(self, config: Dict):
        """Initialize with execution config.

        Args:
            config: Config dict with recognition.inference field:
                {
                    "recognition": {
                        "inference": "auto"|"directml"|"cpu"
                    }
                }
        """
        self.config = config
        self.inference_mode = config.get('recognition', {}).get('inference', 'auto')
        self.logger = logging.getLogger(__name__)

        self.selected_provider = self.select_provider()

        self.logger.info(f"Execution provider selected: {self.selected_provider}")

    def select_provider(self) -> str:
        """Select provider based on config and hardware.

        Algorithm:
        1. If inference mode is 'cpu', force CPU
        2. If inference mode is 'directml', force DirectML (fallback to CPU if unavailable)
        3. If inference mode is 'auto':
           a. Try DirectML if available
           b. Fallback to CPU if GPU providers unavailable

        Returns:
            Selected provider name ('DirectML' or 'CPU')
        """
        if self.inference_mode == 'cpu':
            return 'CPU'

        available_providers = ort.get_available_providers()
        self.logger.debug(f"Available ONNX Runtime providers: {available_providers}")

        if self.inference_mode == 'directml':
            if 'DmlExecutionProvider' in available_providers:
                return 'DirectML'
            else:
                self.logger.warning("DirectML requested but not available, falling back to CPU")
                return 'CPU'

        if self.inference_mode == 'auto':
            if 'DmlExecutionProvider' in available_providers:
                return 'DirectML'

            self.logger.info("No GPU providers available, using CPU")
            return 'CPU'

        self.logger.warning(f"Unknown inference mode: {self.inference_mode}, falling back to CPU")
        return 'CPU'

    def build_provider_list(self) -> Dict[str, Dict]:
        """Build ONNX Runtime provider configuration dict.

        Constructs provider configuration for onnxruntime.InferenceSession(providers=...)
        Returns a dict mapping provider names to their configuration options.

        Provider-specific configuration:
        - DirectML: Uses dynamically selected device_id (from select_device_id())
        - CPU: No options needed (empty dict)

        Returns:
            Dict mapping provider names to config dicts
            Examples:
            - CPU only: {'CPUExecutionProvider': {}}
            - DirectML: {'DmlExecutionProvider': {'device_id': 1}, 'CPUExecutionProvider': {}}
        """
        if self.selected_provider == 'CPU':
            return {'CPUExecutionProvider': {}}

        elif self.selected_provider == 'DirectML':
            device_id, _ = self.select_device_id()
            return {
                'DmlExecutionProvider': {'device_id': device_id},
                'CPUExecutionProvider': {}
            }

        else:
            self.logger.warning(f"Unknown provider {self.selected_provider}, falling back to CPU")
            return {'CPUExecutionProvider': {}}

    def detect_gpu_type(self) -> str:
        """Detect GPU type: 'integrated', 'discrete', or 'cpu'.

        Platform-specific detection:
        - Windows: wmic command (queries Win32_VideoController)

        GPU classification heuristics:
          NVIDIA (easy - all discrete):
          - GeForce, RTX, GTX, Quadro, Tesla → discrete

          Intel (easy - all integrated):
          - Iris, UHD Graphics, HD Graphics → integrated

          AMD (complex - "Radeon" used for both):
          - "Radeon RX 6700", "Radeon Pro" → discrete
          - "Radeon Graphics", "Radeon Vega 8" → integrated
          - Plain "Radeon" with no model → assume discrete (conservative)

        Multi-GPU Priority:
          If both integrated and discrete GPUs are present, prefer discrete for better performance.

        Returns:
            'integrated', 'discrete', or 'cpu'

        Example Multi-GPU System (Windows):
            Name
            Intel(R) UHD Graphics 630
            NVIDIA GeForce RTX 3060
            → Returns 'discrete' (prioritize NVIDIA)

        Example AMD Naming Ambiguity:
            "AMD Radeon(TM) Graphics" → integrated (has "Graphics")
            "AMD Radeon RX 6700 XT" → discrete (has "RX")
            "AMD Radeon Pro W5700" → discrete (has "Pro")
        """
        if self.selected_provider == 'CPU':
            return 'cpu'

        if sys.platform != 'win32':
            self.logger.warning(f"GPU detection only supported on Windows, defaulting to discrete")
            return 'discrete'

        try:
            return self._detect_gpu_type_windows()
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}, defaulting to discrete")
            return 'discrete'

    def _detect_gpu_type_windows(self) -> str:
        """Detect GPU type on Windows using wmic command.

        Uses Windows Management Instrumentation Command-line (wmic) to query
        Win32_VideoController for all installed GPU adapters.

        Returns:
            'integrated' or 'discrete'
        """
        result = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            capture_output=True, text=True, timeout=2
        )
        gpu_output = result.stdout.lower()

        has_discrete = False
        has_integrated = False

        # Check for integrated GPU indicators FIRST (AMD naming overlap)
        # AMD uses "Radeon(TM) Graphics" pattern for integrated GPUs
        integrated_keywords = [
            'intel', 'iris', 'uhd graphics', 'hd graphics',  # Intel
            'vega',  # AMD APU (Vega is always integrated)
        ]
        if any(keyword in gpu_output for keyword in integrated_keywords):
            has_integrated = True

        # Special case: AMD integrated uses "radeon" + "graphics" (with possible trademark in between)
        if 'radeon' in gpu_output and 'graphics' in gpu_output:
            # Check it's not a discrete model (RX, Pro, etc)
            if not any(disc in gpu_output for disc in ['radeon rx', 'radeon pro', 'radeon vii', 'radeon r9', 'radeon r7']):
                has_integrated = True

        discrete_keywords = [
            'nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla',  # NVIDIA
            'radeon rx', 'radeon pro', 'radeon vii', 'radeon r9', 'radeon r7'  # AMD discrete
        ]
        if any(keyword in gpu_output for keyword in discrete_keywords):
            has_discrete = True

        # Edge case: Plain "radeon" without specific model (only if not already classified)
        # Conservative: assume discrete (better to use dGPU settings on iGPU than vice versa)
        if 'radeon' in gpu_output and not has_discrete and not has_integrated:
            has_discrete = True

        if has_discrete:
            if has_integrated:
                self.logger.info("Multi-GPU system detected: Prioritizing discrete GPU")
            return 'discrete'
        elif has_integrated:
            return 'integrated'
        else:
            # Default to discrete if GPU provider is available but detection unclear
            return 'discrete'

    def enumerate_adapters(self) -> List[Dict[str, Union[int, str]]]:
        """Enumerate all GPU adapters with DeviceID and type classification.

        Queries Windows for all video controllers and classifies each as
        'integrated' or 'discrete' based on device name.

        Returns:
            List of dicts with 'index', 'name', 'type' keys:
            [
                {'index': 0, 'name': 'Intel UHD Graphics 630', 'type': 'integrated'},
                {'index': 1, 'name': 'NVIDIA GeForce RTX 3060', 'type': 'discrete'}
            ]

        Empty list for CPU-only mode.
        """
        if self.selected_provider == 'CPU':
            return []

        try:
            if sys.platform != 'win32':
                return []

            import subprocess

            # Query both DeviceID and Name columns
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'DeviceID,Name'],
                capture_output=True, text=True, timeout=2
            )

            adapters = []
            lines = result.stdout.strip().split('\n')

            # parse GPU entries
            for line in lines[1:]:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue

                try:
                    device_id = int(parts[0])
                    name = parts[1].strip()

                    if 'basic display adapter' in name.lower():
                        continue

                    gpu_type = self._classify_gpu_name(name)
                    adapters.append({
                        'index': device_id,
                        'name': name,
                        'type': gpu_type
                    })
                except (ValueError, IndexError):
                    continue

            return adapters

        except Exception as e:
            self.logger.warning(f"Adapter enumeration failed: {e}")
            return []

    def select_device_id(self) -> Tuple[int, str]:
        """Select optimal GPU adapter (device_id and gpu_type).

        Algorithm:
        1. CPU mode → return (-1, 'cpu')
        2. Enumerate all adapters with types
        3. If discrete GPU exists → use first discrete GPU adapter
        4. Else if integrated GPU exists → use first integrated GPU adapter
        5. Fallback → return (0, 'discrete')

        Returns:
            Tuple of (device_id: int, gpu_type: str)
            device_id: Adapter index for DirectML (0, 1, 2, ...) or -1 for CPU
            gpu_type: 'cpu', 'integrated', or 'discrete'

        Example Multi-GPU System:
            Adapters:
              0: Intel UHD Graphics 630 (integrated)
              1: NVIDIA GeForce RTX 3060 (discrete)
            → Returns (1, 'discrete') - use adapter 1 with discrete strategy
        """
        if self._device_id_cache is not None and self._gpu_type_cache is not None:
            return (self._device_id_cache, self._gpu_type_cache)

        if self.selected_provider == 'CPU':
            result = (-1, 'cpu')
            self._device_id_cache, self._gpu_type_cache = result
            return result

        try:
            adapters = self.enumerate_adapters()

            if not adapters:
                # Fallback: unknown hardware, assume discrete at adapter 0
                result = (0, 'discrete')
                self._device_id_cache, self._gpu_type_cache = result
                return result

            # Prefer discrete GPU if multiple GPUs present
            discrete_adapters = [a for a in adapters if a['type'] == 'discrete']
            if discrete_adapters:
                selected = discrete_adapters[0]
                if len(discrete_adapters) > 1 or any(a['type'] == 'integrated' for a in adapters):
                    self.logger.info(f"Multi-GPU detected: Selecting discrete GPU at adapter {selected['index']}")
                result = (selected['index'], 'discrete')
                self._device_id_cache, self._gpu_type_cache = result
                return result

            # Only integrated GPUs available
            integrated_adapters = [a for a in adapters if a['type'] == 'integrated']
            if integrated_adapters:
                selected = integrated_adapters[0]
                result = (selected['index'], 'integrated')
                self._device_id_cache, self._gpu_type_cache = result
                return result

            # unknown configuration
            result = (0, 'discrete')
            self._device_id_cache, self._gpu_type_cache = result
            return result

        except Exception as e:
            self.logger.warning(f"Device selection failed: {e}, defaulting to adapter 0")
            result = (0, 'discrete')
            self._device_id_cache, self._gpu_type_cache = result
            return result

    def _classify_gpu_name(self, name: str) -> str:
        """Classify GPU type from device name string.

        Uses same logic as _detect_gpu_type_windows() but for single GPU.

        Args:
            name: GPU device name from wmic

        Returns:
            'integrated' or 'discrete'
        """
        name_lower = name.lower()

        integrated_keywords = [
            'intel', 'iris', 'uhd graphics', 'hd graphics',  # Intel
            'vega',  # AMD APU
        ]
        if any(keyword in name_lower for keyword in integrated_keywords):
            return 'integrated'

        if 'radeon' in name_lower and 'graphics' in name_lower:
            discrete_models = ['radeon rx', 'radeon pro', 'radeon vii', 'radeon r9', 'radeon r7']
            if not any(model in name_lower for model in discrete_models):
                return 'integrated'

        discrete_keywords = [
            'nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla',  # NVIDIA
            'radeon rx', 'radeon pro', 'radeon vii', 'radeon r9', 'radeon r7'  # AMD discrete
        ]
        if any(keyword in name_lower for keyword in discrete_keywords):
            return 'discrete'

        # Edge case: plain "radeon" → assume discrete (conservative)
        if 'radeon' in name_lower:
            return 'discrete'

        # Default: unknown GPU → assume discrete
        return 'discrete'
