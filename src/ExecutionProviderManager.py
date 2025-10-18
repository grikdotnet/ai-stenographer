"""Execution Provider Manager for ONNX Runtime - Windows ML/DirectML/CPU selection.

This module manages execution provider selection with automatic fallback chain:
WindowsML (Win11 24H2+) → DirectML (GPU) → CPU (baseline)

Design Philosophy:
- Automatic provider selection based on OS version and hardware availability
- Graceful fallback to ensure application always runs
- Simple configuration interface (auto/directml/cpu)
- device_id hardcoded as class property (always primary GPU adapter)
"""

from typing import List, Dict, Union, Tuple, Optional
import onnxruntime as ort
import platform
import sys
import logging


class ExecutionProviderManager:
    """Manages execution provider selection with Windows ML/DirectML/CPU fallback.

    Responsibilities:
    - Detect Windows version (for WindowsML eligibility)
    - Detect available ONNX Runtime providers (WindowsML, DirectML, CPU)
    - Select optimal provider based on config and OS version
    - Build provider list for onnxruntime.InferenceSession()

    Fallback chain: WindowsML (Win11 24H2+) → DirectML → CPU
    """

    DEFAULT_DEVICE_ID = 0  # Primary GPU adapter (not configurable)

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

        # Detect environment
        self.windows_version = self.detect_windows_version()
        self.available_providers = self.detect_available_providers()

        # Select provider
        self.selected_provider = self.select_provider()

        self.logger.info(f"Execution provider selected: {self.selected_provider}")

    def detect_windows_version(self) -> Optional[Tuple[int, int]]:
        """Detect Windows version (major, build).

        Algorithm:
        - Parse platform.version() to extract build number
        - Determine Windows major version (10 vs 11) based on build
        - Windows 11 starts at build 22000

        Returns:
            (major, build) tuple or None for non-Windows
            Example: (11, 26100) for Windows 11 24H2
        """
        if sys.platform != 'win32':
            return None

        try:
            version = platform.version()  # e.g., "10.0.26100"
            parts = version.split('.')
            build = int(parts[2])

            # Windows 11 is build 22000+
            if build >= 22000:
                major = 11
            else:
                major = 10

            return (major, build)
        except (IndexError, ValueError) as e:
            self.logger.warning(f"Failed to detect Windows version: {e}")
            return None

    def detect_available_providers(self) -> List[str]:
        """Detect ONNX Runtime providers.

        Uses onnxruntime.get_available_providers() to query available execution providers.

        Returns:
            List of provider names (e.g., ['DmlExecutionProvider', 'CPUExecutionProvider'])
        """
        providers = ort.get_available_providers()
        self.logger.debug(f"Available ONNX Runtime providers: {providers}")
        return providers

    def select_provider(self) -> str:
        """Select provider based on config and OS version.

        Algorithm:
        1. If inference mode is 'cpu', force CPU
        2. If inference mode is 'directml', force DirectML (fallback to CPU if unavailable)
        3. If inference mode is 'auto':
           a. On Windows 11 24H2+ (build >= 26100): Try WindowsML
           b. On Windows 10 or older Windows 11: Try DirectML
           c. Fallback to CPU if GPU providers unavailable

        Returns:
            Selected provider name ('WindowsML', 'DirectML', or 'CPU')
        """
        # Force CPU mode
        if self.inference_mode == 'cpu':
            return 'CPU'

        # Force DirectML mode
        if self.inference_mode == 'directml':
            if 'DmlExecutionProvider' in self.available_providers:
                return 'DirectML'
            else:
                self.logger.warning("DirectML requested but not available, falling back to CPU")
                return 'CPU'

        # Auto mode: intelligent provider selection
        if self.inference_mode == 'auto':
            # Check for Windows ML on Win11 24H2+
            if self.windows_version and self.windows_version[0] == 11 and self.windows_version[1] >= 26100:
                if 'WindowsMLExecutionProvider' in self.available_providers:
                    return 'WindowsML'

            # Try DirectML (Windows 10+)
            if 'DmlExecutionProvider' in self.available_providers:
                return 'DirectML'

            # Fallback to CPU
            self.logger.info("No GPU providers available, using CPU")
            return 'CPU'

        # Unknown mode, fallback to CPU
        self.logger.warning(f"Unknown inference mode: {self.inference_mode}, falling back to CPU")
        return 'CPU'

    def build_provider_list(self) -> List[Union[str, Tuple[str, Dict]]]:
        """Build ONNX Runtime provider list.

        Constructs provider configuration for onnxruntime.InferenceSession(providers=...)
        Each provider can be either:
        - String: Simple provider name (e.g., 'CPUExecutionProvider')
        - Tuple: (provider_name, options_dict) for providers with configuration

        Provider-specific configuration:
        - DirectML: Uses device_id=0 (primary GPU adapter)
        - WindowsML: No options needed (auto EP selection)
        - CPU: No options needed

        Returns:
            Provider list for onnxruntime.InferenceSession(providers=...)
            Examples:
            - CPU: ['CPUExecutionProvider']
            - DirectML: [('DmlExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
            - WindowsML: ['WindowsMLExecutionProvider', 'CPUExecutionProvider']
        """
        if self.selected_provider == 'CPU':
            return ['CPUExecutionProvider']

        elif self.selected_provider == 'DirectML':
            return [
                ('DmlExecutionProvider', {'device_id': self.DEFAULT_DEVICE_ID}),
                'CPUExecutionProvider'
            ]

        elif self.selected_provider == 'WindowsML':
            return [
                'WindowsMLExecutionProvider',
                'CPUExecutionProvider'
            ]

        else:
            # Fallback to CPU
            self.logger.warning(f"Unknown provider {self.selected_provider}, falling back to CPU")
            return ['CPUExecutionProvider']

    def get_device_info(self) -> Dict:
        """Get device info for GUI display.

        Constructs user-friendly device information showing:
        - selected: The selected provider name
        - available: List of available provider names (human-readable)
        - provider: Technical provider name

        Returns:
            Dict with 'selected', 'available', 'provider' keys
            Example: {
                'selected': 'DirectML',
                'available': ['DirectML', 'CPU'],
                'provider': 'DirectML'
            }
        """
        # Map provider names to human-readable format
        provider_map = {
            'DmlExecutionProvider': 'DirectML',
            'WindowsMLExecutionProvider': 'WindowsML',
            'CPUExecutionProvider': 'CPU'
        }

        # Convert available providers to human-readable names
        available_names = [
            provider_map.get(p, p) for p in self.available_providers
            if p in provider_map
        ]

        return {
            'selected': self.selected_provider,
            'available': available_names,
            'provider': self.selected_provider
        }
