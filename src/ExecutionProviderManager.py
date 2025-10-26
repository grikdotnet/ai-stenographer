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

    def build_provider_list(self) -> List[Union[str, Tuple[str, Dict]]]:
        """Build ONNX Runtime provider configuration list.

        Constructs provider configuration for onnxruntime.InferenceSession(providers=...)
        Returns a list of provider specifications (strings or tuples with options).

        Provider-specific configuration:
        - DirectML: Uses dynamically selected device_id (from select_device_id())
        - CPU: No options needed (just provider name string)

        Returns:
            List of provider specifications
            Examples:
            - CPU only: ['CPUExecutionProvider']
            - DirectML: [('DmlExecutionProvider', {'device_id': 1}), 'CPUExecutionProvider']
        """
        if self.selected_provider == 'CPU':
            return ['CPUExecutionProvider']

        elif self.selected_provider == 'DirectML':
            device_id, _ = self.select_device_id()
            return [
                ('DmlExecutionProvider', {'device_id': device_id}),
                'CPUExecutionProvider'
            ]

        else:
            self.logger.warning(f"Unknown provider {self.selected_provider}, falling back to CPU")
            return ['CPUExecutionProvider']

    def detect_gpu_type(self) -> str:
        """Detect GPU type: 'integrated', 'discrete', or 'cpu'.

        Uses DXGI-based adapter enumeration to determine GPU type.
        Derives type from enumerate_adapters() which uses native DXGI API.
        Filters out virtual GPUs (Microsoft Basic Render Driver, etc.).

        GPU classification (from _classify_gpu_name):
          NVIDIA: GeForce, RTX, GTX, Quadro, Tesla → discrete
          Intel: Iris, UHD Graphics, HD Graphics → integrated
          AMD: "Radeon RX/Pro" → discrete, "Radeon Graphics" → integrated
          Microsoft Basic Render Driver → virtual (filtered out)

        Multi-GPU Priority:
          If both integrated and discrete GPUs are present, prefer discrete for better performance.

        Returns:
            'integrated', 'discrete', or 'cpu'

        Example Multi-GPU System:
            [Intel UHD Graphics 630, NVIDIA GeForce RTX 3060, Microsoft Basic Render Driver]
            → Returns 'discrete' (prioritize NVIDIA, ignore virtual)

        Fallback:
            If enumeration fails or returns empty list → 'discrete' (conservative)
        """
        if self.selected_provider == 'CPU':
            return 'cpu'

        try:
            adapters = self.enumerate_adapters()

            if not adapters:
                # No adapters found - conservative fallback
                self.logger.warning("No GPU adapters found, defaulting to discrete")
                return 'discrete'

            # Filter out virtual GPUs (software renderers)
            real_adapters = [a for a in adapters if a['type'] != 'virtual']

            if not real_adapters:
                # Only virtual GPUs found (edge case)
                self.logger.warning(
                    "Only virtual GPUs detected (e.g., Microsoft Basic Render Driver). "
                    "Defaulting to discrete type."
                )
                return 'discrete'

            # Check for discrete GPUs first (preferred) - from real adapters only
            has_discrete = any(adapter['type'] == 'discrete' for adapter in real_adapters)
            has_integrated = any(adapter['type'] == 'integrated' for adapter in real_adapters)

            if has_discrete:
                if has_integrated:
                    self.logger.info("Multi-GPU system detected: Prioritizing discrete GPU")
                return 'discrete'
            elif has_integrated:
                return 'integrated'
            else:
                # Unknown GPU types - conservative fallback
                return 'discrete'

        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}, defaulting to discrete")
            return 'discrete'

    def enumerate_adapters(self) -> List[Dict[str, Union[int, str]]]:
        """Enumerate all GPU adapters using native DXGI API.

        Uses IDXGIFactory::EnumAdapters for guaranteed-correct DirectML device_id ordering.
        No heuristics - DXGI enumeration order IS the ground truth.

        DXGI Enumeration Order:
        - On laptops: Integrated GPU (primary display) is typically device_id=0
        - Discrete GPU is typically device_id=1
        - On desktops: Primary display adapter is device_id=0
        - DXGI order matches DirectML device_id exactly

        Returns:
            List of dicts with 'index', 'name', 'type' keys in DXGI order:
            [
                {'index': 0, 'name': 'Intel Iris Graphics', 'type': 'integrated'},
                {'index': 1, 'name': 'NVIDIA GeForce RTX', 'type': 'discrete'}
            ]

            Empty list for CPU-only mode or enumeration failures.
        """
        if self.selected_provider == 'CPU':
            return []

        if sys.platform != 'win32':
            return []

        try:
            # Use native DXGI enumeration for ground truth device_id mapping
            adapters = self._enumerate_adapters_dxgi()
            return adapters

        except Exception as e:
            # DXGI enumeration failed - log and return empty list
            # Caller (select_device_id) will handle empty list gracefully
            self.logger.warning(f"DXGI adapter enumeration failed: {e}, using fallback")
            return []

    def select_device_id(self) -> Tuple[int, str]:
        """Select optimal GPU adapter (device_id and gpu_type).

        Algorithm:
        1. CPU mode → return (-1, 'cpu')
        2. Enumerate all adapters with types
        3. Filter out virtual GPUs (Microsoft Basic Render Driver, etc.)
        4. If discrete GPU exists → use first discrete GPU adapter
        5. Else if integrated GPU exists → use first integrated GPU adapter
        6. Fallback → return (0, 'discrete')

        Returns:
            Tuple of (device_id: int, gpu_type: str)
            device_id: Adapter index for DirectML (0, 1, 2, ...) or -1 for CPU
            gpu_type: 'cpu', 'integrated', or 'discrete'

        Example Multi-GPU System:
            Adapters:
              0: Intel UHD Graphics 630 (integrated)
              1: NVIDIA GeForce RTX 3060 (discrete)
              2: Microsoft Basic Render Driver (virtual) ← filtered out
            → Returns (1, 'discrete') - use adapter 1, skip virtual@2

        Virtual GPU Filtering:
            Virtual GPUs (software renderers) are excluded from selection:
            - Microsoft Basic Render Driver (0 VRAM)
            - Microsoft Basic Display Adapter
            - Remote Desktop adapters
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

            # Filter out virtual GPUs (software renderers)
            # Keep DXGI indices intact - just skip virtual GPUs during selection
            real_adapters = [a for a in adapters if a['type'] != 'virtual']

            if not real_adapters:
                # Only virtual GPUs found (edge case)
                # Fall back to device_id=0 with discrete type (conservative)
                self.logger.warning(
                    "Only virtual GPUs detected (e.g., Microsoft Basic Render Driver). "
                    "Using fallback device_id=0. Consider using CPU mode for better performance."
                )
                result = (0, 'discrete')
                self._device_id_cache, self._gpu_type_cache = result
                return result

            # Log if virtual GPUs were filtered
            virtual_count = len(adapters) - len(real_adapters)
            if virtual_count > 0:
                virtual_names = [a['name'] for a in adapters if a['type'] == 'virtual']
                self.logger.debug(f"Filtered out {virtual_count} virtual GPU(s): {virtual_names}")

            # Prefer discrete GPU if available (from real adapters only)
            discrete_adapters = [a for a in real_adapters if a['type'] == 'discrete']
            if discrete_adapters:
                selected = discrete_adapters[0]
                if len(discrete_adapters) > 1 or any(a['type'] == 'integrated' for a in real_adapters):
                    self.logger.info(f"Multi-GPU detected: Selecting discrete GPU at adapter {selected['index']}")
                result = (selected['index'], 'discrete')
                self._device_id_cache, self._gpu_type_cache = result
                return result

            # Only integrated GPUs available
            integrated_adapters = [a for a in real_adapters if a['type'] == 'integrated']
            if integrated_adapters:
                selected = integrated_adapters[0]
                result = (selected['index'], 'integrated')
                self._device_id_cache, self._gpu_type_cache = result
                return result

            # Unknown configuration (should not reach here)
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

        Classifies GPUs into: 'integrated', 'discrete', or 'virtual'.

        Virtual GPUs:
        - Microsoft Basic Render Driver (software renderer)
        - Microsoft Basic Display Adapter (fallback driver)
        - Remote Desktop adapters
        - Any virtual/software rendering adapter

        Args:
            name: GPU device name from DXGI

        Returns:
            'integrated', 'discrete', or 'virtual'
        """
        name_lower = name.lower()

        # Check for virtual/software renderers FIRST (highest priority)
        virtual_keywords = [
            'microsoft basic render',     # Microsoft Basic Render Driver
            'microsoft basic display',    # Microsoft Basic Display Adapter
            'microsoft remote display',   # Remote Desktop
            'virtual',                    # Generic virtual adapters
        ]
        if any(keyword in name_lower for keyword in virtual_keywords):
            return 'virtual'

        # Integrated GPUs
        integrated_keywords = [
            'intel', 'iris', 'uhd graphics', 'hd graphics',  # Intel
            'vega',  # AMD APU
        ]
        if any(keyword in name_lower for keyword in integrated_keywords):
            return 'integrated'

        # AMD special case: "Radeon Graphics" = integrated, "Radeon RX" = discrete
        if 'radeon' in name_lower and 'graphics' in name_lower:
            discrete_models = ['radeon rx', 'radeon pro', 'radeon vii', 'radeon r9', 'radeon r7']
            if not any(model in name_lower for model in discrete_models):
                return 'integrated'

        # Discrete GPUs
        discrete_keywords = [
            'nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla',  # NVIDIA
            'radeon rx', 'radeon pro', 'radeon vii', 'radeon r9', 'radeon r7'  # AMD discrete
        ]
        if any(keyword in name_lower for keyword in discrete_keywords):
            return 'discrete'

        # Edge case: plain "radeon" → assume discrete (conservative)
        if 'radeon' in name_lower:
            return 'discrete'

        # Default: unknown GPU → assume discrete (conservative)
        return 'discrete'

    def _enumerate_adapters_dxgi(self) -> List[Dict[str, Union[int, str, float]]]:
        """Enumerate GPU adapters using native DXGI API (Windows only).

        Uses IDXGIFactory::EnumAdapters to get adapters in EXACT DirectML device_id order.
        This is the ground truth for device_id mapping - no heuristics needed.

        Algorithm:
        1. Load dxgi.dll and call CreateDXGIFactory()
        2. Call IDXGIFactory::EnumAdapters() in loop (index 0, 1, 2...)
        3. For each adapter, call IDXGIAdapter::GetDesc() to get name/VRAM
        4. Classify GPU type using _classify_gpu_name()
        5. Return list in exact DXGI enumeration order

        Returns:
            List of dicts with 'index', 'name', 'type', 'vram_gb' keys in DXGI order:
            [
                {'index': 0, 'name': 'Intel UHD Graphics', 'type': 'integrated', 'vram_gb': 1.0},
                {'index': 1, 'name': 'NVIDIA GeForce RTX', 'type': 'discrete', 'vram_gb': 4.0}
            ]

            Empty list on non-Windows or errors.

        Raises:
            Exception: If DXGI enumeration fails (caller should handle)
        """
        if sys.platform != 'win32':
            return []

        try:
            import ctypes
            from ctypes import wintypes, POINTER, Structure, c_void_p, c_uint

            # DXGI_ERROR_NOT_FOUND = 0x887A0002
            # Note: HRESULT is signed (wintypes.LONG), but error codes are often unsigned.
            # When returned from COM, 0x887A0002 becomes -2005401598 (signed interpretation).
            # We need to handle both forms for comparison.
            DXGI_ERROR_NOT_FOUND_UNSIGNED = 0x887A0002
            DXGI_ERROR_NOT_FOUND_SIGNED = ctypes.c_long(DXGI_ERROR_NOT_FOUND_UNSIGNED).value

            class GUID(Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            class DXGI_ADAPTER_DESC(Structure):
                _fields_ = [
                    ("Description", wintypes.WCHAR * 128),
                    ("VendorId", c_uint),
                    ("DeviceId", c_uint),
                    ("SubSysId", c_uint),
                    ("Revision", c_uint),
                    ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t),
                    ("SharedSystemMemory", ctypes.c_size_t),
                    ("AdapterLuid", wintypes.LARGE_INTEGER),
                ]

            class IDXGIObject(Structure):
                pass

            class IDXGIAdapter(Structure):
                pass

            class IDXGIFactory(Structure):
                pass

            # Function pointer types for vtable methods
            GetDesc_func = ctypes.WINFUNCTYPE(
                wintypes.LONG,
                POINTER(IDXGIAdapter),
                POINTER(DXGI_ADAPTER_DESC)
            )

            EnumAdapters_func = ctypes.WINFUNCTYPE(
                wintypes.LONG,
                POINTER(IDXGIFactory),
                c_uint,  # Adapter index
                POINTER(POINTER(IDXGIAdapter))
            )

            Release_func = ctypes.WINFUNCTYPE(
                c_uint,  # ULONG (refcount)
                c_void_p
            )

            dxgi = ctypes.WinDLL('dxgi.dll')

            CreateDXGIFactory = dxgi.CreateDXGIFactory
            CreateDXGIFactory.argtypes = [POINTER(GUID), POINTER(c_void_p)]
            CreateDXGIFactory.restype = wintypes.LONG

            IID_IDXGIFactory = GUID()
            IID_IDXGIFactory.Data1 = 0x7b7166ec
            IID_IDXGIFactory.Data2 = 0x21c7
            IID_IDXGIFactory.Data3 = 0x44ae

            for i, val in enumerate([0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69]):
                IID_IDXGIFactory.Data4[i] = val

            factory_ptr = c_void_p()
            hr = CreateDXGIFactory(ctypes.byref(IID_IDXGIFactory), ctypes.byref(factory_ptr))

            if hr < 0:
                raise Exception(f"CreateDXGIFactory failed with HRESULT 0x{hr:08X}")

            factory_vtable_ptr = ctypes.cast(factory_ptr, POINTER(c_void_p)).contents

            enum_adapters_ptr = ctypes.cast(
                factory_vtable_ptr.value + 7 * ctypes.sizeof(c_void_p),
                POINTER(c_void_p)
            ).contents
            EnumAdapters = EnumAdapters_func(enum_adapters_ptr.value)

            release_ptr = ctypes.cast(
                factory_vtable_ptr.value + 2 * ctypes.sizeof(c_void_p),
                POINTER(c_void_p)
            ).contents
            Release = Release_func(release_ptr.value)

            adapters = []
            adapter_index = 0

            # Enumerate adapters
            while True:
                adapter_ptr = POINTER(IDXGIAdapter)()
                factory_as_iface = ctypes.cast(factory_ptr, POINTER(IDXGIFactory))
                hr = EnumAdapters(factory_as_iface, adapter_index, ctypes.byref(adapter_ptr))

                # Check for DXGI_ERROR_NOT_FOUND (end of enumeration)
                # HRESULT is signed, so check both unsigned and signed forms
                if hr == DXGI_ERROR_NOT_FOUND_UNSIGNED or hr == DXGI_ERROR_NOT_FOUND_SIGNED:
                    # No more adapters - normal exit
                    break

                if hr < 0:
                    # Other error - unexpected
                    self.logger.warning(f"EnumAdapters({adapter_index}) failed with HRESULT 0x{hr:08X}")
                    break

                desc = DXGI_ADAPTER_DESC()

                # IDXGIAdapter vtable - GetDesc is at index 8
                adapter_vtable_ptr = ctypes.cast(adapter_ptr, POINTER(c_void_p)).contents
                get_desc_ptr = ctypes.cast(
                    adapter_vtable_ptr.value + 8 * ctypes.sizeof(c_void_p),
                    POINTER(c_void_p)
                ).contents
                GetDesc = GetDesc_func(get_desc_ptr.value)

                hr = GetDesc(adapter_ptr, ctypes.byref(desc))

                if hr >= 0:
                    adapter_name = desc.Description
                    vram_bytes = desc.DedicatedVideoMemory
                    vram_gb = vram_bytes / (1024 ** 3)

                    gpu_type = self._classify_gpu_name(adapter_name)

                    adapters.append({
                        'index': adapter_index,
                        'name': adapter_name,
                        'type': gpu_type,
                        'vram_gb': round(vram_gb, 2)
                    })

                    self.logger.debug(
                        f"DXGI device_id {adapter_index}: {adapter_name} "
                        f"({gpu_type}, {vram_gb:.2f} GB VRAM)"
                    )

                # Release adapter
                adapter_release_ptr = ctypes.cast(
                    adapter_vtable_ptr.value + 2 * ctypes.sizeof(c_void_p),
                    POINTER(c_void_p)
                ).contents
                AdapterRelease = Release_func(adapter_release_ptr.value)
                AdapterRelease(adapter_ptr)

                adapter_index += 1

            # Release factory
            Release(factory_ptr)

            self.logger.info(f"DXGI enumeration found {len(adapters)} adapter(s)")
            return adapters

        except Exception as e:
            self.logger.warning(f"DXGI enumeration failed: {e}")
            raise  # Re-raise for caller to handle
