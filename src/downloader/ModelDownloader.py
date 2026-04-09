"""
ModelDownloader provides shared logic for downloading and validating Parakeet STT models.

Responsibilities: CDN download, atomic manifest writing, partial-file cleanup.
No Tkinter dependencies — safe for server-side and client-side use.
"""
import json
import logging
import os
from pathlib import Path
from typing import Callable, List
from urllib.request import getproxies

import requests
import xxhash


PARAKEET_CDN_URL = "https://parakeet.grik.net/"
PARAKEET_FILES: List[str] = [
    "config.json",
    "decoder_joint-model.fp16.onnx",
    "encoder-model.fp16.onnx",
    "nemo128.onnx",
    "vocab.txt",
]

_MANIFEST_FILENAME = "manifest.json"
_MANIFEST_TMP_FILENAME = "manifest.json.tmp"


class ModelDownloader:
    """Download and validate shared STT model assets.

    Args:
        models_dir: Root models directory used for download and validation.
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    def download_parakeet(
        self,
        progress_callback: Callable[[float, int, int], None] | None = None,
    ) -> None:
        """Download the Parakeet model into this downloader's models directory.

        Args:
            progress_callback: Optional progress callback.
        """
        callback = progress_callback or (lambda _progress, _downloaded, _total: None)
        download_parakeet(self._models_dir, callback)

    def validate_parakeet(self) -> bool:
        """Return True when the Parakeet installation passes manifest validation.

        Returns:
            True if the local Parakeet installation is valid.
        """
        return validate_parakeet(self._models_dir)

    @staticmethod
    def cleanup_partial_files(models_dir: Path) -> None:
        """Remove partial download artefacts from models_dir.

        Args:
            models_dir: Root models directory to clean.
        """
        cleanup_partial_files(models_dir)


def _configure_network_environment() -> None:
    """
    Configure network settings for MSIX sandbox and corporate networks.

    Detects Windows proxy settings and sets HTTP_PROXY / HTTPS_PROXY environment
    variables so that requests (and underlying urllib3) can reach the CDN.
    """
    proxies = getproxies()
    if proxies:
        logging.info(f"System proxy detected: {proxies}")
        os.environ["HTTP_PROXY"] = proxies.get("http", "")
        os.environ["HTTPS_PROXY"] = proxies.get("https", "")
    else:
        logging.info("No system proxy detected")


def _write_manifest(parakeet_dir: Path) -> None:
    """
    Atomically write manifest.json for all Parakeet files in parakeet_dir.

    Algorithm:
        1. Compute XXH3-128 hash and byte-size for every file in PARAKEET_FILES.
        2. Serialise as JSON to manifest.json.tmp.
        3. Rename manifest.json.tmp → manifest.json.

    Args:
        parakeet_dir: Directory containing the downloaded Parakeet model files.
    """
    entries = []
    for name in PARAKEET_FILES:
        file_path = parakeet_dir / name
        data = file_path.read_bytes()
        entries.append({
            "name": name,
            "xxh3": xxhash.xxh3_128_hexdigest(data),
            "size": len(data),
        })

    manifest = {"model": "parakeet", "version": 1, "files": entries}
    tmp_path = parakeet_dir / _MANIFEST_TMP_FILENAME
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(parakeet_dir / _MANIFEST_FILENAME)


def cleanup_partial_files(models_dir: Path) -> None:
    """
    Remove partial download artefacts from all known model sub-directories.

    Removes *.partial files as well as manifest.json and manifest.json.tmp
    to ensure a clean slate for the next download attempt.

    Args:
        models_dir: Root models directory (e.g. ./models).
    """
    for subdir_name in ("parakeet", "silero_vad"):
        subdir = models_dir / subdir_name
        if not subdir.exists():
            continue
        for partial_file in subdir.glob("*.partial"):
            try:
                partial_file.unlink()
            except Exception:
                pass
        for manifest_name in (_MANIFEST_FILENAME, _MANIFEST_TMP_FILENAME):
            candidate = subdir / manifest_name
            if candidate.exists():
                try:
                    candidate.unlink()
                except Exception:
                    pass


def _get_remote_file_size(url: str) -> int:
    """Read the remote file size from HTTP headers without downloading the body.

    Args:
        url: File URL to inspect.

    Returns:
        Reported Content-Length in bytes, or 0 when absent.

    Raises:
        RuntimeError: If the request fails.
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        return int(response.headers.get("Content-Length", 0))
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"HTTP error inspecting {url}: "
            f"{exc.response.status_code} {exc.response.reason}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Download timeout while inspecting {url}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Connection error inspecting {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error inspecting {url}: {exc}") from exc
    finally:
        try:
            response.close()
        except Exception:
            pass


def _calculate_overall_progress(
    completed_downloaded: int,
    current_downloaded: int,
    overall_total: int,
) -> float:
    """Calculate byte-weighted overall progress for a multi-file download.

    Args:
        completed_downloaded: Bytes from fully downloaded files.
        current_downloaded: Bytes downloaded for the current file.
        overall_total: Total bytes across all files.

    Returns:
        Progress fraction in the range ``[0.0, 1.0]``.
    """
    if overall_total <= 0:
        return 0.0
    return min((completed_downloaded + current_downloaded) / overall_total, 1.0)


def _partial_download_path(target_path: Path) -> Path:
    """Return the temporary path used while a file is still downloading.

    Args:
        target_path: Final destination path for the downloaded file.

    Returns:
        Sibling path ending with ``.partial``.
    """
    return target_path.with_name(f"{target_path.name}.partial")


def download_parakeet(
    models_dir: Path,
    progress_callback: Callable[[float, int, int], None],
) -> None:
    """
    Download all Parakeet FP16 model files from the CDN and write manifest.json.

    Algorithm:
        1. Configure proxy environment.
        2. Create the parakeet sub-directory if absent.
        3. For each file in PARAKEET_FILES stream-download it with progress updates.
        4. After all files succeed, write manifest.json atomically.

    Args:
        models_dir: Root models directory. The parakeet/ sub-directory is created inside it.
        progress_callback: Called with ``(progress, downloaded_bytes, total_bytes)``.

    Raises:
        RuntimeError: If any file cannot be downloaded.
    """
    _configure_network_environment()

    parakeet_dir = models_dir / "parakeet"
    parakeet_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading FP16 Parakeet models from CDN (1.25GB)...")

    file_sizes = {
        filename: _get_remote_file_size(PARAKEET_CDN_URL + filename)
        for filename in PARAKEET_FILES
    }
    overall_total = sum(file_sizes.values())
    cumulative_downloaded = 0

    for filename in PARAKEET_FILES:
        url = PARAKEET_CDN_URL + filename
        target_path = parakeet_dir / filename
        partial_path = _partial_download_path(target_path)

        logging.info(f"Downloading {filename} from {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            file_size = file_sizes[filename]
            bytes_downloaded = 0

            if partial_path.exists():
                partial_path.unlink()

            with open(partial_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
                        bytes_downloaded += len(chunk)
                        progress_callback(
                            _calculate_overall_progress(
                                cumulative_downloaded,
                                bytes_downloaded,
                                overall_total,
                            ),
                            cumulative_downloaded + bytes_downloaded,
                            overall_total,
                        )

            if partial_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file {filename} is empty")

            partial_path.replace(target_path)
            cumulative_downloaded += bytes_downloaded
            logging.info(f"Downloaded {filename} ({bytes_downloaded} bytes)")

        except requests.exceptions.HTTPError as exc:
            if partial_path.exists():
                partial_path.unlink()
            raise RuntimeError(
                f"HTTP error downloading {filename}: "
                f"{exc.response.status_code} {exc.response.reason}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            if partial_path.exists():
                partial_path.unlink()
            raise RuntimeError(f"Download timeout for {filename} (network too slow)") from exc
        except requests.exceptions.ConnectionError as exc:
            if partial_path.exists():
                partial_path.unlink()
            raise RuntimeError(f"Connection error downloading {filename}: {exc}") from exc
        except Exception as exc:
            if partial_path.exists():
                partial_path.unlink()
            raise RuntimeError(f"Error downloading {filename}: {exc}") from exc

    _write_manifest(parakeet_dir)
    progress_callback(1.0, cumulative_downloaded, overall_total)
    logging.info("FP16 models downloaded and manifest written successfully.")


def validate_parakeet(models_dir: Path) -> bool:
    """
    Validate the Parakeet model installation using manifest.json.

    Algorithm:
        1. manifest.json must exist.
        2. Every file listed must exist on disk.
        3. Each file's byte-size must match the manifest size field.
        4. Each file's XXH3-128 hash must match the manifest xxh3 field.

    Args:
        models_dir: Root models directory.

    Returns:
        True if all files pass all checks, False otherwise.
    """
    manifest_path = models_dir / "parakeet" / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return False

    try:
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            file_path = models_dir / "parakeet" / entry["name"]
            if not file_path.exists():
                return False
            data = file_path.read_bytes()
            if len(data) != entry["size"]:
                return False
            if xxhash.xxh3_128_hexdigest(data) != entry["xxh3"]:
                return False
    except Exception:
        return False

    return True
