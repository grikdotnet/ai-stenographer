"""Tests for the MSIX distribution builder."""

from pathlib import Path


def _write_msix_inputs(root: Path) -> tuple[Path, Path, Path]:
    msix_dir = root / "msix"
    assets_dir = msix_dir / "Assets"
    assets_dir.mkdir(parents=True)
    (msix_dir / "AppxManifest.xml").write_text(
        '<Application Executable="AIStenographer.exe" />',
        encoding="utf-8",
    )
    for asset in ["Square44x44Logo.png", "Square150x150Logo.png", "StoreLogo.png"]:
        (assets_dir / asset).write_bytes(b"png")

    project_root = root / "project"
    project_root.mkdir()
    (project_root / "PRIVACY_POLICY.txt").write_text("privacy", encoding="utf-8")

    staging_dir = root / "staging"
    staging_dir.mkdir()
    return staging_dir, msix_dir, project_root


def _write_valid_msix_structure(staging_dir: Path, runtime_exe: str = "python.exe") -> None:
    (staging_dir / "Assets").mkdir(parents=True)
    for asset in ["Square44x44Logo.png", "Square150x150Logo.png", "StoreLogo.png"]:
        (staging_dir / "Assets" / asset).write_bytes(b"png")

    (staging_dir / "AppxManifest.xml").write_text(
        '<Application Executable="AIStenographer.exe" />',
        encoding="utf-8",
    )
    (staging_dir / "AIStenographer.exe").write_bytes(b"tauri")
    (staging_dir / "_internal" / "runtime").mkdir(parents=True)
    (staging_dir / "_internal" / "runtime" / runtime_exe).write_bytes(b"python")
    (staging_dir / "_internal" / "app").mkdir(parents=True)
    (staging_dir / "_internal" / "app" / "main.pyc").write_bytes(b"pyc")


def test_copy_msix_specific_files_does_not_require_or_copy_launcher(tmp_path):
    import msix.build_msix_distribution as msix_builder

    staging_dir, msix_dir, project_root = _write_msix_inputs(tmp_path)

    result = msix_builder.copy_msix_specific_files(staging_dir, msix_dir, project_root)

    assert result is True
    assert (staging_dir / "AppxManifest.xml").exists()
    assert (staging_dir / "Assets" / "Square44x44Logo.png").exists()
    assert (staging_dir / "PRIVACY_POLICY.txt").exists()
    assert not (staging_dir / "AIStenographer.exe").exists()


def test_validate_package_structure_requires_python_exe_not_pythonw(tmp_path):
    import msix.build_msix_distribution as msix_builder

    staging_dir = tmp_path / "staging"
    _write_valid_msix_structure(staging_dir, runtime_exe="pythonw.exe")

    assert msix_builder.validate_package_structure(staging_dir) is False


def test_validate_package_structure_accepts_tauri_supervisor_runtime_contract(tmp_path):
    import msix.build_msix_distribution as msix_builder

    staging_dir = tmp_path / "staging"
    _write_valid_msix_structure(staging_dir, runtime_exe="python.exe")

    assert msix_builder.validate_package_structure(staging_dir) is True


def test_msix_tauri_executable_copy_is_byte_identical_to_release_output(tmp_path):
    import msix.build_msix_distribution as msix_builder

    src_tauri_dir = tmp_path / "client" / "tauri" / "src-tauri"
    release_dir = src_tauri_dir / "target" / "release"
    release_dir.mkdir(parents=True)
    release_exe = release_dir / "stt-tauri-client.exe"
    release_exe.write_bytes(b"tauri release exe bytes")

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    assert msix_builder.copy_tauri_executable(
        src_tauri_dir,
        staging_dir,
        dst_filename="AIStenographer.exe",
    ) is True

    staged_exe = staging_dir / "AIStenographer.exe"
    assert staged_exe.read_bytes() == release_exe.read_bytes()


def test_msix_main_builds_tauri_before_python_packaging_and_makeappx(tmp_path, monkeypatch):
    import msix.build_msix_distribution as msix_builder

    calls: list[str] = []
    tauri_copy_kwargs: dict[str, str] = {}
    fake_script = tmp_path / "repo" / "msix" / "build_msix_distribution.py"
    monkeypatch.setattr(msix_builder, "__file__", str(fake_script))
    monkeypatch.setattr(msix_builder, "CREATE_SIGNED_COPY", False)
    monkeypatch.setattr(msix_builder, "SKIP_SIGNING", True)
    monkeypatch.setattr(msix_builder, "check_build_prerequisites", lambda: True)

    def record(name: str, return_value=True):
        def _stub(*args, **kwargs):
            calls.append(name)
            return return_value

        return _stub

    def create_dirs(staging_dir: Path):
        calls.append("dirs")
        return {
            "runtime": staging_dir / "_internal" / "runtime",
            "lib": staging_dir / "_internal" / "Lib" / "site-packages",
            "app": staging_dir / "_internal" / "app",
            "models": staging_dir / "_internal" / "models",
        }

    def create_msix_package(staging_dir: Path, output_path: Path, sdk_dir: Path) -> bool:
        calls.append("makeappx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"msix")
        return True

    def copy_tauri(*args, **kwargs) -> bool:
        calls.append("tauri_copy")
        tauri_copy_kwargs.update(kwargs)
        return True

    def find_sdk() -> Path:
        calls.append("sdk")
        return tmp_path / "sdk"

    monkeypatch.setattr(msix_builder, "find_windows_sdk", find_sdk)
    monkeypatch.setattr(msix_builder, "download_embedded_python", lambda v, c: tmp_path / "py.zip")
    monkeypatch.setattr(msix_builder, "create_directory_structure", create_dirs)

    for name, stub in {
        "build_tauri_frontend": record("tauri_frontend"),
        "build_tauri_release": record("tauri_release"),
        "extract_embedded_python": record("extract"),
        "copy_signed_executables_from_system": record("signed_python"),
        "verify_signatures": record("verify_signatures"),
        "create_pth_file": record("pth"),
        "enable_pip": record("pip"),
        "verify_pip": record("verify_pip"),
        "collect_third_party_licenses": record("licenses"),
        "install_dependencies": record("install"),
        "remove_pip_from_distribution": record("remove_pip"),
        "remove_tests_from_distribution": record("remove_tests"),
        "copy_application_code": record("copy_app"),
        "compile_to_pyc": record("compile_app"),
        "compile_site_packages": record("compile_site"),
        "zip_site_packages": record("zip_site"),
        "cleanup_package_metadata": record("cleanup"),
        "copy_bundled_silero_vad": record("silero"),
        "copy_assets_and_config": record("assets_config"),
        "copy_msix_specific_files": record("msix_files"),
        "copy_tauri_executable": copy_tauri,
        "copy_legal_documents_msix": record("legal"),
        "create_store_readme": record("readme"),
    }.items():
        monkeypatch.setattr(msix_builder, name, stub, raising=False)

    monkeypatch.setattr(msix_builder, "verify_native_libraries", lambda lib: {"dummy": True})
    monkeypatch.setattr(msix_builder, "test_imports", lambda py, mods: {m: True for m in mods})
    monkeypatch.setattr(msix_builder, "create_msix_package", create_msix_package)

    assert msix_builder.main() == 0

    order = {name: index for index, name in enumerate(calls)}
    assert order["sdk"] < order["tauri_frontend"]
    assert order["tauri_frontend"] < order["tauri_release"]
    assert order["tauri_release"] < order["install"]
    assert order["tauri_copy"] > order["msix_files"]
    assert order["tauri_copy"] < order["legal"]
    assert order["tauri_copy"] < order["makeappx"]
    assert tauri_copy_kwargs == {"dst_filename": "AIStenographer.exe"}


def test_manifest_executable_matches_staged_tauri_filename():
    manifest = Path("msix") / "AppxManifest.xml"

    assert 'Executable="AIStenographer.exe"' in manifest.read_text(encoding="utf-8")


def test_repository_has_no_obsolete_msix_launcher_directory():
    assert not (Path("msix") / "launcher").exists()
