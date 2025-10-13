"""
LicenseCollector - Automated third-party license collection for distribution.

This module collects license files from installed Python packages and
organizes them for distribution compliance.

Usage:
    from src.LicenseCollector import LicenseCollector

    collector = LicenseCollector(output_dir="LICENSES")
    collector.collect_all_licenses()
    collector.create_python_license_entry()
    collector.generate_notices_file()
    collector.generate_readme()

Integrated into:
    - build_distribution.py (step 10)
    - rebuild_quick.py (optional with --refresh-licenses)
"""
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import subprocess


class LicenseCollector:
    """Collects and organizes license files from installed Python packages."""

    # Core runtime dependencies (from requirements.txt and CLAUDE.md)
    RUNTIME_PACKAGES = [
        'sounddevice',
        'numpy',
        'onnx-asr',
        'onnxruntime',
        'Pillow',
        'soundfile',
        'scipy',
        'torch',
        'torchaudio',
        'huggingface-hub',
        'requests',
        'PyYAML',
        'tqdm',
        'coloredlogs',
    ]

    # Test-only dependencies (not needed in distribution)
    TEST_PACKAGES = [
        'pytest',
        'pluggy',
        'iniconfig',
        'packaging',
    ]

    def __init__(self, output_dir: str = 'LICENSES'):
        """
        Initialize LicenseCollector.

        Args:
            output_dir: Directory to store collected licenses (default: 'LICENSES')
        """
        self.output_dir = Path(output_dir)
        self.site_packages = self._get_site_packages_path()
        self.collected_licenses: List[Dict] = []

    def _get_site_packages_path(self) -> Path:
        """Get the site-packages directory path."""
        # Get the actual site-packages used by pip
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'pip'],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                return Path(line.split(':', 1)[1].strip())

        # Fallback to sysconfig
        import sysconfig
        return Path(sysconfig.get_path('purelib'))

    def _get_package_metadata(self, package_name: str) -> Optional[Dict]:
        """Get package metadata using pip show."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', '--verbose', package_name],
                capture_output=True,
                text=True,
                check=True
            )

            metadata = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()

            return metadata
        except subprocess.CalledProcessError:
            return None

    def _find_license_file(self, package_name: str) -> Optional[Path]:
        """Find license file for a package in dist-info directory."""
        # Try different package name variations
        name_lower = package_name.lower()
        name_with_underscore = name_lower.replace('-', '_')
        name_with_hyphen = name_lower.replace('_', '-')

        possible_names = [
            name_lower,
            name_with_underscore,
            name_with_hyphen,
            package_name,  # Original case
        ]

        # Common license filenames (can be in root or licenses/ subdirectory)
        license_names = ['LICENSE', 'LICENSE.txt', 'LICENSE.md',
                        'LICENCE', 'LICENCE.txt',  # British spelling
                        'COPYING', 'COPYING.txt', 'COPYRIGHT']

        for name in possible_names:
            # Look for .dist-info directory (case-insensitive glob)
            for pattern in [f"{name}*.dist-info", f"{name.upper()}*.dist-info"]:
                dist_info_dirs = list(self.site_packages.glob(pattern))

                for dist_info in dist_info_dirs:
                    # Check root of dist-info
                    for license_name in license_names:
                        license_file = dist_info / license_name
                        if license_file.exists():
                            return license_file

                    # Check licenses/ subdirectory
                    licenses_dir = dist_info / 'licenses'
                    if licenses_dir.exists():
                        for license_file in licenses_dir.rglob('LICENSE*'):
                            if license_file.is_file():
                                return license_file

        return None

    def _normalize_package_name(self, package_name: str) -> str:
        """Normalize package name for filesystem."""
        return package_name.replace('-', '_').replace('.', '_')

    def collect_package_license(self, package_name: str) -> bool:
        """
        Collect license file for a specific package.

        Args:
            package_name: Name of the package to collect license for

        Returns:
            True if license was found and collected, False otherwise
        """
        print(f"Collecting license for: {package_name}")

        # Get package metadata
        metadata = self._get_package_metadata(package_name)
        if not metadata:
            print(f"  WARNING: Package not found: {package_name}")
            return False

        # Find license file
        license_file = self._find_license_file(package_name)
        if not license_file:
            print(f"  WARNING: License file not found for: {package_name}")
            return False

        # Create output directory for this package
        normalized_name = self._normalize_package_name(package_name)
        package_dir = self.output_dir / normalized_name
        package_dir.mkdir(parents=True, exist_ok=True)

        # Copy license file
        output_file = package_dir / 'LICENSE.txt'
        shutil.copy2(license_file, output_file)

        # Store metadata
        self.collected_licenses.append({
            'package': package_name,
            'version': metadata.get('Version', 'unknown'),
            'license': metadata.get('License', 'unknown'),
            'homepage': metadata.get('Home-page', ''),
            'author': metadata.get('Author', 'unknown'),
            'license_file': str(output_file.relative_to(self.output_dir.parent)),
        })

        print(f"  OK: License collected: {license_file.name}")
        return True

    def collect_all_licenses(self):
        """Collect licenses for all runtime packages."""
        print("\n" + "="*70)
        print("Collecting licenses for runtime dependencies...")
        print("="*70 + "\n")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collect licenses
        success_count = 0
        for package in self.RUNTIME_PACKAGES:
            if self.collect_package_license(package):
                success_count += 1

        print(f"\nOK: Collected {success_count}/{len(self.RUNTIME_PACKAGES)} licenses")

    def create_python_license_entry(self):
        """Add Python runtime license information."""
        print("\nAdding Python runtime license...")

        python_dir = self.output_dir / 'python'
        python_dir.mkdir(parents=True, exist_ok=True)

        # Get Python license
        try:
            result = subprocess.run(
                [sys.executable, '-c', 'import sys; print(sys.copyright)'],
                capture_output=True,
                text=True,
                check=True
            )
            copyright_info = result.stdout.strip()
        except:
            copyright_info = "Python Software Foundation"

        # Create Python license file
        python_license = python_dir / 'LICENSE.txt'
        python_license.write_text(f"""Python Software Foundation License

{copyright_info}

PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
2024, 2025 Python Software Foundation; All Rights Reserved" are retained in Python
alone or in any derivative version prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.

For more information: https://docs.python.org/3/license.html

BEOPEN.COM LICENSE AGREEMENT FOR PYTHON 2.0
-------------------------------------------

BEOPEN PYTHON OPEN SOURCE LICENSE AGREEMENT VERSION 1

1. This LICENSE AGREEMENT is between BeOpen.com ("BeOpen"), having an
office at 160 Saratoga Avenue, Santa Clara, CA 95051, and the
Individual or Organization ("Licensee") accessing and otherwise using
this software in source or binary form and its associated
documentation ("the Software").

[... Additional historical Python licenses would be included here ...]

Note: This is a permissive license that allows commercial use,
modification, and distribution without requiring source code disclosure.
""", encoding='utf-8')

        self.collected_licenses.append({
            'package': 'Python',
            'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'license': 'PSF License',
            'homepage': 'https://www.python.org',
            'author': 'Python Software Foundation',
            'license_file': str(python_license.relative_to(self.output_dir.parent)),
        })

        print("  OK: Python license added")

    def generate_notices_file(self):
        """Generate THIRD_PARTY_NOTICES.txt file."""
        print("\nGenerating THIRD_PARTY_NOTICES.txt...")

        notices_file = Path('THIRD_PARTY_NOTICES.txt')

        with open(notices_file, 'w', encoding='utf-8') as f:
            f.write("THIRD PARTY NOTICES\n")
            f.write("="*70 + "\n\n")
            f.write("This software includes or depends upon the following third-party\n")
            f.write("software components. Each component has its own license terms that\n")
            f.write("must be respected.\n\n")

            # Sort by package name
            sorted_licenses = sorted(self.collected_licenses, key=lambda x: x['package'].lower())

            for idx, info in enumerate(sorted_licenses, 1):
                f.write(f"{idx}. {info['package']} v{info['version']}\n")
                f.write(f"   License: {info['license']}\n")
                if info['homepage']:
                    f.write(f"   Homepage: {info['homepage']}\n")
                f.write(f"   Author: {info['author']}\n")
                f.write(f"   License File: {info['license_file']}\n")
                f.write("\n")

            f.write("\n" + "="*70 + "\n")
            f.write("IMPORTANT NOTICES\n")
            f.write("="*70 + "\n\n")

            f.write("1. LGPL NOTICE (libsndfile via soundfile)\n")
            f.write("-" * 70 + "\n")
            f.write("   The soundfile package depends on libsndfile, which is licensed\n")
            f.write("   under the GNU Lesser General Public License (LGPL).\n\n")
            f.write("   LGPL Compliance: libsndfile is dynamically linked. Users may\n")
            f.write("   replace this library with their own version if desired.\n\n")
            f.write("   libsndfile source code: https://github.com/libsndfile/libsndfile\n")
            f.write("   libsndfile license: GNU LGPL v2.1 or later\n\n")

            f.write("2. ML MODELS\n")
            f.write("-" * 70 + "\n")
            f.write("   This software downloads and uses machine learning models:\n\n")
            f.write("   - Parakeet STT Model (nemo-parakeet-tdt-0.6b-v3)\n")
            f.write("     License: Apache 2.0 (NVIDIA NeMo)\n")
            f.write("     Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b\n\n")
            f.write("   - Silero VAD Model (silero_vad.onnx)\n")
            f.write("     License: MIT\n")
            f.write("     Source: https://github.com/snakers4/silero-vad\n\n")

            f.write("3. ATTRIBUTION REQUIREMENT\n")
            f.write("-" * 70 + "\n")
            f.write("   All license texts in the LICENSES/ directory must be included\n")
            f.write("   in any distribution of this software. See individual license\n")
            f.write("   files for specific terms and conditions.\n\n")

            f.write("\n" + "="*70 + "\n")
            f.write(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n")

        print(f"  OK: THIRD_PARTY_NOTICES.txt created")

    def generate_readme(self):
        """Generate README.txt for LICENSES directory."""
        readme_file = self.output_dir / 'README.txt'

        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("LICENSES DIRECTORY\n")
            f.write("="*70 + "\n\n")
            f.write("This directory contains license files for all third-party software\n")
            f.write("components used in this application.\n\n")
            f.write("Structure:\n")
            f.write("  - Each subdirectory is named after a package\n")
            f.write("  - Each subdirectory contains the package's LICENSE.txt file\n\n")
            f.write("For attribution information, see: ../THIRD_PARTY_NOTICES.txt\n\n")
            f.write("IMPORTANT: These license files MUST be included in any distribution\n")
            f.write("of this software to comply with third-party license requirements.\n")

        print(f"  OK: LICENSES/README.txt created")
