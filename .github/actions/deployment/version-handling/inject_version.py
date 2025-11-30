#!/usr/bin/env python3
import re
import sys
import argparse


def generate_version(fallback_version="0.0.0.0a0",
                     include_distance=True,
                     is_prerelease=False):
    try:
        import setuptools_scm

        def custom_version_scheme(version):
            return version.format_with("{tag}a{distance}")

        def custom_local_scheme(version):
            return ""

        version = setuptools_scm.get_version(
            version_scheme=custom_version_scheme,
            local_scheme=custom_local_scheme,
            fallback_version=fallback_version,
            tag_regex=r'^v?(\d+\.\d+\.\d+)(?:\.\d+)?$'
        )

        # If include_distance is False, strip the 'aN' suffix
        if not include_distance and 'a' in version:
            version = version.split('a')[0]
            if is_prerelease and 'b' not in version:
                version = version + 'b' + "".join(version.split('a')[1:])

        return version
    except Exception:
        return fallback_version


def inject_version(version, pyproject_path="pyproject.toml"):
    try:
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return False

    content = re.sub(
        r'dynamic\s*=\s*\["version"\]',
        f'version = "{version}"',
        content
    )

    try:
        with open(pyproject_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fallback-version', default='0.0.0.0a0')
    parser.add_argument('--pyproject-path', default='pyproject.toml')
    parser.add_argument('--include-distance', default='true', choices=['true', 'false'],
                        help='Include distance (aN) in version for non-exact tags')
    parser.add_argument('--is-prerelease', default='false', choices=['true', 'false'],
                        help='Mark version as prerelease with b suffix')

    args = parser.parse_args()

    include_distance = args.include_distance.lower() == 'true'
    is_prerelease = args.is_prerelease.lower() == 'true'

    version = generate_version(
        fallback_version=args.fallback_version,
        include_distance=include_distance,
        is_prerelease=is_prerelease
    )

    success = inject_version(version, args.pyproject_path)

    print(f"version={version}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
