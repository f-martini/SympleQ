#!/usr/bin/env python3
import re
import sys
import argparse


def generate_version(version_scheme="post-release", local_scheme="node-and-date", fallback_version="0.0.0.dev0"):
    try:
        import setuptools_scm
        version = setuptools_scm.get_version(
            version_scheme=version_scheme,
            local_scheme=local_scheme,
            fallback_version=fallback_version,
            tag_regex=r'^v?(\d+\.\d+\.\d+)(?:\.\d+)?$'
        )
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
    parser.add_argument('--version-scheme', default='post-release')
    parser.add_argument('--local-scheme', default='node-and-date')
    parser.add_argument('--fallback-version', default='0.0.0.dev0')
    parser.add_argument('--pyproject-path', default='pyproject.toml')

    args = parser.parse_args()

    version = generate_version(
        version_scheme=args.version_scheme,
        local_scheme=args.local_scheme,
        fallback_version=args.fallback_version
    )

    success = inject_version(version, args.pyproject_path)

    print(f"version={version}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
