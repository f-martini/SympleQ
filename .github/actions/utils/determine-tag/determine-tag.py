#!/usr/bin/env python3
"""
Determine the next SemVer tag based on PR body and commit history.
"""
import os
import sys
import subprocess


PROJECT_VERSION = "0.0.0"


def run_command(cmd, check=True):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, encoding='utf-8',
        errors='replace', check=False
    )
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def set_output(name, value):
    """Set GitHub Actions output."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding='utf-8') as f:
            f.write(f"{name}={value}\n")
    print(f"{name}={value}")


def get_latest_tag():
    """Get the latest SemVer tag on dev branch."""
    run_command("git fetch --all --tags")
    cmd = "git describe --tags $(git rev-list --tags --remotes=origin/dev --max-count=1) 2>/dev/null"
    tag = run_command(cmd, check=False)
    print(f"Latest dev tag: {tag}")
    return tag if tag else f"v{PROJECT_VERSION}.0"


def get_valid_version(tag):
    """Parse version string into components and ensure single bump following semver logic."""
    version = tag.lstrip('v')
    parts = version.split('.')

    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    extra = int(parts[3]) if len(parts) > 3 else 0

    # Get the base version from PROJECT_VERSION
    base_parts = PROJECT_VERSION.split('.')
    base_major = int(base_parts[0]) if len(base_parts) > 0 else 0
    base_minor = int(base_parts[1]) if len(base_parts) > 1 else 0
    base_patch = int(base_parts[2]) if len(base_parts) > 2 else 0

    # Ensure single bump following semver logic
    if major > base_major:
        if major != base_major + 1 or (minor != 0 or patch != 0):
            print("Error: Invalid major version bump.", file=sys.stderr)
            sys.exit(1)
    elif major == base_major and minor > base_minor:
        if minor != base_minor + 1 or patch != 0:
            print("Error: Invalid minor version bump.", file=sys.stderr)
            sys.exit(1)
    elif major == base_major and minor == base_minor and patch > base_patch:
        if patch != base_patch + 1:
            print("Error: Invalid patch version bump.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Invalid version.")
        sys.exit(1)

    return major, minor, patch, extra


def count_commits(tag):
    """Count commits on dev since last tag."""
    if tag == f"v{PROJECT_VERSION}":
        cmd = "git rev-list --count origin/dev"
    else:
        cmd = f"git rev-list --count {tag}..origin/dev"

    count = run_command(cmd)
    return int(count) if count else 0


def main():
    # Get inputs from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    commit_sha = os.environ.get("COMMIT_SHA")

    if not github_token or not commit_sha:
        print("Error: GITHUB_TOKEN and COMMIT_SHA must be set", file=sys.stderr)
        sys.exit(1)

    # Configure git
    run_command('git config user.name "github-actions"')
    run_command('git config user.email "github-actions@github.com"')

    # Get latest tag and calculate next version
    latest_tag = get_latest_tag()
    major, minor, patch, extra = get_valid_version(latest_tag)

    # Calculate commit count
    commit_count = count_commits(latest_tag) + extra
    final_tag = f"v{major}.{minor}.{patch}.{commit_count}"

    # Set outputs
    set_output("final_tag", final_tag)
    set_output("copy_tag", "false")

    print(f"Final tag: {final_tag}")


if __name__ == "__main__":
    main()
