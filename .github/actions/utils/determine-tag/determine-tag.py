#!/usr/bin/env python3
"""
Determine the next SemVer tag based on PR body and commit history.
"""
import os
import re
import sys
import subprocess
import json


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


def find_merged_pr(commit_sha, github_token):
    """Find merged PR for the given commit."""
    cmd = f'gh pr list --state merged --search {commit_sha} --json number --jq ".[0].number"'
    pr_number = run_command(cmd)

    if not pr_number or pr_number == "null":
        print("No merged PR found for this commit.", file=sys.stderr)
        sys.exit(1)

    return pr_number


def get_pr_body(pr_number, github_token):
    """Get PR body content."""
    cmd = f'gh pr view {pr_number} --json body --jq ".body"'
    return run_command(cmd)


def get_latest_tag():
    """Get the latest SemVer tag on dev branch."""
    run_command("git fetch --all --tags")
    cmd = "git describe --tags $(git rev-list --tags --remotes=origin/dev --max-count=1) 2>/dev/null || echo 'v0.0.0'"
    tag = run_command(cmd, check=False)
    print(f"Latest dev tag: {tag}")
    return tag if tag else "v0.0.0"


def determine_bump(pr_body):
    """Determine version bump type from PR body."""
    if re.search(r'\[x\]\s*Major', pr_body, re.IGNORECASE):
        return "major"
    elif re.search(r'\[x\]\s*Minor', pr_body, re.IGNORECASE):
        return "minor"
    elif re.search(r'\[x\]\s*Patch', pr_body, re.IGNORECASE):
        return "patch"
    else:
        return "commits"


def parse_version(tag):
    """Parse version string into components."""
    version = tag.lstrip('v')
    parts = version.split('.')

    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    extra = int(parts[3]) if len(parts) > 3 else 0

    return major, minor, patch, extra


def bump_version(tag, bump_type):
    """Calculate next version based on bump type."""
    major, minor, patch, extra = parse_version(tag)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
        extra = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
        extra = 0
    elif bump_type == "patch":
        patch += 1
        extra = 0

    return major, minor, patch, extra


def count_commits(tag):
    """Count commits on dev since last tag."""
    if tag == "v0.0.0":
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

    # Find merged PR
    pr_number = find_merged_pr(commit_sha, github_token)
    print(f"Found PR #{pr_number}")

    # Get PR body and determine bump type
    pr_body = get_pr_body(pr_number, github_token)
    bump_type = determine_bump(pr_body)
    print(f"Bump type: {bump_type}")

    # Get latest tag and calculate next version
    latest_tag = get_latest_tag()
    major, minor, patch, extra = bump_version(latest_tag, bump_type)

    # Calculate commit count if needed
    if bump_type == "commits":
        commit_count = count_commits(latest_tag) + extra
    else:
        commit_count = 0

    # Generate final tag
    final_tag = f"v{major}.{minor}.{patch}.{commit_count}"

    # Set outputs
    set_output("final_tag", final_tag)
    set_output("copy_tag", "false")

    print(f"Final tag: {final_tag}")


if __name__ == "__main__":
    main()
