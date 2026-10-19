"""Check staged Git blobs before committing: every file must be below 20 MB."""
import subprocess

paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
too_large = []
for path in filter(None, paths):
    size = int(subprocess.check_output(["git", "cat-file", "-s", ":" + path]))
    if size >= 20_000_000:
        too_large.append((path, size))
if too_large:
    raise SystemExit(f"Files at or above 20 MB belong on Hugging Face: {too_large}")
print("All staged/tracked files are below 20,000,000 bytes.")
