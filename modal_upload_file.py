"""
Upload a local file to a Modal volume.

Usage:
    python modal_upload_file.py /path/to/local/file.txt my-volume-name /remote/path/file.txt
"""

import argparse
import sys
from pathlib import Path

import modal


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload a local file to a Modal volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "local_path",
        type=str,
        help="Path to the local file to upload",
    )

    parser.add_argument(
        "volume_name",
        type=str,
        help="Name of the Modal volume",
    )

    parser.add_argument(
        "remote_path",
        type=str,
        help="Remote path in the volume where file will be uploaded",
    )

    return parser.parse_args()


def upload_file(local_path: str, volume_name: str, remote_path: str):
    """
    Upload a local file to a Modal volume.

    Args:
        local_path: Path to the local file
        volume_name: Name of the Modal volume
        remote_path: Destination path in the volume
    """
    # Validate local path
    local_file = Path(local_path)
    if not local_file.exists():
        print(f"Error: Local path '{local_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not local_file.is_file():
        print(f"Error: Local path '{local_path}' is not a file", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading file: {local_file}")
    print(f"To Modal volume: {volume_name}")
    print(f"Remote path: {remote_path}")
    print("-" * 60)

    # Get the volume
    try:
        volume = modal.Volume.from_name(volume_name)
    except Exception as e:
        print(f"Error accessing volume '{volume_name}': {e}", file=sys.stderr)
        sys.exit(1)

    # Upload the file
    try:
        with volume.batch_upload() as batch:
            batch.put_file(local_path, remote_path)
        print("-" * 60)
        print(f"✓ Successfully uploaded file to volume '{volume_name}'")

    except Exception as e:
        print(f"Error during upload: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    upload_file(args.local_path, args.volume_name, args.remote_path)


if __name__ == "__main__":
    main()
