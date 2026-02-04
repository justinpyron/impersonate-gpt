"""
Upload a local folder to a Modal volume.

Usage:
    python modal_upload.py /path/to/local/folder my-volume-name
"""

import argparse
import sys
from pathlib import Path

import modal


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload a local folder to a Modal volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "local_path",
        type=str,
        help="Path to the local folder to upload",
    )

    parser.add_argument(
        "volume_name",
        type=str,
        help="Name of the Modal volume",
    )

    parser.add_argument(
        "--remote-path",
        type=str,
        default="/",
        help="Remote path in the volume where files will be uploaded (default: /)",
    )

    return parser.parse_args()


def upload_folder(local_path: str, volume_name: str, remote_path: str = "/"):
    """
    Upload a local folder to a Modal volume.

    Args:
        local_path: Path to the local folder
        volume_name: Name of the Modal volume
        remote_path: Destination path in the volume
    """
    # Validate local path
    local_dir = Path(local_path)
    if not local_dir.exists():
        print(f"Error: Local path '{local_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not local_dir.is_dir():
        print(f"Error: Local path '{local_path}' is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading folder: {local_dir}")
    print(f"To Modal volume: {volume_name}")
    print(f"Remote path: {remote_path}")
    print("-" * 60)

    # Get or create the volume
    try:
        volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    except Exception as e:
        print(f"Error accessing volume '{volume_name}': {e}", file=sys.stderr)
        sys.exit(1)

    # Upload the folder
    try:
        # Collect all files to upload
        files_to_upload = []
        for item in local_dir.rglob("*"):
            if item.is_file():
                # Calculate relative path from local_dir
                rel_path = item.relative_to(local_dir)
                files_to_upload.append((str(item), str(rel_path)))

        if not files_to_upload:
            print("Warning: No files found to upload", file=sys.stderr)
            return

        print(f"Found {len(files_to_upload)} file(s) to upload")

        # Upload files using Modal's batch upload
        with volume.batch_upload() as batch:
            for local_file, rel_path in files_to_upload:
                # Construct the remote path
                remote_file_path = f"{remote_path.rstrip('/')}/{rel_path}"
                batch.put_file(local_file, remote_file_path)
                print(f"  Uploading: {rel_path}")

        print("-" * 60)
        print(
            f"✓ Successfully uploaded {len(files_to_upload)} file(s) to volume '{volume_name}'"
        )

    except Exception as e:
        print(f"Error during upload: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    upload_folder(args.local_path, args.volume_name, args.remote_path)


if __name__ == "__main__":
    main()
