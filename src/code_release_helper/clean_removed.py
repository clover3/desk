import os
from pathlib import Path


class BackupFileCleaner:
    def __init__(self, root_directory):
        self.root_directory = Path(root_directory)

    def find_backup_files(self, extensions=None):
        """Find all backup files with specified extensions."""
        if extensions is None:
            extensions = ['.removed', '.bak']

        backup_files = []

        for ext in extensions:
            # Find files ending with the extension
            pattern = f"*{ext}"
            backup_files.extend(self.root_directory.rglob(pattern))

        return backup_files

    def remove_backup_files(self, extensions=None, dry_run=True):
        """Remove all backup files with specified extensions."""
        if extensions is None:
            extensions = ['.removed', '.bak']

        backup_files = self.find_backup_files(extensions)

        print(f"Found {len(backup_files)} backup files")

        if dry_run:
            print("\n--- DRY RUN MODE ---")
            print("Files that would be DELETED:")
            for file_path in sorted(backup_files):
                file_size = file_path.stat().st_size if file_path.exists() else 0
                print(f"  {file_path} ({file_size:,} bytes)")

            total_size = sum(f.stat().st_size for f in backup_files if f.exists())
            print(f"\nTotal size: {total_size:,} bytes ({total_size / (1024 * 1024):.2f} MB)")
            return

        # Actually remove files
        removed_count = 0
        failed_count = 0
        total_size = 0

        for file_path in backup_files:
            try:
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    print(f"Deleting: {file_path}")
                    file_path.unlink()
                    removed_count += 1
                    total_size += file_size
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
                failed_count += 1

        print(f"\n--- SUMMARY ---")
        print(f"Successfully deleted: {removed_count} files")
        print(f"Failed to delete: {failed_count} files")
        print(f"Total space freed: {total_size:,} bytes ({total_size / (1024 * 1024):.2f} MB)")


def main():
    # Configuration variables
    directory = r"C:\work\code\desk\src"
    extensions = ['.removed', '.bak']  # File extensions to remove
    dry_run = False  # Set to False to actually delete files

    # Validate directory exists
    if not Path(directory).exists():
        print(f"Error: Directory does not exist: {directory}")
        return

    if not Path(directory).is_dir():
        print(f"Error: Path is not a directory: {directory}")
        return

    # Create cleaner and run
    cleaner = BackupFileCleaner(directory)

    if dry_run:
        print(f"Scanning directory: {directory}")
        print(f"Looking for files with extensions: {', '.join(extensions)}")
        print("Running in DRY-RUN mode (set dry_run=False to actually delete files)\n")
    else:
        print(f"WARNING: About to DELETE files in: {directory}")
        print(f"Extensions to remove: {', '.join(extensions)}")
        response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Operation cancelled")
            return
        print()

    cleaner.remove_backup_files(extensions=extensions, dry_run=dry_run)


if __name__ == "__main__":
    main()