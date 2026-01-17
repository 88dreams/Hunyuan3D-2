"""
Tag Manager - Central tagging system for render outputs.

Manages tags for Gen3C, LTX-2, Hunyuan, Trellis, and 2DGS outputs.
Tags are stored in a central JSON file for easy querying and filtering.

Usage:
    from utils.tag_manager import TagManager
    
    tm = TagManager()
    tm.set_tags("/path/to/video.mp4", ["favorite", "approved"], model="Gen3C")
    tags = tm.get_tags("/path/to/video.mp4")
    videos = tm.list_files_by_tag("favorite", model_filter="Gen3C")
"""

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

# Default storage location
DEFAULT_TAGS_FILE = "/srv/searidge_share/outputs/tags.json"

# Default predefined tags
# Column 1: Positive tags
# Column 2: Action/status tags
DEFAULT_PREDEFINED_TAGS = [
    "approved",
    "favorite", 
    "best-take",
    "review",
    "bad",  # Marks low-quality renders
    "delete",  # Special tag - hides file from views and marks for deletion
]

# Valid model types
VALID_MODELS = ["Gen3C", "LTX-2", "Hunyuan", "Trellis", "2DGS"]


class TagManager:
    """
    Manages file tagging with thread-safe JSON storage.
    
    Data structure:
    {
        "files": {
            "/full/path/to/file": {
                "tags": ["tag1", "tag2"],
                "model": "Gen3C",
                "created": "2026-01-14T10:30:00",
                "tagged": "2026-01-14T10:35:00"
            }
        },
        "predefined_tags": ["favorite", "approved", ...],
        "custom_tags": ["client-a", "project-x"]
    }
    """
    
    def __init__(self, tags_file: Optional[str] = None):
        """
        Initialize TagManager.
        
        Args:
            tags_file: Path to the JSON storage file. Defaults to DEFAULT_TAGS_FILE.
        """
        self.tags_file = Path(tags_file or DEFAULT_TAGS_FILE)
        self._lock = Lock()
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """Create the tags file with default structure if it doesn't exist."""
        if not self.tags_file.exists():
            # Ensure parent directory exists
            self.tags_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write default structure
            default_data = {
                "files": {},
                "predefined_tags": DEFAULT_PREDEFINED_TAGS.copy(),
                "custom_tags": []
            }
            self._write_data(default_data)
    
    def _read_data(self) -> Dict[str, Any]:
        """Read and parse the JSON file."""
        try:
            with open(self.tags_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Return default structure if file is corrupted or missing
            return {
                "files": {},
                "predefined_tags": DEFAULT_PREDEFINED_TAGS.copy(),
                "custom_tags": []
            }
    
    def _write_data(self, data: Dict[str, Any]) -> None:
        """Write data to the JSON file."""
        with open(self.tags_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _normalize_path(self, file_path: str) -> str:
        """Normalize a file path to absolute path string."""
        return str(Path(file_path).resolve())
    
    # =========================================================================
    # Core CRUD Operations
    # =========================================================================
    
    def get_tags(self, file_path: str) -> List[str]:
        """
        Get tags for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of tags, or empty list if file not tagged
        """
        with self._lock:
            data = self._read_data()
            norm_path = self._normalize_path(file_path)
            file_entry = data["files"].get(norm_path, {})
            return file_entry.get("tags", [])
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get full tag info for a file including model and timestamps.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with tags, model, created, tagged - or None if not found
        """
        with self._lock:
            data = self._read_data()
            norm_path = self._normalize_path(file_path)
            return data["files"].get(norm_path)
    
    def set_tags(
        self, 
        file_path: str, 
        tags: List[str], 
        model: Optional[str] = None,
        created: Optional[str] = None
    ) -> None:
        """
        Set tags for a file (replaces existing tags).
        
        Args:
            file_path: Path to the file
            tags: List of tags to set
            model: Model that created the file (Gen3C, LTX-2, etc.)
            created: ISO timestamp when file was created (auto-detected if not provided)
        """
        with self._lock:
            data = self._read_data()
            norm_path = self._normalize_path(file_path)
            
            # Get or create file entry
            file_entry = data["files"].get(norm_path, {})
            
            # Update tags
            file_entry["tags"] = list(set(tags))  # Deduplicate
            file_entry["tagged"] = datetime.now().isoformat()
            
            # Set model if provided
            if model:
                if model not in VALID_MODELS:
                    print(f"[TagManager] Warning: Unknown model '{model}', expected one of {VALID_MODELS}")
                file_entry["model"] = model
            
            # Set created timestamp
            if created:
                file_entry["created"] = created
            elif "created" not in file_entry:
                # Try to get from file mtime
                try:
                    mtime = os.path.getmtime(file_path)
                    file_entry["created"] = datetime.fromtimestamp(mtime).isoformat()
                except OSError:
                    file_entry["created"] = datetime.now().isoformat()
            
            data["files"][norm_path] = file_entry
            
            # Track any new custom tags
            predefined = set(data.get("predefined_tags", []))
            custom = set(data.get("custom_tags", []))
            for tag in tags:
                if tag not in predefined and tag not in custom:
                    custom.add(tag)
            data["custom_tags"] = sorted(list(custom))
            
            self._write_data(data)
    
    def add_tag(self, file_path: str, tag: str, model: Optional[str] = None) -> None:
        """
        Add a single tag to a file.
        
        Args:
            file_path: Path to the file
            tag: Tag to add
            model: Model that created the file (optional, only set if not already set)
        """
        with self._lock:
            data = self._read_data()
            norm_path = self._normalize_path(file_path)
            
            file_entry = data["files"].get(norm_path, {"tags": []})
            
            if tag not in file_entry.get("tags", []):
                file_entry.setdefault("tags", []).append(tag)
                file_entry["tagged"] = datetime.now().isoformat()
                
                # Set model if not already set
                if model and "model" not in file_entry:
                    file_entry["model"] = model
                
                # Set created if not already set
                if "created" not in file_entry:
                    try:
                        mtime = os.path.getmtime(file_path)
                        file_entry["created"] = datetime.fromtimestamp(mtime).isoformat()
                    except OSError:
                        file_entry["created"] = datetime.now().isoformat()
                
                data["files"][norm_path] = file_entry
                
                # Track custom tag
                predefined = set(data.get("predefined_tags", []))
                custom = set(data.get("custom_tags", []))
                if tag not in predefined and tag not in custom:
                    custom.add(tag)
                    data["custom_tags"] = sorted(list(custom))
                
                self._write_data(data)
    
    def remove_tag(self, file_path: str, tag: str) -> None:
        """
        Remove a single tag from a file.
        
        Args:
            file_path: Path to the file
            tag: Tag to remove
        """
        with self._lock:
            data = self._read_data()
            norm_path = self._normalize_path(file_path)
            
            if norm_path in data["files"]:
                file_entry = data["files"][norm_path]
                if tag in file_entry.get("tags", []):
                    file_entry["tags"].remove(tag)
                    file_entry["tagged"] = datetime.now().isoformat()
                    self._write_data(data)
    
    def clear_tags(self, file_path: str) -> None:
        """
        Remove all tags from a file.
        
        Args:
            file_path: Path to the file
        """
        with self._lock:
            data = self._read_data()
            norm_path = self._normalize_path(file_path)
            
            if norm_path in data["files"]:
                del data["files"][norm_path]
                self._write_data(data)
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def list_files_by_tag(
        self, 
        tag: str, 
        model_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all files with a specific tag.
        
        Args:
            tag: Tag to search for
            model_filter: Optional model to filter by (Gen3C, LTX-2, etc.)
            
        Returns:
            List of dicts with 'path', 'tags', 'model', 'created', 'tagged'
        """
        with self._lock:
            data = self._read_data()
            results = []
            
            for path, info in data["files"].items():
                if tag in info.get("tags", []):
                    if model_filter is None or info.get("model") == model_filter:
                        results.append({
                            "path": path,
                            "tags": info.get("tags", []),
                            "model": info.get("model"),
                            "created": info.get("created"),
                            "tagged": info.get("tagged")
                        })
            
            # Sort by created date, newest first
            results.sort(key=lambda x: x.get("created", ""), reverse=True)
            return results
    
    def list_files_by_model(self, model: str) -> List[Dict[str, Any]]:
        """
        Get all tagged files for a specific model.
        
        Args:
            model: Model type (Gen3C, LTX-2, etc.)
            
        Returns:
            List of dicts with file info
        """
        with self._lock:
            data = self._read_data()
            results = []
            
            for path, info in data["files"].items():
                if info.get("model") == model:
                    results.append({
                        "path": path,
                        "tags": info.get("tags", []),
                        "model": info.get("model"),
                        "created": info.get("created"),
                        "tagged": info.get("tagged")
                    })
            
            results.sort(key=lambda x: x.get("created", ""), reverse=True)
            return results
    
    def list_all_tagged_files(
        self, 
        model_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all tagged files with optional filters.
        
        Args:
            model_filter: Optional model to filter by
            tag_filter: Optional list of tags (file must have ALL tags)
            
        Returns:
            List of dicts with file info
        """
        with self._lock:
            data = self._read_data()
            results = []
            
            for path, info in data["files"].items():
                # Apply model filter
                if model_filter and info.get("model") != model_filter:
                    continue
                
                # Apply tag filter (AND logic - must have all tags)
                if tag_filter:
                    file_tags = set(info.get("tags", []))
                    if not all(t in file_tags for t in tag_filter):
                        continue
                
                results.append({
                    "path": path,
                    "tags": info.get("tags", []),
                    "model": info.get("model"),
                    "created": info.get("created"),
                    "tagged": info.get("tagged")
                })
            
            results.sort(key=lambda x: x.get("created", ""), reverse=True)
            return results
    
    def search_files(
        self,
        tags: Optional[List[str]] = None,
        model: Optional[str] = None,
        filename_contains: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for files matching criteria.
        
        Args:
            tags: Tags to filter by (ANY match)
            model: Model to filter by
            filename_contains: Substring to match in filename
            
        Returns:
            List of matching file info dicts
        """
        with self._lock:
            data = self._read_data()
            results = []
            
            for path, info in data["files"].items():
                # Model filter
                if model and info.get("model") != model:
                    continue
                
                # Tag filter (OR logic - any tag matches)
                if tags:
                    file_tags = set(info.get("tags", []))
                    if not any(t in file_tags for t in tags):
                        continue
                
                # Filename filter
                if filename_contains:
                    if filename_contains.lower() not in Path(path).name.lower():
                        continue
                
                results.append({
                    "path": path,
                    "tags": info.get("tags", []),
                    "model": info.get("model"),
                    "created": info.get("created"),
                    "tagged": info.get("tagged")
                })
            
            results.sort(key=lambda x: x.get("created", ""), reverse=True)
            return results
    
    # =========================================================================
    # Tag Management
    # =========================================================================
    
    def get_all_tags(self) -> Dict[str, List[str]]:
        """
        Get all available tags (predefined + custom).
        
        Returns:
            Dict with 'predefined' and 'custom' tag lists
        """
        with self._lock:
            data = self._read_data()
            return {
                "predefined": data.get("predefined_tags", DEFAULT_PREDEFINED_TAGS.copy()),
                "custom": data.get("custom_tags", [])
            }
    
    def get_all_tags_flat(self) -> List[str]:
        """
        Get all available tags as a flat sorted list.
        
        Returns:
            Sorted list of all tags (predefined first, then custom)
        """
        tags = self.get_all_tags()
        return tags["predefined"] + sorted(tags["custom"])
    
    def get_predefined_tags(self) -> List[str]:
        """Get the list of predefined tags."""
        with self._lock:
            data = self._read_data()
            return data.get("predefined_tags", DEFAULT_PREDEFINED_TAGS.copy())
    
    def get_custom_tags(self) -> List[str]:
        """Get the list of user-created custom tags."""
        with self._lock:
            data = self._read_data()
            return data.get("custom_tags", [])
    
    def add_custom_tag(self, tag: str) -> None:
        """
        Add a new custom tag to the autocomplete list.
        
        Args:
            tag: Tag name to add
        """
        with self._lock:
            data = self._read_data()
            
            # Don't add if it's already predefined
            if tag in data.get("predefined_tags", []):
                return
            
            custom = set(data.get("custom_tags", []))
            custom.add(tag)
            data["custom_tags"] = sorted(list(custom))
            
            self._write_data(data)
    
    def remove_custom_tag(self, tag: str) -> None:
        """
        Remove a custom tag from the autocomplete list.
        Note: This doesn't remove the tag from files that have it.
        
        Args:
            tag: Tag name to remove
        """
        with self._lock:
            data = self._read_data()
            custom = data.get("custom_tags", [])
            
            if tag in custom:
                custom.remove(tag)
                data["custom_tags"] = custom
                self._write_data(data)
    
    def add_predefined_tag(self, tag: str) -> None:
        """
        Add a new predefined tag.
        
        Args:
            tag: Tag name to add
        """
        with self._lock:
            data = self._read_data()
            predefined = data.get("predefined_tags", [])
            
            if tag not in predefined:
                predefined.append(tag)
                data["predefined_tags"] = predefined
                
                # Remove from custom if it was there
                custom = data.get("custom_tags", [])
                if tag in custom:
                    custom.remove(tag)
                    data["custom_tags"] = custom
                
                self._write_data(data)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_tag_counts(self) -> Dict[str, int]:
        """
        Get count of files for each tag.
        
        Returns:
            Dict mapping tag name to file count
        """
        with self._lock:
            data = self._read_data()
            counts: Dict[str, int] = {}
            
            for info in data["files"].values():
                for tag in info.get("tags", []):
                    counts[tag] = counts.get(tag, 0) + 1
            
            return counts
    
    def get_model_counts(self) -> Dict[str, int]:
        """
        Get count of tagged files for each model.
        
        Returns:
            Dict mapping model name to file count
        """
        with self._lock:
            data = self._read_data()
            counts: Dict[str, int] = {}
            
            for info in data["files"].values():
                model = info.get("model", "Unknown")
                counts[model] = counts.get(model, 0) + 1
            
            return counts
    
    def file_exists_check(self, remove_missing: bool = False) -> List[str]:
        """
        Check which tagged files still exist on disk.
        
        Args:
            remove_missing: If True, remove entries for missing files
            
        Returns:
            List of paths that are missing
        """
        with self._lock:
            data = self._read_data()
            missing = []
            
            for path in list(data["files"].keys()):
                if not os.path.exists(path):
                    missing.append(path)
                    if remove_missing:
                        del data["files"][path]
            
            if remove_missing and missing:
                self._write_data(data)
            
            return missing
    
    def export_data(self) -> Dict[str, Any]:
        """Export all tag data for backup."""
        with self._lock:
            return self._read_data()
    
    def import_data(self, data: Dict[str, Any], merge: bool = True) -> None:
        """
        Import tag data.
        
        Args:
            data: Tag data dict to import
            merge: If True, merge with existing. If False, replace.
        """
        with self._lock:
            if merge:
                existing = self._read_data()
                
                # Merge files
                existing["files"].update(data.get("files", {}))
                
                # Merge custom tags
                custom = set(existing.get("custom_tags", []))
                custom.update(data.get("custom_tags", []))
                existing["custom_tags"] = sorted(list(custom))
                
                self._write_data(existing)
            else:
                self._write_data(data)
    
    # =========================================================================
    # Delete Tag Handling
    # =========================================================================
    
    def is_marked_for_delete(self, file_path: str) -> bool:
        """Check if a file is marked with the 'delete' tag."""
        tags = self.get_tags(file_path)
        return "delete" in tags
    
    def get_files_marked_for_delete(self) -> List[str]:
        """Get all file paths marked with 'delete' tag."""
        with self._lock:
            data = self._read_data()
            deleted = []
            
            for path, info in data["files"].items():
                if "delete" in info.get("tags", []):
                    deleted.append(path)
            
            return deleted
    
    def delete_marked_files(self, dry_run: bool = False) -> List[Tuple[str, bool]]:
        """
        Delete files marked with 'delete' tag from disk.
        
        Args:
            dry_run: If True, just return list without actually deleting
            
        Returns:
            List of (path, success) tuples
        """
        results = []
        files_to_delete = self.get_files_marked_for_delete()
        
        for file_path in files_to_delete:
            if dry_run:
                results.append((file_path, True))
                continue
            
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # Also remove from tag database
                    self.clear_tags(file_path)
                    results.append((file_path, True))
                else:
                    # File already gone, just clean up tag entry
                    self.clear_tags(file_path)
                    results.append((file_path, True))
            except OSError as e:
                print(f"[TagManager] Failed to delete {file_path}: {e}")
                results.append((file_path, False))
        
        return results
    
    def clear_tags_for_files(self, file_paths: List[str]) -> int:
        """
        Clear all tags from multiple files.
        
        Args:
            file_paths: List of file paths to clear tags from
            
        Returns:
            Number of files cleared
        """
        cleared = 0
        for path in file_paths:
            self.clear_tags(path)
            cleared += 1
        return cleared


# Module-level singleton instance
_default_manager: Optional[TagManager] = None


def get_tag_manager() -> TagManager:
    """Get or create the default TagManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TagManager()
    return _default_manager


# Convenience functions using the default manager
def get_tags(file_path: str) -> List[str]:
    """Get tags for a file using the default manager."""
    return get_tag_manager().get_tags(file_path)


def set_tags(file_path: str, tags: List[str], model: Optional[str] = None) -> None:
    """Set tags for a file using the default manager."""
    get_tag_manager().set_tags(file_path, tags, model)


def add_tag(file_path: str, tag: str, model: Optional[str] = None) -> None:
    """Add a tag to a file using the default manager."""
    get_tag_manager().add_tag(file_path, tag, model)


def remove_tag(file_path: str, tag: str) -> None:
    """Remove a tag from a file using the default manager."""
    get_tag_manager().remove_tag(file_path, tag)


if __name__ == "__main__":
    # Test the TagManager
    print("Testing TagManager...")
    
    tm = TagManager("/tmp/test_tags.json")
    
    # Test setting tags
    tm.set_tags("/test/video1.mp4", ["favorite", "approved"], model="Gen3C")
    tm.set_tags("/test/video2.mp4", ["review"], model="LTX-2")
    tm.add_tag("/test/video2.mp4", "custom-tag")
    
    # Test getting tags
    print(f"video1 tags: {tm.get_tags('/test/video1.mp4')}")
    print(f"video2 tags: {tm.get_tags('/test/video2.mp4')}")
    
    # Test listing
    print(f"Files with 'favorite': {tm.list_files_by_tag('favorite')}")
    print(f"All Gen3C files: {tm.list_files_by_model('Gen3C')}")
    
    # Test tag management
    print(f"All tags: {tm.get_all_tags()}")
    print(f"Tag counts: {tm.get_tag_counts()}")
    
    print("\nTagManager test complete!")
