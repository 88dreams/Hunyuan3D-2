"""
Version tracking utilities for upstream model repositories.

This module provides functions to check the latest commits from upstream
repositories and compare them with locally deployed versions.
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Tuple, Optional

# Repository information for each model
UPSTREAM_REPOS = {
    "sharp": {
        "name": "SHARP",
        "owner": "apple",
        "repo": "ml-sharp",
        "branch": "main",
        "url": "https://github.com/apple/ml-sharp",
    },
    "gen3c": {
        "name": "GEN3C",
        "owner": "nv-tlabs",
        "repo": "GEN3C",
        "branch": "main",
        "url": "https://github.com/nv-tlabs/GEN3C",
    },
    "lyra": {
        "name": "Lyra",
        "owner": "nv-tlabs",
        "repo": "LYRA",
        "branch": "main",
        "url": "https://github.com/nv-tlabs/LYRA",
    },
    "trellis": {
        "name": "TRELLIS.2",
        "owner": "microsoft",
        "repo": "TRELLIS",
        "branch": "main",
        "url": "https://github.com/microsoft/TRELLIS",
    },
    "hunyuan": {
        "name": "Hunyuan3D",
        "owner": "Tencent",
        "repo": "Hunyuan3D-2",
        "branch": "main",
        "url": "https://github.com/Tencent/Hunyuan3D-2",
    },
}

# Local version tracking file
VERSION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".version_tracking.json")


def _load_local_versions() -> Dict:
    """Load locally tracked versions from file."""
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_local_versions(versions: Dict) -> None:
    """Save local versions to file."""
    try:
        with open(VERSION_FILE, "w") as f:
            json.dump(versions, f, indent=2)
    except Exception as e:
        print(f"[VERSION] Warning: Could not save versions: {e}")


def save_version_notes(notes: str) -> str:
    """Save version notes to the tracking file."""
    versions = _load_local_versions()
    versions["notes"] = notes
    versions["notes_updated"] = datetime.now().isoformat()
    _save_local_versions(versions)
    return f"Saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def load_version_notes() -> str:
    """Load version notes from the tracking file."""
    versions = _load_local_versions()
    return versions.get("notes", """SHARP: Not yet tracked
GEN3C: Not yet tracked
Lyra: Not yet tracked
TRELLIS: Not yet tracked
Hunyuan3D: Not yet tracked""")


def get_upstream_commit(model_key: str) -> Tuple[str, str]:
    """
    Get the latest commit info from upstream repository.
    
    Returns:
        Tuple of (commit_sha_short, commit_date) or (error_message, "")
    """
    if model_key not in UPSTREAM_REPOS:
        return "Unknown model", ""
    
    repo_info = UPSTREAM_REPOS[model_key]
    api_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}/commits/{repo_info['branch']}"
    
    try:
        headers = {"Accept": "application/vnd.github.v3+json"}
        # Add token if available for higher rate limits
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            sha_short = data["sha"][:7]
            commit_date = data["commit"]["committer"]["date"][:10]  # YYYY-MM-DD
            return sha_short, commit_date
        elif response.status_code == 403:
            return "Rate limited", ""
        elif response.status_code == 404:
            return "Repo not found", ""
        else:
            return f"Error {response.status_code}", ""
    except requests.exceptions.Timeout:
        return "Timeout", ""
    except Exception as e:
        return f"Error: {str(e)[:20]}", ""


def get_local_version(model_key: str) -> str:
    """Get the locally tracked version for a model."""
    versions = _load_local_versions()
    return versions.get(f"{model_key}_version", "Unknown")


def set_local_version(model_key: str, version: str) -> None:
    """Set the locally tracked version for a model."""
    versions = _load_local_versions()
    versions[f"{model_key}_version"] = version
    versions[f"{model_key}_updated"] = datetime.now().isoformat()
    _save_local_versions(versions)


def check_all_versions() -> Dict[str, Dict]:
    """
    Check all upstream versions and compare with local.
    
    Returns:
        Dict with model keys containing upstream and local version info
    """
    results = {}
    
    for model_key in UPSTREAM_REPOS:
        upstream_sha, upstream_date = get_upstream_commit(model_key)
        local_version = get_local_version(model_key)
        
        if upstream_date:
            upstream_display = f"{upstream_sha} ({upstream_date})"
        else:
            upstream_display = upstream_sha
        
        results[model_key] = {
            "upstream": upstream_display,
            "local": local_version,
            "name": UPSTREAM_REPOS[model_key]["name"],
        }
    
    return results


def format_version_check_status(results: Dict) -> str:
    """Format the version check results for display."""
    lines = []
    for model_key, info in results.items():
        status = "✓" if info["local"] != "Unknown" else "?"
        lines.append(f"{status} {info['name']}: upstream={info['upstream']}")
    
    return " | ".join(lines)


def query_runpod_versions(
    unified_endpoint: str,
    unified_api_key: str,
    trellis_endpoint: str = "",
    trellis_api_key: str = "",
    hunyuan_endpoint: str = "",
    hunyuan_api_key: str = "",
) -> Dict[str, Dict]:
    """
    Query RunPod endpoints for their deployed versions.
    
    The unified endpoint (GEN3C/SHARP/Lyra) returns versions for all models.
    TRELLIS and Hunyuan have separate endpoints.
    
    Args:
        unified_endpoint: Endpoint ID for the unified handler (GEN3C/SHARP/Lyra)
        unified_api_key: API key for the unified endpoint
        trellis_endpoint: Optional endpoint ID for TRELLIS
        trellis_api_key: Optional API key for TRELLIS
        hunyuan_endpoint: Optional endpoint ID for Hunyuan
        hunyuan_api_key: Optional API key for Hunyuan
    
    Returns:
        Dict with version info for each model
    """
    results = {
        "sharp": {"display": "Not queried", "error": None},
        "gen3c": {"display": "Not queried", "error": None},
        "lyra": {"display": "Not queried", "error": None},
        "trellis": {"display": "Not queried", "error": None},
        "hunyuan": {"display": "Not queried", "error": None},
    }
    
    # Query unified endpoint for SHARP/GEN3C/Lyra
    if unified_endpoint and unified_api_key:
        try:
            url = f"https://api.runpod.ai/v2/{unified_endpoint}/runsync"
            headers = {
                "Authorization": f"Bearer {unified_api_key}",
                "Content-Type": "application/json",
            }
            payload = {"input": {"action": "version"}}
            
            # Use longer timeout for cold starts (serverless workers may take 60+ seconds)
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "")
                
                if status == "COMPLETED":
                    output = data.get("output", {})
                    versions = output.get("versions", {})
                    
                    for model in ["sharp", "gen3c", "lyra", "trellis"]:
                        if model in versions:
                            results[model] = versions[model]
                elif status == "IN_QUEUE":
                    for model in ["sharp", "gen3c", "lyra"]:
                        results[model]["error"] = "Worker starting (cold start)"
                elif status == "IN_PROGRESS":
                    for model in ["sharp", "gen3c", "lyra"]:
                        results[model]["error"] = "Worker still processing"
                else:
                    error = data.get("error", f"Status: {status}")
                    for model in ["sharp", "gen3c", "lyra"]:
                        results[model]["error"] = f"Job failed: {error}"
            else:
                error = f"HTTP {response.status_code}: {response.text[:100]}"
                for model in ["sharp", "gen3c", "lyra"]:
                    results[model]["error"] = error
        except requests.exceptions.Timeout:
            for model in ["sharp", "gen3c", "lyra"]:
                results[model]["error"] = "Timeout (>120s) - worker may be starting"
        except Exception as e:
            for model in ["sharp", "gen3c", "lyra"]:
                results[model]["error"] = str(e)[:50]
    
    # Query TRELLIS endpoint separately if configured
    if trellis_endpoint and trellis_api_key:
        try:
            url = f"https://api.runpod.ai/v2/{trellis_endpoint}/runsync"
            headers = {
                "Authorization": f"Bearer {trellis_api_key}",
                "Content-Type": "application/json",
            }
            payload = {"input": {"action": "version"}}
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "")
                if status == "COMPLETED":
                    output = data.get("output", {})
                    versions = output.get("versions", {})
                    if "trellis" in versions:
                        results["trellis"] = versions["trellis"]
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    results["trellis"]["error"] = f"Worker starting ({status})"
                else:
                    results["trellis"]["error"] = f"Job failed: {data.get('error', status)}"
            else:
                results["trellis"]["error"] = f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            results["trellis"]["error"] = "Timeout (>120s) - worker may be starting"
        except Exception as e:
            results["trellis"]["error"] = str(e)[:80]
    
    # Query Hunyuan endpoint separately if configured
    if hunyuan_endpoint and hunyuan_api_key:
        try:
            url = f"https://api.runpod.ai/v2/{hunyuan_endpoint}/runsync"
            headers = {
                "Authorization": f"Bearer {hunyuan_api_key}",
                "Content-Type": "application/json",
            }
            payload = {"input": {"action": "version"}}
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "")
                if status == "COMPLETED":
                    output = data.get("output", {})
                    versions = output.get("versions", {})
                    if "hunyuan" in versions:
                        results["hunyuan"] = versions["hunyuan"]
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    results["hunyuan"]["error"] = f"Worker starting ({status})"
                else:
                    results["hunyuan"]["error"] = f"Job failed: {data.get('error', status)}"
            else:
                results["hunyuan"]["error"] = f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            results["hunyuan"]["error"] = "Timeout (>120s) - worker may be starting"
        except Exception as e:
            results["hunyuan"]["error"] = str(e)[:80]
    
    return results


def update_local_versions_from_runpod(versions: Dict[str, Dict]) -> None:
    """
    Update local version tracking file with versions queried from RunPod.
    
    Args:
        versions: Dict from query_runpod_versions()
    """
    for model_key, info in versions.items():
        display = info.get("display", "")
        if display and display not in ["Not queried", "unknown", "Not installed"]:
            if not info.get("error"):
                set_local_version(model_key, display)

