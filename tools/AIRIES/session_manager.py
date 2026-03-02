# session_manager.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import uuid


class SessionManager:
    """
    Advanced session management for region-based semi-automatic picker.
    Handles save/load/resume functionality for complex picking workflows.
    """

    def __init__(self, base_output_dir: Union[str, Path]):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Session metadata
        self.current_session = None
        self.session_history = []
        self.auto_save_enabled = True
        self.backup_count = 5

        # Session file paths
        self.sessions_dir = self.base_output_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.backups_dir = self.sessions_dir / "backups"
        self.backups_dir.mkdir(exist_ok=True)

    def create_new_session(
        self, image_filename: str, session_name: Optional[str] = None
    ) -> str:
        """
        Create a new picking session.

        Args:
            image_filename: Name of the radar image being processed
            session_name: Optional custom session name

        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        if session_name is None:
            session_name = f"session_{Path(image_filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session_data = {
            "session_id": session_id,
            "session_name": session_name,
            "image_filename": image_filename,
            "created_timestamp": timestamp,
            "last_modified": timestamp,
            "version": 1,
            "status": "active",
            "workflow_type": "region_based_semi_automatic",
            "regions": {"surface": [], "bed": []},
            "global_settings": {
                "template_params": {
                    "window_size": (20, 10),
                    "confidence_threshold": 0.7,
                    "max_search_range": 30,
                    "template_update_rate": 0.3,
                },
                "detection_params": {
                    "use_multi_scale": True,
                    "prominence_range": [15, 50],
                    "scales": 4,
                },
            },
            "processing_history": [],
            "quality_metrics": {
                "surface_coverage": 0.0,
                "bed_coverage": 0.0,
                "total_control_points": 0,
                "avg_confidence_score": 0.0,
            },
            "user_annotations": {"notes": [], "problem_areas": [], "quality_flags": []},
        }

        self.current_session = session_data

        # Auto-save new session
        if self.auto_save_enabled:
            self.save_session()

        print(f"INFO: Created new session '{session_name}' with ID: {session_id}")
        return session_id

    def save_session(self, session_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Save current session to file.

        Args:
            session_path: Optional custom save path

        Returns:
            bool: Success status
        """
        if self.current_session is None:
            print("WARNING: No active session to save")
            return False

        try:
            # Update last modified timestamp
            self.current_session["last_modified"] = datetime.now().isoformat()
            self.current_session["version"] += 1

            # Determine save path
            if session_path is None:
                session_filename = f"{self.current_session['session_name']}.json"
                session_path = self.sessions_dir / session_filename
            else:
                session_path = Path(session_path)

            # Create backup of existing file if it exists
            if session_path.exists():
                self._create_backup(session_path)

            # Convert numpy arrays to JSON-serializable format
            serializable_data = self._make_json_serializable(self.current_session)

            # Save to file with pretty formatting
            with open(session_path, "w") as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

            print(f"INFO: Session saved to {session_path}")
            return True

        except Exception as e:
            print(f"ERROR: Failed to save session: {e}")
            return False

    def load_session(self, session_path: Union[str, Path]) -> bool:
        """
        Load session from file.

        Args:
            session_path: Path to session file

        Returns:
            bool: Success status
        """
        session_path = Path(session_path)

        if not session_path.exists():
            print(f"ERROR: Session file not found: {session_path}")
            return False

        try:
            with open(session_path, "r") as f:
                session_data = json.load(f)

            # Validate session format
            if not self._validate_session_format(session_data):
                print("ERROR: Invalid session file format")
                return False

            self.current_session = session_data

            # Add to session history
            self.session_history.append(
                {
                    "session_id": session_data.get("session_id"),
                    "loaded_timestamp": datetime.now().isoformat(),
                    "session_path": str(session_path),
                }
            )

            print(
                f"INFO: Loaded session '{session_data['session_name']}' (ID: {session_data['session_id']})"
            )
            return True

        except Exception as e:
            print(f"ERROR: Failed to load session: {e}")
            return False

    def resume_session(self, session_id: str) -> bool:
        """
        Resume a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            bool: Success status
        """
        # Search for session file
        session_files = list(self.sessions_dir.glob("*.json"))

        for session_file in session_files:
            try:
                with open(session_file, "r") as f:
                    session_data = json.load(f)

                if session_data.get("session_id") == session_id:
                    return self.load_session(session_file)

            except Exception:
                continue

        print(f"ERROR: Session with ID '{session_id}' not found")
        return False

    def update_regions_data(self, regions_data: Dict[str, List[Dict]]) -> bool:
        """
        Update regions data in current session.

        Args:
            regions_data: Regions data from RegionsManager

        Returns:
            bool: Success status
        """
        if self.current_session is None:
            print("WARNING: No active session to update")
            return False

        try:
            # Update regions
            self.current_session["regions"] = self._make_json_serializable(regions_data)

            # Update quality metrics
            self._update_quality_metrics()

            # Auto-save if enabled
            if self.auto_save_enabled:
                self.save_session()

            return True

        except Exception as e:
            print(f"ERROR: Failed to update regions data: {e}")
            return False

    def add_processing_event(self, event_type: str, details: Dict[str, Any]) -> bool:
        """
        Add processing event to session history.

        Args:
            event_type: Type of processing event
            details: Event details

        Returns:
            bool: Success status
        """
        if self.current_session is None:
            return False

        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": self._make_json_serializable(details),
        }

        self.current_session["processing_history"].append(event)

        # Auto-save if enabled
        if self.auto_save_enabled:
            self.save_session()

        return True

    def add_user_note(self, note: str, location: Optional[Dict] = None) -> bool:
        """
        Add user annotation note.

        Args:
            note: User note text
            location: Optional location information

        Returns:
            bool: Success status
        """
        if self.current_session is None:
            return False

        annotation = {
            "timestamp": datetime.now().isoformat(),
            "note": note,
            "location": location,
        }

        self.current_session["user_annotations"]["notes"].append(annotation)

        # Auto-save if enabled
        if self.auto_save_enabled:
            self.save_session()

        return True

    def get_session_summary(self) -> Optional[Dict]:
        """
        Get summary of current session.

        Returns:
            Optional[Dict]: Session summary or None if no active session
        """
        if self.current_session is None:
            return None

        surface_regions = self.current_session["regions"].get("surface", [])
        bed_regions = self.current_session["regions"].get("bed", [])

        summary = {
            "session_info": {
                "id": self.current_session["session_id"],
                "name": self.current_session["session_name"],
                "image": self.current_session["image_filename"],
                "status": self.current_session["status"],
                "created": self.current_session["created_timestamp"],
                "last_modified": self.current_session["last_modified"],
                "version": self.current_session["version"],
            },
            "regions_summary": {
                "total_regions": len(surface_regions) + len(bed_regions),
                "surface_regions": len(surface_regions),
                "bed_regions": len(bed_regions),
                "total_control_points": sum(
                    len(r.get("control_points", []))
                    for r in surface_regions + bed_regions
                ),
            },
            "quality_metrics": self.current_session["quality_metrics"],
            "processing_events": len(self.current_session["processing_history"]),
            "user_notes": len(self.current_session["user_annotations"]["notes"]),
        }

        return summary

    def export_session_report(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Export detailed session report.

        Args:
            output_path: Optional output path for report

        Returns:
            bool: Success status
        """
        if self.current_session is None:
            return False

        try:
            # Generate comprehensive report
            report = {
                "session_summary": self.get_session_summary(),
                "detailed_regions": self.current_session["regions"],
                "processing_timeline": self.current_session["processing_history"],
                "user_annotations": self.current_session["user_annotations"],
                "global_settings": self.current_session["global_settings"],
                "export_timestamp": datetime.now().isoformat(),
            }

            # Determine output path
            if output_path is None:
                report_filename = f"{self.current_session['session_name']}_report.json"
                output_path = self.sessions_dir / report_filename
            else:
                output_path = Path(output_path)

            # Save report
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"INFO: Session report exported to {output_path}")
            return True

        except Exception as e:
            print(f"ERROR: Failed to export session report: {e}")
            return False

    def list_available_sessions(self) -> List[Dict]:
        """
        List all available sessions.

        Returns:
            List[Dict]: List of session metadata
        """
        sessions = []
        session_files = list(self.sessions_dir.glob("*.json"))

        for session_file in session_files:
            try:
                with open(session_file, "r") as f:
                    session_data = json.load(f)

                sessions.append(
                    {
                        "session_id": session_data.get("session_id"),
                        "session_name": session_data.get("session_name"),
                        "image_filename": session_data.get("image_filename"),
                        "created_timestamp": session_data.get("created_timestamp"),
                        "last_modified": session_data.get("last_modified"),
                        "status": session_data.get("status"),
                        "file_path": str(session_file),
                    }
                )

            except Exception as e:
                print(f"WARNING: Could not read session file {session_file}: {e}")
                continue

        # Sort by last modified (most recent first)
        sessions.sort(key=lambda x: x.get("last_modified", ""), reverse=True)

        return sessions

    def cleanup_old_sessions(self, days_to_keep: int = 30) -> int:
        """
        Clean up old session files.

        Args:
            days_to_keep: Number of days to keep sessions

        Returns:
            int: Number of sessions cleaned up
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleanup_count = 0

        sessions = self.list_available_sessions()

        for session in sessions:
            try:
                last_modified = datetime.fromisoformat(session["last_modified"])

                if last_modified < cutoff_date:
                    session_path = Path(session["file_path"])

                    # Move to backups before deletion
                    backup_path = self.backups_dir / session_path.name
                    session_path.rename(backup_path)

                    cleanup_count += 1
                    print(
                        f"INFO: Moved old session to backup: {session['session_name']}"
                    )

            except Exception as e:
                print(
                    f"WARNING: Could not cleanup session {session['session_name']}: {e}"
                )
                continue

        return cleanup_count

    def _create_backup(self, file_path: Path) -> bool:
        """Create backup of existing session file."""
        try:
            backup_name = f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = self.backups_dir / backup_name

            # Copy existing file to backup
            import shutil

            shutil.copy2(file_path, backup_path)

            # Clean up old backups (keep only last N backups)
            backup_files = list(
                self.backups_dir.glob(f"{file_path.stem}_backup_*.json")
            )
            if len(backup_files) > self.backup_count:
                # Sort by creation time and remove oldest
                backup_files.sort(key=lambda x: x.stat().st_mtime)
                for old_backup in backup_files[: -self.backup_count]:
                    old_backup.unlink()

            return True

        except Exception as e:
            print(f"WARNING: Could not create backup: {e}")
            return False

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(v) for v in obj]
        elif hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            try:
                # Test if object is JSON serializable
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # Return string representation for non-serializable objects
                return str(obj)

    def _validate_session_format(self, session_data: Dict) -> bool:
        """Validate session file format."""
        required_fields = [
            "session_id",
            "session_name",
            "image_filename",
            "created_timestamp",
            "workflow_type",
            "regions",
        ]

        for field in required_fields:
            if field not in session_data:
                print(f"ERROR: Missing required field in session: {field}")
                return False

        return True

    def _update_quality_metrics(self) -> None:
        """Update quality metrics based on current regions data."""
        if self.current_session is None:
            return

        surface_regions = self.current_session["regions"].get("surface", [])
        bed_regions = self.current_session["regions"].get("bed", [])

        total_control_points = sum(
            len(r.get("control_points", [])) for r in surface_regions + bed_regions
        )

        # Calculate coverage estimates (simplified)
        surface_coverage = len(surface_regions) * 10.0  # Rough estimate
        bed_coverage = len(bed_regions) * 10.0

        # Calculate average confidence (if available)
        confidence_scores = []
        for region in surface_regions + bed_regions:
            if "confidence_scores" in region:
                confidence_scores.extend(region["confidence_scores"])

        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        # Update metrics
        self.current_session["quality_metrics"] = {
            "surface_coverage": min(100.0, surface_coverage),
            "bed_coverage": min(100.0, bed_coverage),
            "total_control_points": total_control_points,
            "avg_confidence_score": float(avg_confidence),
        }
