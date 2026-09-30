"""
Structured progress tracking and statistics utilities.

Provides progress tracking with counters, ETA estimates, operation statistics,
and error aggregation for long-running operations (ingestion, export, etc.).
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OperationStatistics:
    """Statistics tracking for long-running operations."""

    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    item_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "item_counts": dict(self.item_counts),
            "error_counts": dict(self.error_counts),
            "error_count": len(self.errors),
        }


class ProgressTracker:
    """
    Progress tracker with ETA estimates and statistics.
    
    Tracks progress of long-running operations with:
    - Progress counters and percentages
    - ETA (Estimated Time to Arrival) calculations
    - Operation statistics (successful, failed, skipped items)
    - Error aggregation and reporting
    - Custom logging messages
    
    Example:
        tracker = ProgressTracker(total=1000, description="Processing images")
        for item in items:
            try:
                process_item(item)
                tracker.update()
            except Exception as e:
                tracker.record_error("processing", str(e))
        tracker.finish()
    """

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the operation being tracked
        """
        if total < 0:
            raise ValueError(f"Total must be non-negative, got {total}")
        
        self.total = total
        self.description = description
        self.current = 0
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.last_update_count: int = 0
        
        # Statistics tracking
        self.stats = OperationStatistics(total_items=total)
        
        # Error tracking
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Start tracking
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_count = 0
        
        logger.info(f"Starting {description}: {total} items to process")

    def update(self, count: int = 1, success: bool = True) -> None:
        """
        Update progress counter.
        
        Args:
            count: Number of items processed in this update (default: 1)
            success: Whether the items were processed successfully (default: True)
        """
        if count < 0:
            raise ValueError(f"Count must be non-negative, got {count}")
        
        self.current += count
        self.stats.processed_items += count
        
        if success:
            self.stats.successful_items += count
        else:
            self.stats.failed_items += count
        
        # Update timing for ETA calculation
        current_time = time.time()
        self.last_update_time = current_time
        self.last_update_count = self.current
        
        # Log progress periodically (every 10% or every 100 items, whichever is smaller)
        log_interval = max(1, min(self.total // 10, 100))
        if self.current % log_interval == 0 or self.current == self.total:
            self._log_progress()

    def record_success(self, item_type: Optional[str] = None) -> None:
        """
        Record a successful item processing by type.
        
        Note: This method only tracks item_type counts. The overall successful_items
        counter is incremented by update(success=True). Do not call both methods
        for the same item, or use this method standalone if not using update().
        
        Args:
            item_type: Optional type/category of the item (e.g., "image", "annotation")
        """
        if item_type:
            self.stats.item_counts[item_type] += 1

    def record_failure(self, item_type: Optional[str] = None) -> None:
        """
        Record a failed item processing by type.
        
        Note: This method only tracks item_type counts. The overall failed_items
        counter is incremented by update(success=False). Do not call both methods
        for the same item, or use this method standalone if not using update().
        
        Args:
            item_type: Optional type/category of the item
        """
        if item_type:
            self.stats.item_counts[item_type] += 1

    def record_skip(self, item_type: Optional[str] = None, reason: Optional[str] = None) -> None:
        """
        Record a skipped item by type.
        
        Note: This method only tracks item_type counts and skip reasons. 
        The overall skipped_items counter should be incremented separately 
        if tracking overall skip counts.
        
        Args:
            item_type: Optional type/category of the item
            reason: Optional reason for skipping
        """
        if item_type:
            self.stats.item_counts[item_type] += 1
        if reason:
            self.error_counts[f"skipped_{reason}"] += 1

    def record_error(
        self,
        error_type: str,
        error_message: str,
        item_id: Optional[str] = None,
        item_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an error with details.
        
        Args:
            error_type: Type/category of error (e.g., "parsing", "validation", "database")
            error_message: Error message
            item_id: Optional identifier of the item that caused the error
            item_path: Optional file path of the item that caused the error
            details: Optional additional error details
        """
        error_record = {
            "type": error_type,
            "message": error_message,
            "timestamp": time.time(),
        }
        
        if item_id:
            error_record["item_id"] = item_id
        if item_path:
            error_record["item_path"] = item_path
        if details:
            error_record["details"] = details
        
        self.errors.append(error_record)
        self.stats.errors.append(error_record)
        self.error_counts[error_type] += 1
        self.stats.error_counts[error_type] += 1
        
        logger.debug(f"Error recorded: {error_type} - {error_message}")

    def log(self, message: str, level: int = logging.INFO) -> None:
        """
        Log a custom message.
        
        Args:
            message: Message to log
            level: Logging level (default: INFO)
        """
        logger.log(level, f"[{self.description}] {message}")

    def _log_progress(self) -> None:
        """Log current progress with ETA."""
        if self.current == 0:
            return
        
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Calculate ETA
        eta_seconds = self._calculate_eta()
        eta_str = self._format_duration(eta_seconds) if eta_seconds is not None else "calculating..."
        
        # Calculate rate
        rate = self.current / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(
            f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) "
            f"- Rate: {rate:.1f} items/s - ETA: {eta_str}"
        )

    def _calculate_eta(self) -> Optional[float]:
        """
        Calculate estimated time to arrival in seconds.
        
        Returns:
            ETA in seconds, or None if calculation is not possible
        """
        if self.current == 0 or self.total == 0:
            return None
        
        if self.start_time is None:
            return None
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time <= 0:
            return None
        
        # Calculate average rate
        rate = self.current / elapsed_time
        if rate <= 0:
            return None
        
        # Calculate remaining items
        remaining = self.total - self.current
        if remaining <= 0:
            return 0.0
        
        # ETA = remaining items / rate
        eta = remaining / rate
        return eta

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """
        Format duration in seconds to human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string (e.g., "2h 30m 15s", "45s")
        """
        if seconds < 0:
            return "0s"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)

    def finish(self) -> None:
        """
        Finish progress tracking and log final statistics.
        
        Logs completion message with final statistics including:
        - Total items processed
        - Success/failure counts
        - Processing time
        - Error summary
        """
        if self.start_time is None:
            return
        
        elapsed_time = time.time() - self.start_time
        elapsed_str = self._format_duration(elapsed_time)
        
        # Calculate final statistics
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        rate = self.current / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(
            f"Completed {self.description}: {self.current}/{self.total} ({percentage:.1f}%) "
            f"in {elapsed_str} ({rate:.1f} items/s)"
        )
        
        # Log statistics summary
        logger.info(
            f"Statistics - Successful: {self.stats.successful_items}, "
            f"Failed: {self.stats.failed_items}, "
            f"Skipped: {self.stats.skipped_items}"
        )
        
        # Log item type counts if any
        if self.stats.item_counts:
            counts_str = ", ".join(f"{k}: {v}" for k, v in self.stats.item_counts.items())
            logger.info(f"Item type counts: {counts_str}")
        
        # Log error summary
        if self.errors:
            logger.warning(f"Total errors: {len(self.errors)}")
            if self.error_counts:
                error_summary = ", ".join(f"{k}: {v}" for k, v in self.error_counts.items())
                logger.warning(f"Error breakdown: {error_summary}")
        else:
            logger.info("No errors encountered")

    def get_statistics(self) -> OperationStatistics:
        """
        Get current operation statistics.
        
        Returns:
            OperationStatistics object with current statistics
        """
        return self.stats

    def get_errors(self) -> List[Dict[str, Any]]:
        """
        Get all recorded errors.
        
        Returns:
            List of error dictionaries
        """
        return self.errors.copy()

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get error summary statistics.
        
        Returns:
            Dictionary with error counts and summary information
        """
        return {
            "total_errors": len(self.errors),
            "error_counts": dict(self.error_counts),
            "errors_by_type": {
                error_type: [
                    {
                        "message": e["message"],
                        "item_id": e.get("item_id"),
                        "item_path": e.get("item_path"),
                    }
                    for e in self.errors
                    if e["type"] == error_type
                ]
                for error_type in self.error_counts.keys()
            },
        }