"""
Task Management Module

This module provides a task management system for DrugDiscoveryBench-style tasks.
It includes:
- Task definitions and categories
- Task execution engine
- Task result management
- Task validation and grading
"""

from .task import Task, TaskCategory, TaskStatus
from .runner import TaskRunner
from .registry import TaskRegistry
from .evaluation import TaskEvaluator

__all__ = [
    "Task",
    "TaskCategory",
    "TaskStatus",
    "TaskRunner",
    "TaskRegistry",
    "TaskEvaluator",
]
