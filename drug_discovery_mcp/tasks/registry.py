"""
Task Registry Module

Provides a registry for managing and organizing drug discovery tasks.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .task import Task, TaskCategory, TaskDifficulty, create_example_tasks

logger = logging.getLogger(__name__)


class TaskRegistry:
    """
    Registry for managing drug discovery tasks
    
    The TaskRegistry is responsible for:
    - Storing and organizing tasks
    - Loading tasks from files
    - Saving tasks to files
    - Categorizing tasks
    - Searching and filtering tasks
    """
    
    def __init__(self, tasks: Optional[List[Task]] = None):
        """
        Initialize the task registry
        
        Args:
            tasks: Initial list of tasks
        """
        self.tasks: Dict[str, Task] = {}
        self.categories: Dict[TaskCategory, List[str]] = {}
        self.difficulties: Dict[TaskDifficulty, List[str]] = {}
        
        # Initialize with example tasks
        if tasks is None:
            tasks = create_example_tasks()
        
        # Register all tasks
        for task in tasks:
            self.register_task(task)
        
        logger.info(f"TaskRegistry initialized with {len(self.tasks)} tasks")
    
    def register_task(self, task: Task) -> None:
        """
        Register a new task
        
        Args:
            task: Task to register
        """
        if task.id in self.tasks:
            logger.warning(f"Task {task.id} already registered, overwriting")
        
        self.tasks[task.id] = task
        
        # Add to category index
        if task.category not in self.categories:
            self.categories[task.category] = []
        if task.id not in self.categories[task.category]:
            self.categories[task.category].append(task.id)
        
        # Add to difficulty index
        if task.difficulty not in self.difficulties:
            self.difficulties[task.difficulty] = []
        if task.id not in self.difficulties[task.difficulty]:
            self.difficulties[task.difficulty].append(task.id)
        
        logger.debug(f"Registered task: {task.id} ({task.category.value})")
    
    def unregister_task(self, task_id: str) -> bool:
        """
        Unregister a task
        
        Args:
            task_id: ID of task to unregister
            
        Returns:
            True if task was unregistered, False if not found
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Remove from category index
        if task.category in self.categories:
            if task_id in self.categories[task.category]:
                self.categories[task.category].remove(task_id)
        
        # Remove from difficulty index
        if task.difficulty in self.difficulties:
            if task_id in self.difficulties[task.difficulty]:
                self.difficulties[task.difficulty].remove(task_id)
        
        # Remove from tasks
        del self.tasks[task_id]
        
        logger.debug(f"Unregistered task: {task_id}")
        return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            Task object or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_tasks(self) -> List[Task]:
        """
        Get all tasks
        
        Returns:
            List of all Task objects
        """
        return list(self.tasks.values())
    
    def get_tasks_by_category(self, category: TaskCategory) -> List[Task]:
        """
        Get tasks by category
        
        Args:
            category: Task category
            
        Returns:
            List of Task objects in the category
        """
        task_ids = self.categories.get(category, [])
        return [self.tasks[task_id] for task_id in task_ids if task_id in self.tasks]
    
    def get_tasks_by_difficulty(self, difficulty: TaskDifficulty) -> List[Task]:
        """
        Get tasks by difficulty
        
        Args:
            difficulty: Task difficulty
            
        Returns:
            List of Task objects with the difficulty
        """
        task_ids = self.difficulties.get(difficulty, [])
        return [self.tasks[task_id] for task_id in task_ids if task_id in self.tasks]
    
    def get_categories(self) -> List[TaskCategory]:
        """
        Get all task categories
        
        Returns:
            List of TaskCategory objects
        """
        return list(self.categories.keys())
    
    def get_difficulties(self) -> List[TaskDifficulty]:
        """
        Get all task difficulties
        
        Returns:
            List of TaskDifficulty objects
        """
        return list(self.difficulties.keys())
    
    def search_tasks(
        self,
        query: str,
        category: Optional[TaskCategory] = None,
        difficulty: Optional[TaskDifficulty] = None
    ) -> List[Task]:
        """
        Search for tasks by name, description, or prompt
        
        Args:
            query: Search query
            category: Filter by category
            difficulty: Filter by difficulty
            
        Returns:
            List of matching Task objects
        """
        query = query.lower()
        
        tasks = self.get_tasks()
        
        if category:
            tasks = [t for t in tasks if t.category == category]
        
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]
        
        # Search in name, description, and prompt
        results = []
        for task in tasks:
            if (query in task.name.lower() or 
                query in task.description.lower() or 
                query in task.prompt.lower()):
                results.append(task)
        
        return results
    
    def load_tasks(self, file_path: Union[str, Path]) -> bool:
        """
        Load tasks from a JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            True if tasks were loaded successfully
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear existing tasks
            self.tasks.clear()
            self.categories.clear()
            self.difficulties.clear()
            
            # Load tasks
            for task_data in data:
                task = Task.from_dict(task_data)
                self.register_task(task)
            
            logger.info(f"Loaded {len(self.tasks)} tasks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load tasks from {file_path}: {e}")
            return False
    
    def save_tasks(self, file_path: Union[str, Path]) -> bool:
        """
        Save tasks to a JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            True if tasks were saved successfully
        """
        file_path = Path(file_path)
        
        try:
            tasks_data = [task.to_dict() for task in self.get_tasks()]
            
            with open(file_path, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            logger.info(f"Saved {len(self.tasks)} tasks to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tasks to {file_path}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the task registry
        
        Returns:
            Dictionary with task statistics
        """
        return {
            "total_tasks": len(self.tasks),
            "categories": {
                category.value: len(task_ids) 
                for category, task_ids in self.categories.items()
            },
            "difficulties": {
                difficulty.value: len(task_ids) 
                for difficulty, task_ids in self.difficulties.items()
            },
            "category_count": len(self.categories),
            "difficulty_count": len(self.difficulties)
        }
    
    def add_example_tasks(self) -> int:
        """
        Add example tasks to the registry
        
        Returns:
            Number of tasks added
        """
        example_tasks = create_example_tasks()
        initial_count = len(self.tasks)
        
        for task in example_tasks:
            if task.id not in self.tasks:
                self.register_task(task)
        
        added_count = len(self.tasks) - initial_count
        logger.info(f"Added {added_count} example tasks")
        return added_count
    
    def clear(self) -> None:
        """Clear all tasks from the registry"""
        self.tasks.clear()
        self.categories.clear()
        self.difficulties.clear()
        logger.info("Cleared all tasks from registry")


# Global task registry instance
_global_registry = TaskRegistry()


def get_registry() -> TaskRegistry:
    """Get the global task registry"""
    return _global_registry


def set_registry(registry: TaskRegistry) -> None:
    """Set the global task registry"""
    global _global_registry
    _global_registry = registry
