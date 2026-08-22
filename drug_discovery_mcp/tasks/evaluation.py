"""
Task Evaluation Module

Provides evaluation and grading functionality for drug discovery tasks.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .task import Task, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class RubricCriterion:
    """A single criterion in a rubric"""
    name: str
    description: str
    weight: float
    expected: Any
    tolerance: Optional[float] = None  # For numeric comparisons
    
    def evaluate(self, actual: Any) -> Tuple[float, bool]:
        """
        Evaluate whether the actual value meets this criterion
        
        Args:
            actual: The actual value to evaluate
            
        Returns:
            Tuple of (score, passed)
        """
        # Handle different types of comparisons
        
        # Boolean comparison
        if isinstance(self.expected, bool):
            passed = actual == self.expected
            return (self.weight if passed else 0.0, passed)
        
        # Numeric comparison with tolerance
        if isinstance(self.expected, (int, float)) and isinstance(actual, (int, float)):
            if self.tolerance is not None:
                passed = abs(actual - self.expected) <= self.tolerance
            else:
                passed = actual == self.expected
            return (self.weight if passed else 0.0, passed)
        
        # String comparison
        if isinstance(self.expected, str) and isinstance(actual, str):
            passed = actual.strip().lower() == self.expected.strip().lower()
            return (self.weight if passed else 0.0, passed)
        
        # List comparison (order matters)
        if isinstance(self.expected, list) and isinstance(actual, list):
            passed = actual == self.expected
            return (self.weight if passed else 0.0, passed)
        
        # List comparison (order doesn't matter)
        if isinstance(self.expected, list) and isinstance(actual, list):
            passed = set(actual) == set(self.expected)
            return (self.weight if passed else 0.0, passed)
        
        # Dictionary comparison
        if isinstance(self.expected, dict) and isinstance(actual, dict):
            passed = actual == self.expected
            return (self.weight if passed else 0.0, passed)
        
        # Default: exact equality
        passed = actual == self.expected
        return (self.weight if passed else 0.0, passed)


@dataclass
class RubricSection:
    """A section of a rubric (e.g., outcome, process)"""
    name: str
    criteria: List[RubricCriterion]
    weight: float = 1.0
    
    def evaluate(self, actual: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the actual value against all criteria in this section
        
        Args:
            actual: The actual value to evaluate
            
        Returns:
            Tuple of (section_score, results_dict)
        """
        total_score = 0.0
        total_weight = 0.0
        results = {}
        
        for criterion in self.criteria:
            score, passed = criterion.evaluate(actual)
            total_score += score
            total_weight += criterion.weight
            
            results[criterion.name] = {
                "score": score,
                "passed": passed,
                "weight": criterion.weight,
                "expected": criterion.expected
            }
        
        # Normalize score by total weight
        if total_weight > 0:
            section_score = (total_score / total_weight) * self.weight
        else:
            section_score = 0.0
        
        return (section_score, results)


class TaskEvaluator:
    """
    Evaluates task results using rubrics
    
    The TaskEvaluator is responsible for:
    - Parsing and validating rubrics
    - Evaluating task answers against rubrics
    - Calculating scores and pass/fail status
    - Providing detailed feedback
    """
    
    def __init__(self):
        """Initialize the task evaluator"""
        pass
    
    def evaluate(
        self,
        task: Task,
        answer: Any,
        rubric: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, bool]:
        """
        Evaluate a task answer using the task's rubric
        
        Args:
            task: The task
            answer: The answer to evaluate
            rubric: Optional custom rubric (uses task.rubric if not provided)
            
        Returns:
            Tuple of (total_score, passed)
        """
        if rubric is None:
            rubric = task.rubric
        
        # Parse the rubric
        sections = self._parse_rubric(rubric)
        
        if not sections:
            # No rubric, assume passed
            return (100.0, True)
        
        # Evaluate each section
        total_score = 0.0
        total_weight = 0.0
        section_results = {}
        
        for section in sections:
            score, results = section.evaluate(answer)
            total_score += score * 100  # Convert to percentage
            total_weight += section.weight
            section_results[section.name] = results
        
        # Calculate final score
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        # Determine pass/fail
        passed = final_score >= 100.0  # Most tasks require 100% to pass
        
        return (final_score, passed)
    
    def evaluate_with_details(
        self,
        task: Task,
        answer: Any,
        rubric: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a task answer with detailed feedback
        
        Args:
            task: The task
            answer: The answer to evaluate
            rubric: Optional custom rubric
            
        Returns:
            Dictionary with detailed evaluation results
        """
        if rubric is None:
            rubric = task.rubric
        
        # Parse the rubric
        sections = self._parse_rubric(rubric)
        
        if not sections:
            return {
                "score": 100.0,
                "passed": True,
                "feedback": "No rubric provided, assuming correct",
                "sections": {}
            }
        
        # Evaluate each section
        total_score = 0.0
        total_weight = 0.0
        section_results = {}
        
        for section in sections:
            score, results = section.evaluate(answer)
            total_score += score * 100  # Convert to percentage
            total_weight += section.weight
            section_results[section.name] = results
        
        # Calculate final score
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        # Determine pass/fail
        passed = final_score >= 100.0
        
        # Generate feedback
        feedback = self._generate_feedback(task, answer, section_results, passed)
        
        return {
            "score": final_score,
            "passed": passed,
            "feedback": feedback,
            "sections": section_results,
            "total_weight": total_weight,
            "task_id": task.id
        }
    
    def _parse_rubric(self, rubric: Dict[str, Any]) -> List[RubricSection]:
        """
        Parse a rubric dictionary into RubricSection objects
        
        Args:
            rubric: Rubric dictionary
            
        Returns:
            List of RubricSection objects
        """
        sections = []
        
        for section_name, section_data in rubric.items():
            if isinstance(section_data, dict):
                # This is a section with criteria
                criteria = []
                
                for criterion_name, criterion_data in section_data.items():
                    if isinstance(criterion_data, dict):
                        # Full criterion specification
                        criterion = RubricCriterion(
                            name=criterion_name,
                            description=criterion_data.get("description", ""),
                            weight=criterion_data.get("weight", 10.0),
                            expected=criterion_data.get("expected"),
                            tolerance=criterion_data.get("tolerance")
                        )
                        criteria.append(criterion)
                    else:
                        # Simple criterion (just expected value)
                        criterion = RubricCriterion(
                            name=criterion_name,
                            description="",
                            weight=10.0,
                            expected=criterion_data
                        )
                        criteria.append(criterion)
                
                # Create section
                section = RubricSection(
                    name=section_name,
                    criteria=criteria,
                    weight=section_data.get("weight", 1.0)
                )
                sections.append(section)
        
        return sections
    
    def _generate_feedback(
        self,
        task: Task,
        answer: Any,
        section_results: Dict[str, Any],
        passed: bool
    ) -> str:
        """
        Generate feedback for a task evaluation
        
        Args:
            task: The task
            answer: The answer
            section_results: Results from section evaluations
            passed: Whether the task passed
            
        Returns:
            Feedback string
        """
        feedback_parts = []
        
        if passed:
            feedback_parts.append("✓ Task completed successfully!")
        else:
            feedback_parts.append("✗ Task did not pass. Please review the following:")
        
        # Add section feedback
        for section_name, results in section_results.items():
            section_score = sum(r["score"] for r in results.values())
            section_max = sum(r["weight"] for r in results.values())
            
            if section_max > 0:
                section_percentage = (section_score / section_max) * 100
            else:
                section_percentage = 0
            
            feedback_parts.append(f"\n{section_name.capitalize()} ({section_percentage:.0f}%):")
            
            for criterion_name, criterion_result in results.items():
                if criterion_result["passed"]:
                    feedback_parts.append(f"  ✓ {criterion_name}")
                else:
                    expected = criterion_result.get("expected", "")
                    feedback_parts.append(f"  ✗ {criterion_name} (expected: {expected})")
        
        return "\n".join(feedback_parts)
    
    def validate_rubric(self, rubric: Dict[str, Any]) -> bool:
        """
        Validate that a rubric is properly formatted
        
        Args:
            rubric: Rubric dictionary
            
        Returns:
            True if rubric is valid
        """
        try:
            sections = self._parse_rubric(rubric)
            return len(sections) > 0
        except Exception as e:
            logger.error(f"Invalid rubric: {e}")
            return False
    
    def create_rubric(
        self,
        outcome_criteria: List[Dict[str, Any]],
        process_criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a rubric from outcome and process criteria
        
        Args:
            outcome_criteria: List of outcome criteria
            process_criteria: List of process criteria
            
        Returns:
            Rubric dictionary
        """
        rubric = {
            "outcome": {},
            "process": {}
        }
        
        for criterion in outcome_criteria:
            rubric["outcome"][criterion["name"]] = criterion
        
        for criterion in process_criteria:
            rubric["process"][criterion["name"]] = criterion
        
        return rubric


# Singleton instance
_evaluator = TaskEvaluator()


def evaluate_task(
    task: Task,
    answer: Any,
    rubric: Optional[Dict[str, Any]] = None
) -> Tuple[float, bool]:
    """
    Evaluate a task answer
    
    Args:
        task: The task
        answer: The answer to evaluate
        rubric: Optional custom rubric
        
    Returns:
        Tuple of (score, passed)
    """
    return _evaluator.evaluate(task, answer, rubric)


def evaluate_task_with_details(
    task: Task,
    answer: Any,
    rubric: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate a task answer with detailed feedback
    
    Args:
        task: The task
        answer: The answer to evaluate
        rubric: Optional custom rubric
        
    Returns:
        Dictionary with detailed evaluation results
    """
    return _evaluator.evaluate_with_details(task, answer, rubric)
