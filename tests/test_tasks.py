"""
Tests for Task Modules
"""

import pytest
from unittest.mock import patch, MagicMock
from drug_discovery_mcp.tasks import (
    Task,
    TaskCategory,
    TaskDifficulty,
    TaskStatus,
    TaskResult,
    TaskRunner,
    TaskRegistry,
    TaskEvaluator,
    create_example_tasks
)


class TestTask:
    """Tests for Task class"""
    
    def test_task_creation(self):
        """Test creating a task"""
        task = Task(
            id="test_001",
            name="Test Task",
            category=TaskCategory.CHEMINFORMATICS,
            prompt="Calculate descriptors for CC(=O)O"
        )
        
        assert task.id == "test_001"
        assert task.name == "Test Task"
        assert task.category == TaskCategory.CHEMINFORMATICS
        assert task.prompt == "Calculate descriptors for CC(=O)O"
    
    def test_task_to_dict(self):
        """Test converting task to dictionary"""
        task = Task(
            id="test_001",
            name="Test Task",
            category=TaskCategory.CHEMINFORMATICS,
            prompt="Calculate descriptors for CC(=O)O"
        )
        
        task_dict = task.to_dict()
        
        assert task_dict is not None
        assert task_dict["id"] == "test_001"
        assert task_dict["name"] == "Test Task"
        assert task_dict["category"] == "cheminformatics"
    
    def test_task_from_dict(self):
        """Test creating task from dictionary"""
        task_data = {
            "id": "test_001",
            "name": "Test Task",
            "category": "cheminformatics",
            "prompt": "Calculate descriptors for CC(=O)O"
        }
        
        task = Task.from_dict(task_data)
        
        assert task is not None
        assert task.id == "test_001"
        assert task.name == "Test Task"
        assert task.category == TaskCategory.CHEMINFORMATICS
    
    def test_task_validate(self):
        """Test task validation"""
        # Valid task
        task1 = Task(
            id="test_001",
            name="Test Task",
            category=TaskCategory.CHEMINFORMATICS,
            prompt="Calculate descriptors for CC(=O)O"
        )
        assert task1.validate() is True
        
        # Invalid task (missing required fields)
        task2 = Task(id="", name="", category=TaskCategory.CHEMINFORMATICS, prompt="")
        assert task2.validate() is False


class TestTaskResult:
    """Tests for TaskResult"""
    
    def test_result_creation(self):
        """Test creating a task result"""
        result = TaskResult(
            task_id="test_001",
            status=TaskStatus.COMPLETED,
            answer={"result": "success"},
            score=100.0,
            passed=True,
            execution_time=10.0
        )
        
        assert result.task_id == "test_001"
        assert result.status == TaskStatus.COMPLETED
        assert result.score == 100.0
        assert result.passed is True
    
    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = TaskResult(
            task_id="test_001",
            status=TaskStatus.COMPLETED,
            answer={"result": "success"},
            score=100.0,
            passed=True
        )
        
        result_dict = result.to_dict()
        
        assert result_dict is not None
        assert result_dict["task_id"] == "test_001"
        assert result_dict["status"] == "completed"
        assert result_dict["score"] == 100.0
        assert result_dict["passed"] is True


class TestTaskRunner:
    """Tests for TaskRunner"""
    
    @pytest.fixture
    def runner(self):
        return TaskRunner()
    
    def test_initialization(self, runner):
        """Test runner initialization"""
        assert runner is not None
        assert hasattr(runner, 'registry')
        assert hasattr(runner, 'evaluator')
    
    def test_list_tasks(self, runner):
        """Test listing tasks"""
        tasks = runner.list_tasks()
        
        assert tasks is not None
        assert isinstance(tasks, list)
        assert len(tasks) > 0  # Should have example tasks
    
    def test_get_task(self, runner):
        """Test getting a specific task"""
        tasks = runner.list_tasks()
        if tasks:
            task_id = tasks[0]["id"]
            task = runner.get_task(task_id)
            
            assert task is not None
            assert task.id == task_id
    
    def test_run_task(self, runner):
        """Test running a task"""
        # Use a simple task
        task_data = {
            "id": "chem_001",
            "name": "Calculate Molecular Descriptors",
            "category": "cheminformatics",
            "prompt": "Calculate descriptors for CC(=O)O",
            "answer_format": "json",
            "expected_answer": {
                "molecular_weight": 60.05,
                "logp": -0.65,
                "hba": 2,
                "hbd": 1,
                "tpsa": 37.3,
                "rotatable_bonds": 0,
                "aromatic_rings": 0
            }
        }
        
        # Register the task
        runner.registry.register_task(Task.from_dict(task_data))
        
        # Run the task
        result = runner.run_task("chem_001")
        
        assert result is not None
        assert result.task_id == "chem_001"
        # The task might not complete successfully if RDKit is not available
        # but it should still return a result


class TestTaskRegistry:
    """Tests for TaskRegistry"""
    
    @pytest.fixture
    def registry(self):
        return TaskRegistry()
    
    def test_initialization(self, registry):
        """Test registry initialization"""
        assert registry is not None
        assert len(registry.get_tasks()) > 0  # Should have example tasks
    
    def test_register_task(self, registry):
        """Test registering a task"""
        task = Task(
            id="new_task_001",
            name="New Task",
            category=TaskCategory.CHEMINFORMATICS,
            prompt="Test prompt"
        )
        
        initial_count = len(registry.get_tasks())
        registry.register_task(task)
        new_count = len(registry.get_tasks())
        
        assert new_count == initial_count + 1
        assert registry.get_task("new_task_001") is not None
    
    def test_unregister_task(self, registry):
        """Test unregistering a task"""
        # First register a task
        task = Task(
            id="temp_task_001",
            name="Temp Task",
            category=TaskCategory.CHEMINFORMATICS,
            prompt="Test prompt"
        )
        registry.register_task(task)
        
        # Unregister it
        result = registry.unregister_task("temp_task_001")
        
        assert result is True
        assert registry.get_task("temp_task_001") is None
    
    def test_get_tasks_by_category(self, registry):
        """Test getting tasks by category"""
        tasks = registry.get_tasks_by_category(TaskCategory.CHEMINFORMATICS)
        
        assert tasks is not None
        assert isinstance(tasks, list)
        # Should have at least the example cheminformatics task
        assert len(tasks) >= 1
    
    def test_search_tasks(self, registry):
        """Test searching tasks"""
        results = registry.search_tasks("descriptors")
        
        assert results is not None
        assert isinstance(results, list)
        # Should find the descriptor calculation task
        assert len(results) >= 1
    
    def test_get_statistics(self, registry):
        """Test getting registry statistics"""
        stats = registry.get_statistics()
        
        assert stats is not None
        assert "total_tasks" in stats
        assert "categories" in stats
        assert "difficulties" in stats


class TestTaskEvaluator:
    """Tests for TaskEvaluator"""
    
    @pytest.fixture
    def evaluator(self):
        return TaskEvaluator()
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator is not None
    
    def test_evaluate(self, evaluator):
        """Test evaluating a task answer"""
        # Create a simple task with rubric
        task = Task(
            id="test_eval_001",
            name="Test Evaluation",
            category=TaskCategory.CHEMINFORMATICS,
            prompt="Test prompt",
            expected_answer=42,
            rubric={
                "outcome": {
                    "correct_answer": {"weight": 30, "expected": 42}
                }
            }
        )
        
        # Evaluate correct answer
        score, passed = evaluator.evaluate(task, 42)
        
        assert score > 0
        assert passed is True
        
        # Evaluate incorrect answer
        score, passed = evaluator.evaluate(task, 43)
        
        assert score == 0
        assert passed is False
    
    def test_evaluate_with_details(self, evaluator):
        """Test evaluating with detailed feedback"""
        task = Task(
            id="test_eval_001",
            name="Test Evaluation",
            category=TaskCategory.CHEMINFORMATICS,
            prompt="Test prompt",
            expected_answer=42,
            rubric={
                "outcome": {
                    "correct_answer": {"weight": 30, "expected": 42}
                }
            }
        )
        
        result = evaluator.evaluate_with_details(task, 42)
        
        assert result is not None
        assert "score" in result
        assert "passed" in result
        assert "feedback" in result


class TestExampleTasks:
    """Tests for example tasks"""
    
    def test_create_example_tasks(self):
        """Test creating example tasks"""
        tasks = create_example_tasks()
        
        assert tasks is not None
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        
        # Check that we have tasks from different categories
        categories = {task.category for task in tasks}
        assert len(categories) > 1
    
    def test_example_task_properties(self):
        """Test that example tasks have proper properties"""
        tasks = create_example_tasks()
        
        for task in tasks:
            assert task.id is not None
            assert task.name is not None
            assert task.category is not None
            assert task.prompt is not None
            assert task.validate() is True
