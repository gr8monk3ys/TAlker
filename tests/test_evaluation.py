"""
Tests for the RAGAS evaluation framework.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestEvaluationSample:
    """Tests for EvaluationSample dataclass."""

    def test_sample_creation(self):
        """Test creating an evaluation sample."""
        from src.dashboard.evaluation import EvaluationSample

        sample = EvaluationSample(
            question="What is the grading policy?",
            answer="The final exam is worth 40%.",
            contexts=["Context 1", "Context 2"],
            ground_truth="Final exam: 40%, Midterm: 30%, Assignments: 30%"
        )

        assert sample.question == "What is the grading policy?"
        assert len(sample.contexts) == 2
        assert sample.ground_truth is not None

    def test_sample_without_ground_truth(self):
        """Test sample creation without ground truth."""
        from src.dashboard.evaluation import EvaluationSample

        sample = EvaluationSample(
            question="Test question",
            answer="Test answer",
            contexts=["Context"]
        )

        assert sample.ground_truth is None


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation(self):
        """Test creating an evaluation result."""
        from src.dashboard.evaluation import EvaluationResult

        result = EvaluationResult(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
            context_relevancy=0.7
        )

        assert result.faithfulness == 0.9
        assert result.answer_relevancy == 0.85

    def test_overall_score_calculation(self):
        """Test overall score is calculated correctly."""
        from src.dashboard.evaluation import EvaluationResult

        result = EvaluationResult(
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_precision=1.0,
            context_recall=1.0,
            context_relevancy=1.0
        )

        # With all metrics at 1.0, overall should be 1.0
        assert result.overall_score == 1.0

    def test_overall_score_weighted(self):
        """Test that overall score uses weighted calculation."""
        from src.dashboard.evaluation import EvaluationResult

        result = EvaluationResult(
            faithfulness=0.8,
            answer_relevancy=0.6,
            context_precision=0.4,
            context_recall=0.2,
            context_relevancy=0.0
        )

        # Verify weights are applied
        expected = (
            0.8 * 0.25 +  # faithfulness
            0.6 * 0.25 +  # answer_relevancy
            0.4 * 0.20 +  # context_precision
            0.2 * 0.15 +  # context_recall
            0.0 * 0.15    # context_relevancy
        )

        assert abs(result.overall_score - expected) < 0.001

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.dashboard.evaluation import EvaluationResult

        result = EvaluationResult(0.9, 0.85, 0.8, 0.75, 0.7)
        result_dict = result.to_dict()

        assert 'faithfulness' in result_dict
        assert 'overall_score' in result_dict
        assert result_dict['faithfulness'] == 0.9


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def test_report_creation(self):
        """Test creating an evaluation report."""
        from src.dashboard.evaluation import (
            EvaluationReport,
            EvaluationSample,
            EvaluationResult
        )

        samples = [
            EvaluationSample("Q1", "A1", ["C1"]),
            EvaluationSample("Q2", "A2", ["C2"]),
        ]

        results = [
            EvaluationResult(0.9, 0.8, 0.7, 0.6, 0.5),
            EvaluationResult(0.8, 0.7, 0.6, 0.5, 0.4),
        ]

        report = EvaluationReport(samples=samples, results=results)

        assert len(report.samples) == 2
        assert len(report.results) == 2
        assert report.timestamp is not None

    def test_average_scores(self):
        """Test average score calculation."""
        from src.dashboard.evaluation import (
            EvaluationReport,
            EvaluationSample,
            EvaluationResult
        )

        samples = [
            EvaluationSample("Q1", "A1", ["C1"]),
            EvaluationSample("Q2", "A2", ["C2"]),
        ]

        results = [
            EvaluationResult(1.0, 1.0, 1.0, 1.0, 1.0),
            EvaluationResult(0.5, 0.5, 0.5, 0.5, 0.5),
        ]

        report = EvaluationReport(samples=samples, results=results)
        avg = report.average_scores

        assert avg.faithfulness == 0.75
        assert avg.answer_relevancy == 0.75

    def test_average_scores_empty(self):
        """Test average scores with empty results."""
        from src.dashboard.evaluation import EvaluationReport

        report = EvaluationReport(samples=[], results=[])
        avg = report.average_scores

        assert avg.faithfulness == 0
        assert avg.overall_score == 0

    def test_to_dict(self):
        """Test report conversion to dictionary."""
        from src.dashboard.evaluation import (
            EvaluationReport,
            EvaluationSample,
            EvaluationResult
        )

        samples = [EvaluationSample("Q1", "A1", ["C1"])]
        results = [EvaluationResult(0.9, 0.8, 0.7, 0.6, 0.5)]

        report = EvaluationReport(samples=samples, results=results)
        report_dict = report.to_dict()

        assert 'timestamp' in report_dict
        assert 'num_samples' in report_dict
        assert 'average_scores' in report_dict
        assert report_dict['num_samples'] == 1


class TestRAGASEvaluator:
    """Tests for RAGASEvaluator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        with patch('src.dashboard.evaluation.ChatOpenAI') as mock:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = MagicMock(
                content='{"score": 0.85, "reasoning": "Test reasoning"}'
            )
            mock.return_value = mock_instance
            yield mock_instance

    def test_evaluator_initialization(self, mock_llm):
        """Test evaluator initialization."""
        from src.dashboard.evaluation import RAGASEvaluator

        evaluator = RAGASEvaluator(model="gpt-4o-mini")

        assert evaluator.llm is not None
        assert evaluator.faithfulness_prompt is not None
        assert evaluator.relevancy_prompt is not None

    def test_parse_score_valid_json(self, mock_llm):
        """Test parsing valid JSON score response."""
        from src.dashboard.evaluation import RAGASEvaluator

        evaluator = RAGASEvaluator()

        response = '{"score": 0.85, "reasoning": "Good answer"}'
        score, reasoning = evaluator._parse_score(response)

        assert score == 0.85
        assert reasoning == "Good answer"

    def test_parse_score_invalid_json(self, mock_llm):
        """Test parsing invalid JSON falls back gracefully."""
        from src.dashboard.evaluation import RAGASEvaluator

        evaluator = RAGASEvaluator()

        response = "This is not JSON"
        score, reasoning = evaluator._parse_score(response)

        assert score == 0.5  # Default
        assert "Failed" in reasoning

    def test_parse_score_out_of_range(self, mock_llm):
        """Test score is clamped to valid range."""
        from src.dashboard.evaluation import RAGASEvaluator

        evaluator = RAGASEvaluator()

        # Score > 1
        response = '{"score": 1.5, "reasoning": "Too high"}'
        score, _ = evaluator._parse_score(response)
        assert score == 1.0

        # Score < 0
        response = '{"score": -0.5, "reasoning": "Too low"}'
        score, _ = evaluator._parse_score(response)
        assert score == 0.0

    def test_evaluate_faithfulness(self, mock_llm):
        """Test faithfulness evaluation."""
        from src.dashboard.evaluation import RAGASEvaluator, EvaluationSample

        evaluator = RAGASEvaluator()

        sample = EvaluationSample(
            question="What is ML?",
            answer="ML is machine learning.",
            contexts=["Machine learning is a type of AI."]
        )

        score = evaluator.evaluate_faithfulness(sample)

        assert 0 <= score <= 1
        mock_llm.invoke.assert_called_once()

    def test_evaluate_sample(self, mock_llm):
        """Test full sample evaluation."""
        from src.dashboard.evaluation import RAGASEvaluator, EvaluationSample

        evaluator = RAGASEvaluator()

        sample = EvaluationSample(
            question="What is ML?",
            answer="ML is machine learning.",
            contexts=["Machine learning is a type of AI."]
        )

        result = evaluator.evaluate_sample(sample)

        assert result.faithfulness == 0.85
        assert result.answer_relevancy == 0.85
        # Called for each metric
        assert mock_llm.invoke.call_count == 5

    def test_evaluate_batch(self, mock_llm):
        """Test batch evaluation."""
        from src.dashboard.evaluation import RAGASEvaluator, EvaluationSample

        evaluator = RAGASEvaluator()

        samples = [
            EvaluationSample("Q1", "A1", ["C1"]),
            EvaluationSample("Q2", "A2", ["C2"]),
        ]

        report = evaluator.evaluate_batch(samples, {"test": True})

        assert len(report.results) == 2
        assert report.metadata == {"test": True}


class TestEvaluationPersistence:
    """Tests for saving and loading evaluation reports."""

    def test_save_report(self, tmp_path):
        """Test saving evaluation report."""
        from src.dashboard.evaluation import (
            RAGASEvaluator,
            EvaluationReport,
            EvaluationSample,
            EvaluationResult
        )

        with patch('src.dashboard.evaluation.ChatOpenAI'):
            evaluator = RAGASEvaluator()

            samples = [EvaluationSample("Q1", "A1", ["C1"])]
            results = [EvaluationResult(0.9, 0.8, 0.7, 0.6, 0.5)]
            report = EvaluationReport(samples=samples, results=results)

            output_path = tmp_path / "test_report.json"
            evaluator.save_report(report, output_path)

            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                data = json.load(f)
                assert 'timestamp' in data
                assert 'average_scores' in data

    def test_load_report(self, tmp_path):
        """Test loading evaluation report."""
        from src.dashboard.evaluation import RAGASEvaluator

        # Create a test report file
        report_data = {
            'timestamp': '2024-01-01T00:00:00',
            'num_samples': 1,
            'average_scores': {
                'faithfulness': 0.9,
                'answer_relevancy': 0.8
            }
        }

        report_path = tmp_path / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f)

        with patch('src.dashboard.evaluation.ChatOpenAI'):
            evaluator = RAGASEvaluator()
            loaded = evaluator.load_report(report_path)

            assert loaded['num_samples'] == 1
            assert loaded['average_scores']['faithfulness'] == 0.9


class TestRAGEvaluationPipeline:
    """Tests for the evaluation pipeline."""

    @pytest.fixture
    def mock_llm_chain(self):
        """Create a mock LLM chain."""
        from src.dashboard.llm import RAGConfig, RAGResponse, RetrievalResult

        mock = MagicMock()
        mock.config = RAGConfig()
        mock.get_structured_response.return_value = RAGResponse(
            answer="Test answer",
            sources=[
                RetrievalResult("Context 1", "file1.pdf", 1, 0.9, "id1"),
                RetrievalResult("Context 2", "file2.pdf", 2, 0.8, "id2"),
            ],
            confidence=0.85,
            tokens_used=100
        )
        return mock

    def test_pipeline_initialization(self, mock_llm_chain):
        """Test pipeline initialization."""
        from src.dashboard.evaluation import RAGEvaluationPipeline

        with patch('src.dashboard.evaluation.ChatOpenAI'):
            pipeline = RAGEvaluationPipeline(mock_llm_chain)

            assert pipeline.llm_chain is not None
            assert pipeline.evaluator is not None

    def test_create_sample_from_query(self, mock_llm_chain):
        """Test creating evaluation sample from query."""
        from src.dashboard.evaluation import RAGEvaluationPipeline

        with patch('src.dashboard.evaluation.ChatOpenAI'):
            pipeline = RAGEvaluationPipeline(mock_llm_chain)

            sample = pipeline.create_sample_from_query(
                "What is the grading policy?",
                "40% final, 30% midterm, 30% assignments"
            )

            assert sample.question == "What is the grading policy?"
            assert sample.answer == "Test answer"
            assert len(sample.contexts) == 2
            assert sample.ground_truth is not None

    def test_evaluate_questions(self, mock_llm_chain):
        """Test evaluating a list of questions."""
        from src.dashboard.evaluation import RAGEvaluationPipeline

        with patch('src.dashboard.evaluation.ChatOpenAI') as mock_chat:
            mock_chat.return_value.invoke.return_value = MagicMock(
                content='{"score": 0.8, "reasoning": "Good"}'
            )

            pipeline = RAGEvaluationPipeline(mock_llm_chain)

            questions = ["Q1?", "Q2?"]
            report = pipeline.evaluate_questions(questions)

            assert len(report.samples) == 2
            assert len(report.results) == 2
            assert 'rag_config' in report.metadata
