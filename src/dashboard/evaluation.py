"""
RAGAS (Retrieval Augmented Generation Assessment) Evaluation Framework

Implements comprehensive evaluation metrics for RAG systems:
- Faithfulness: How factually consistent is the answer with the context?
- Answer Relevancy: How relevant is the answer to the question?
- Context Precision: Are the retrieved contexts ranked by relevance?
- Context Recall: Does the context contain all necessary information?
- Context Relevancy: How relevant are the retrieved contexts?
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None


@dataclass
class EvaluationResult:
    """Results from a single evaluation."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_relevancy: float

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            'faithfulness': 0.25,
            'answer_relevancy': 0.25,
            'context_precision': 0.20,
            'context_recall': 0.15,
            'context_relevancy': 0.15
        }
        return (
            self.faithfulness * weights['faithfulness'] +
            self.answer_relevancy * weights['answer_relevancy'] +
            self.context_precision * weights['context_precision'] +
            self.context_recall * weights['context_recall'] +
            self.context_relevancy * weights['context_relevancy']
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'faithfulness': self.faithfulness,
            'answer_relevancy': self.answer_relevancy,
            'context_precision': self.context_precision,
            'context_recall': self.context_recall,
            'context_relevancy': self.context_relevancy,
            'overall_score': self.overall_score
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    samples: list[EvaluationSample]
    results: list[EvaluationResult]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    @property
    def average_scores(self) -> EvaluationResult:
        """Calculate average scores across all samples."""
        n = len(self.results)
        if n == 0:
            return EvaluationResult(0, 0, 0, 0, 0)

        return EvaluationResult(
            faithfulness=sum(r.faithfulness for r in self.results) / n,
            answer_relevancy=sum(r.answer_relevancy for r in self.results) / n,
            context_precision=sum(r.context_precision for r in self.results) / n,
            context_recall=sum(r.context_recall for r in self.results) / n,
            context_relevancy=sum(r.context_relevancy for r in self.results) / n
        )

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'num_samples': len(self.samples),
            'average_scores': self.average_scores.to_dict(),
            'individual_results': [r.to_dict() for r in self.results]
        }


class RAGASEvaluator:
    """
    RAGAS-based evaluator for RAG systems.

    Uses LLM-as-judge approach for evaluation metrics.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(temperature=0, model=model)
        self._setup_prompts()

    def _setup_prompts(self):
        """Initialize evaluation prompts."""

        self.faithfulness_prompt = PromptTemplate(
            template="""Evaluate if the answer is factually consistent with the provided context.

Context:
{context}

Answer:
{answer}

Score the faithfulness from 0 to 1:
- 1.0: All claims in the answer are supported by the context
- 0.5: Some claims are supported, some are not verifiable
- 0.0: The answer contains claims contradicting the context

Provide your score as a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}

Evaluation:""",
            input_variables=["context", "answer"]
        )

        self.relevancy_prompt = PromptTemplate(
            template="""Evaluate how relevant the answer is to the question.

Question:
{question}

Answer:
{answer}

Score the answer relevancy from 0 to 1:
- 1.0: The answer directly and completely addresses the question
- 0.5: The answer partially addresses the question
- 0.0: The answer is irrelevant or off-topic

Provide your score as a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}

Evaluation:""",
            input_variables=["question", "answer"]
        )

        self.context_precision_prompt = PromptTemplate(
            template="""Evaluate if the most relevant contexts are ranked higher.

Question:
{question}

Contexts (in retrieval order):
{contexts}

Ground Truth (if available):
{ground_truth}

Score context precision from 0 to 1:
- 1.0: Most relevant contexts appear first
- 0.5: Relevant contexts mixed with irrelevant ones
- 0.0: Relevant contexts ranked lower than irrelevant

Provide your score as a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}

Evaluation:""",
            input_variables=["question", "contexts", "ground_truth"]
        )

        self.context_recall_prompt = PromptTemplate(
            template="""Evaluate if the retrieved context contains all information needed to answer the question.

Question:
{question}

Retrieved Context:
{context}

Ground Truth Answer (if available):
{ground_truth}

Score context recall from 0 to 1:
- 1.0: Context contains all necessary information
- 0.5: Context contains some but not all necessary information
- 0.0: Context is missing critical information

Provide your score as a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}

Evaluation:""",
            input_variables=["question", "context", "ground_truth"]
        )

        self.context_relevancy_prompt = PromptTemplate(
            template="""Evaluate how relevant each piece of retrieved context is to the question.

Question:
{question}

Contexts:
{contexts}

Score context relevancy from 0 to 1:
- 1.0: All retrieved contexts are highly relevant
- 0.5: Some contexts are relevant, some are not
- 0.0: Retrieved contexts are mostly irrelevant

Provide your score as a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}

Evaluation:""",
            input_variables=["question", "contexts"]
        )

    def _parse_score(self, response: str) -> tuple[float, str]:
        """Parse score from LLM response."""
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                score = float(data.get('score', 0.5))
                reasoning = data.get('reasoning', '')
                return min(1.0, max(0.0, score)), reasoning
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse score: {e}")

        return 0.5, "Failed to parse evaluation"

    def evaluate_faithfulness(self, sample: EvaluationSample) -> float:
        """Evaluate faithfulness of answer to context."""
        context = "\n\n".join(sample.contexts)
        prompt = self.faithfulness_prompt.format(
            context=context,
            answer=sample.answer
        )

        response = self.llm.invoke(prompt).content
        score, _ = self._parse_score(response)
        return score

    def evaluate_answer_relevancy(self, sample: EvaluationSample) -> float:
        """Evaluate relevancy of answer to question."""
        prompt = self.relevancy_prompt.format(
            question=sample.question,
            answer=sample.answer
        )

        response = self.llm.invoke(prompt).content
        score, _ = self._parse_score(response)
        return score

    def evaluate_context_precision(self, sample: EvaluationSample) -> float:
        """Evaluate if relevant contexts are ranked higher."""
        contexts_text = "\n\n".join([
            f"[Context {i+1}]: {ctx}"
            for i, ctx in enumerate(sample.contexts)
        ])

        prompt = self.context_precision_prompt.format(
            question=sample.question,
            contexts=contexts_text,
            ground_truth=sample.ground_truth or "Not available"
        )

        response = self.llm.invoke(prompt).content
        score, _ = self._parse_score(response)
        return score

    def evaluate_context_recall(self, sample: EvaluationSample) -> float:
        """Evaluate if context contains all necessary information."""
        context = "\n\n".join(sample.contexts)
        prompt = self.context_recall_prompt.format(
            question=sample.question,
            context=context,
            ground_truth=sample.ground_truth or "Not available"
        )

        response = self.llm.invoke(prompt).content
        score, _ = self._parse_score(response)
        return score

    def evaluate_context_relevancy(self, sample: EvaluationSample) -> float:
        """Evaluate relevancy of retrieved contexts."""
        contexts_text = "\n\n".join([
            f"[Context {i+1}]: {ctx}"
            for i, ctx in enumerate(sample.contexts)
        ])

        prompt = self.context_relevancy_prompt.format(
            question=sample.question,
            contexts=contexts_text
        )

        response = self.llm.invoke(prompt).content
        score, _ = self._parse_score(response)
        return score

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Run all evaluations on a single sample."""
        logger.info(f"Evaluating sample: {sample.question[:50]}...")

        return EvaluationResult(
            faithfulness=self.evaluate_faithfulness(sample),
            answer_relevancy=self.evaluate_answer_relevancy(sample),
            context_precision=self.evaluate_context_precision(sample),
            context_recall=self.evaluate_context_recall(sample),
            context_relevancy=self.evaluate_context_relevancy(sample)
        )

    def evaluate_batch(
        self,
        samples: list[EvaluationSample],
        metadata: Optional[dict] = None
    ) -> EvaluationReport:
        """Evaluate multiple samples and generate report."""
        results = []

        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}")
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate sample {i+1}: {e}")
                # Add neutral scores on failure
                results.append(EvaluationResult(0.5, 0.5, 0.5, 0.5, 0.5))

        return EvaluationReport(
            samples=samples,
            results=results,
            metadata=metadata or {}
        )

    def save_report(self, report: EvaluationReport, path: Path) -> None:
        """Save evaluation report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation report to {path}")

    def load_report(self, path: Path) -> dict:
        """Load evaluation report from JSON file."""
        with open(path) as f:
            return json.load(f)


class RAGEvaluationPipeline:
    """
    End-to-end evaluation pipeline for RAG systems.

    Integrates with LlmChain for automatic evaluation.
    """

    def __init__(self, llm_chain, evaluator: Optional[RAGASEvaluator] = None):
        self.llm_chain = llm_chain
        self.evaluator = evaluator or RAGASEvaluator()

    def create_sample_from_query(
        self,
        question: str,
        ground_truth: Optional[str] = None
    ) -> EvaluationSample:
        """Create evaluation sample by querying the RAG system."""
        # Get response and sources
        response = self.llm_chain.get_structured_response(question)

        # Extract context from sources
        contexts = [src.content for src in response.sources]

        return EvaluationSample(
            question=question,
            answer=response.answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

    def evaluate_questions(
        self,
        questions: list[str],
        ground_truths: Optional[list[str]] = None
    ) -> EvaluationReport:
        """Evaluate RAG system on a list of questions."""
        if ground_truths is None:
            ground_truths = [None] * len(questions)

        samples = []
        for q, gt in zip(questions, ground_truths):
            try:
                sample = self.create_sample_from_query(q, gt)
                samples.append(sample)
            except Exception as e:
                logger.error(f"Failed to create sample for '{q}': {e}")

        metadata = {
            'rag_config': {
                'llm_model': self.llm_chain.config.llm_model,
                'embedding_model': self.llm_chain.config.embedding_model,
                'chunk_size': self.llm_chain.config.chunk_size,
                'initial_k': self.llm_chain.config.initial_k,
                'final_k': self.llm_chain.config.final_k,
            }
        }

        return self.evaluator.evaluate_batch(samples, metadata)

    def run_evaluation_suite(
        self,
        test_file: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> EvaluationReport:
        """Run complete evaluation suite."""
        # Default test questions if no file provided
        default_questions = [
            "What are the main topics covered in this course?",
            "What are the prerequisites for this class?",
            "How is the final grade calculated?",
            "What is the late submission policy?",
            "Who is the instructor and how can I contact them?",
        ]

        questions = default_questions
        ground_truths = None

        # Load from file if provided
        if test_file and test_file.exists():
            try:
                with open(test_file) as f:
                    data = json.load(f)
                    questions = data.get('questions', default_questions)
                    ground_truths = data.get('ground_truths')
            except Exception as e:
                logger.warning(f"Failed to load test file: {e}")

        # Run evaluation
        report = self.evaluate_questions(questions, ground_truths)

        # Save report
        if output_dir:
            output_path = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.evaluator.save_report(report, output_path)

        return report


def print_evaluation_report(report: EvaluationReport) -> None:
    """Pretty print evaluation report."""
    print("\n" + "=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Samples Evaluated: {len(report.samples)}")

    avg = report.average_scores
    print("\n--- Average Scores ---")
    print(f"Faithfulness:      {avg.faithfulness:.2%}")
    print(f"Answer Relevancy:  {avg.answer_relevancy:.2%}")
    print(f"Context Precision: {avg.context_precision:.2%}")
    print(f"Context Recall:    {avg.context_recall:.2%}")
    print(f"Context Relevancy: {avg.context_relevancy:.2%}")
    print(f"\nOverall Score:     {avg.overall_score:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    from src.dashboard.llm import LlmChain

    print("Initializing RAG system...")
    llm_chain = LlmChain()

    print("Creating evaluation pipeline...")
    pipeline = RAGEvaluationPipeline(llm_chain)

    print("Running evaluation suite...")
    report = pipeline.run_evaluation_suite(
        output_dir=Path("data/evaluations")
    )

    print_evaluation_report(report)
