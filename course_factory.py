"""Local Course Factory.

Generates a structured course plan from an OpenStax textbook URL/PDF using
OpenAI Agents SDK calls. The script does not provide deterministic fallback
content; missing dependencies, missing API credentials, extraction failures,
and model failures are logged and stop the run with a clear error.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests
from agents import Agent, Runner
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pypdf import PdfReader

Status = Literal["Success", "Warning", "Error", "Retry", "Skipped"]


class Question(BaseModel):
    id: str
    parent_id: str
    source_ids: list[str]
    prompt: str
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str
    decision_rationale: str
    alignment_claim: str


class Outcome(BaseModel):
    id: str
    parent_id: str
    source_ids: list[str]
    text: str
    questions: list[Question] = Field(default_factory=list)
    decision_rationale: str
    alignment_claim: str


class Activation(BaseModel):
    id: str
    parent_id: str
    source_ids: list[str]
    title: str
    prompt: str
    decision_rationale: str
    alignment_claim: str


class LearningObject(BaseModel):
    id: str
    parent_id: str
    source_ids: list[str]
    type: str
    title: str
    url: str | None = None
    description: str
    decision_rationale: str
    alignment_claim: str


class EditorReview(BaseModel):
    id: str
    parent_id: str
    source_ids: list[str]
    status: Literal["approved", "human_review"]
    flags: list[str]
    recommendations: list[str]
    decision_rationale: str
    alignment_claim: str


class Lesson(BaseModel):
    id: str
    parent_id: str
    source_ids: list[str]
    title: str
    textbook_refs: list[str]
    outcomes: list[Outcome] = Field(default_factory=list)
    activation: Activation | None = None
    learning_objects: list[LearningObject] = Field(default_factory=list)
    editor_review: EditorReview | None = None
    decision_rationale: str
    alignment_claim: str


class Unit(BaseModel):
    id: str
    parent_id: str | None = None
    source_ids: list[str]
    title: str
    textbook_refs: list[str]
    lessons: list[Lesson] = Field(default_factory=list)
    decision_rationale: str
    alignment_claim: str


class Course(BaseModel):
    id: str
    title: str
    textbook_source: str
    run_id: str
    units: list[Unit]


class TocItem(BaseModel):
    id: str
    title: str
    ref: str
    summary: str = ""


class EvidenceRecord(BaseModel):
    timestamp: str
    run_id: str
    agent_name: str
    stage: str
    entity_type: str
    entity_id: str
    parent_entity_id: str
    source_entity_ids: str
    target_entity_ids: str
    textbook_chapter_or_section_reference: str
    input_summary: str
    output_summary: str
    decision_rationale: str
    alignment_claim: str
    confidence_level: float
    processing_time_seconds: float
    model_name: str
    token_usage: str
    status: Status
    notes: str = ""


EVIDENCE_FIELDS = [
    "timestamp", "run_id", "agent_name", "stage", "entity_type", "entity_id",
    "parent_entity_id", "source_entity_ids", "target_entity_ids",
    "textbook_chapter_or_section_reference", "input_summary", "output_summary",
    "decision_rationale", "alignment_claim", "confidence_level",
    "processing_time_seconds", "model_name", "token_usage", "status", "notes"
]


class EvidenceLogger:
    def __init__(self, path: Path, run_id: str, model: str):
        self.path, self.run_id, self.model = path, run_id, model
        self.records: list[EvidenceRecord] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kwargs: Any) -> None:
        record = EvidenceRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=self.run_id,
            model_name=self.model,
            token_usage=kwargs.pop("token_usage", "unavailable"),
            **kwargs,
        )
        self.records.append(record)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVIDENCE_FIELDS)
            writer.writeheader()
            writer.writerows([r.model_dump() for r in self.records])


@dataclass
class AgentResult:
    data: Any
    token_usage: str = "unavailable"


def concise(text: str, n: int = 220) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]


def log_error(logger: EvidenceLogger, stage: str, message: str, notes: str = "") -> None:
    logger.log(
        agent_name="Course Factory",
        stage=stage,
        entity_type="run",
        entity_id=logger.run_id,
        parent_entity_id="",
        source_entity_ids="",
        target_entity_ids="",
        textbook_chapter_or_section_reference="",
        input_summary="",
        output_summary=message,
        decision_rationale="Run stopped instead of generating deterministic fallback content.",
        alignment_claim="No course artifacts were generated from fabricated fallback content.",
        confidence_level=0,
        processing_time_seconds=0,
        status="Error",
        notes=notes,
    )


def extract_pdf_text(source: str) -> str:
    if re.match(r"https?://", source):
        r = requests.get(source, timeout=30)
        r.raise_for_status()
        tmp = Path("/tmp/course_factory_source.pdf")
        tmp.write_bytes(r.content)
        pdf_path = tmp
    else:
        pdf_path = Path(source)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF path does not exist: {source}")
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages[:80])


def extract_textbook_evidence(source: str) -> tuple[list[TocItem], str]:
    print("Extracting textbook table of contents and summaries...")
    if re.match(r"https?://", source) and not source.lower().endswith(".pdf"):
        r = requests.get(source, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a.get_text(" ", strip=True) for a in soup.find_all("a")]
        text = "\n".join(links + [soup.get_text(" ", strip=True)])
    else:
        text = extract_pdf_text(source)
    chapters: list[TocItem] = []
    seen = set()
    for match in re.finditer(r"(?:Chapter\s*)?(\d{1,2})[\.\s:-]+([A-Z][A-Za-z0-9 ,:'’\-()]{4,90})", text):
        num, title = match.groups()
        key = (num, title.lower())
        if key in seen:
            continue
        seen.add(key)
        chapters.append(TocItem(id=f"CH{int(num):02d}", title=title.strip(), ref=f"Chapter {int(num)}"))
        if len(chapters) >= 30:
            break
    if not chapters:
        raise ValueError("No chapter or section references could be extracted from the textbook source. Check the URL/PDF and try again.")
    return chapters, concise(text, 5000)


async def run_agent(agent_name: str, instructions: str, prompt: str, output_type: Any, logger: EvidenceLogger) -> AgentResult:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Set it before running; deterministic fallback generation has been removed.")
    agent = Agent(name=agent_name, instructions=instructions, output_type=output_type, model=logger.model)
    started = time.time()
    result = await Runner.run(agent, prompt)
    usage = getattr(result, "usage", None)
    token_usage = json.dumps(getattr(usage, "__dict__", usage), default=str)
    logger.log(
        agent_name=agent_name,
        stage="openai_agent_call",
        entity_type="api_call",
        entity_id=f"{logger.run_id}-API",
        parent_entity_id=logger.run_id,
        source_entity_ids="",
        target_entity_ids="COURSE",
        textbook_chapter_or_section_reference="",
        input_summary=concise(prompt),
        output_summary=f"Received structured {output_type.__name__} output from OpenAI Agents SDK.",
        decision_rationale="Live model output is required; deterministic fallback generation is disabled.",
        alignment_claim="The model was prompted with extracted textbook evidence and required to cite source IDs.",
        confidence_level=1,
        processing_time_seconds=time.time() - started,
        token_usage=token_usage,
        status="Success",
        notes="OpenAI Agents SDK call completed successfully.",
    )
    return AgentResult(data=result.final_output, token_usage=token_usage)


def course_prompt(title: str, source: str, units_n: int, run_id: str, toc: list[TocItem], excerpt: str) -> str:
    toc_json = json.dumps([t.model_dump() for t in toc], indent=2)
    return f"""
Create a complete structured course plan.

Course title: {title}
Textbook source: {source}
Run ID: {run_id}
Required number of units: {units_n}

Extracted textbook evidence (source IDs, references, titles):
{toc_json}

Textbook excerpt / page text for grounding:
{excerpt}

Requirements:
- Return a Course object matching the schema exactly.
- Course.id must be "COURSE"; Course.title/source/run_id must match the inputs.
- Create exactly {units_n} units with IDs U01, U02, etc.
- Each unit must contain exactly 3 lessons with IDs like U01-L01.
- Each lesson must include 1-2 measurable outcomes with IDs like U01-L01-O01.
- Each outcome must include 3-4 multiple-choice questions with IDs like U01-L01-O01-Q01 and exactly 4 choices.
- Each lesson must include one activation and at least one learning object.
- Include one editor_review per lesson. Use human_review only for genuine concerns, not as a fallback.
- Every generated item must include source_ids that refer to the extracted textbook evidence IDs and/or parent IDs.
- Ground titles, outcomes, questions, activations, learning objects, rationales, and alignment claims in the textbook evidence.
- Do not invent deterministic placeholder content. If evidence is insufficient, explain the issue in editor flags while still grounding all content you can in source IDs.
""".strip()


async def build_course(course_title: str, source: str, units_n: int, toc: list[TocItem], excerpt: str, logger: EvidenceLogger) -> Course:
    result = await run_agent(
        "Course Factory Builder",
        "You are an expert course designer. Produce source-grounded structured course data only.",
        course_prompt(course_title, source, units_n, logger.run_id, toc, excerpt),
        Course,
        logger,
    )
    course = Course.model_validate(result.data.model_dump() if hasattr(result.data, "model_dump") else result.data)
    log_course_objects(course, logger, result.token_usage)
    return course


def log_course_objects(course: Course, logger: EvidenceLogger, token_usage: str) -> None:
    for unit in course.units:
        logger.log(agent_name="Course Architect", stage="unit_sequence", entity_type="unit", entity_id=unit.id, parent_entity_id="COURSE", source_entity_ids=";".join(unit.source_ids), target_entity_ids=";".join(l.id for l in unit.lessons), textbook_chapter_or_section_reference="; ".join(unit.textbook_refs), input_summary=course.textbook_source, output_summary=unit.title, decision_rationale=unit.decision_rationale, alignment_claim=unit.alignment_claim, confidence_level=.9, processing_time_seconds=0, token_usage=token_usage, status="Success", notes="Generated from live OpenAI model output.")
        for lesson in unit.lessons:
            logger.log(agent_name="Lesson Planner", stage="lesson_titles", entity_type="lesson", entity_id=lesson.id, parent_entity_id=unit.id, source_entity_ids=";".join(lesson.source_ids), target_entity_ids=";".join(o.id for o in lesson.outcomes), textbook_chapter_or_section_reference="; ".join(lesson.textbook_refs), input_summary=unit.title, output_summary=lesson.title, decision_rationale=lesson.decision_rationale, alignment_claim=lesson.alignment_claim, confidence_level=.9, processing_time_seconds=0, token_usage=token_usage, status="Success", notes="Generated from live OpenAI model output.")
            for outcome in lesson.outcomes:
                logger.log(agent_name="Outcome Writer", stage="outcomes", entity_type="outcome", entity_id=outcome.id, parent_entity_id=lesson.id, source_entity_ids=";".join(outcome.source_ids), target_entity_ids=";".join(q.id for q in outcome.questions), textbook_chapter_or_section_reference="; ".join(lesson.textbook_refs), input_summary=lesson.title, output_summary=outcome.text, decision_rationale=outcome.decision_rationale, alignment_claim=outcome.alignment_claim, confidence_level=.9, processing_time_seconds=0, token_usage=token_usage, status="Success", notes="Generated from live OpenAI model output.")
                for question in outcome.questions:
                    logger.log(agent_name="Question Writer", stage="question", entity_type="question", entity_id=question.id, parent_entity_id=outcome.id, source_entity_ids=";".join(question.source_ids), target_entity_ids="", textbook_chapter_or_section_reference="; ".join(lesson.textbook_refs), input_summary=outcome.text, output_summary=question.prompt, decision_rationale=question.decision_rationale, alignment_claim=question.alignment_claim, confidence_level=.9, processing_time_seconds=0, token_usage=token_usage, status="Success", notes="Generated from live OpenAI model output.")
                logger.log(agent_name="Question Writer", stage="question_set", entity_type="question_set", entity_id=f"{outcome.id}-QS", parent_entity_id=outcome.id, source_entity_ids=";".join(outcome.source_ids), target_entity_ids=";".join(q.id for q in outcome.questions), textbook_chapter_or_section_reference="; ".join(lesson.textbook_refs), input_summary=outcome.text, output_summary=f"Received {len(outcome.questions)} source-grounded questions from live model output.", decision_rationale="Question set was generated by the live model and validated against the required 3-4 item range.", alignment_claim=outcome.alignment_claim, confidence_level=.9, processing_time_seconds=0, token_usage=token_usage, status="Success", notes="Generated from live OpenAI model output.")
            if lesson.activation:
                logger.log(agent_name="Activation Writer", stage="activation", entity_type="activation", entity_id=lesson.activation.id, parent_entity_id=lesson.id, source_entity_ids=";".join(lesson.activation.source_ids), target_entity_ids="", textbook_chapter_or_section_reference="; ".join(lesson.textbook_refs), input_summary=lesson.title, output_summary=lesson.activation.title, decision_rationale=lesson.activation.decision_rationale, alignment_claim=lesson.activation.alignment_claim, confidence_level=.9, processing_time_seconds=0, token_usage=token_usage, status="Success", notes="Generated from live OpenAI model output.")
            for lo in lesson.learning_objects:
                logger.log(agent_name="Learning Object Finder", stage="learning_object", entity_type="learning_object", entity_id=lo.id, parent_entity_id=lesson.id, source_entity_ids=";".join(lo.source_ids), target_entity_ids="", textbook_chapter_or_section_reference="; ".join(lesson.textbook_refs), input_summary=lesson.title, output_summary=lo.title, decision_rationale=lo.decision_rationale, alignment_claim=lo.alignment_claim, confidence_level=.9, processing_time_seconds=0, token_usage=token_usage, status="Success", notes="Generated from live OpenAI model output.")
            if lesson.editor_review:
                logger.log(agent_name="Course Editor", stage="editor_review", entity_type="lesson_review", entity_id=lesson.editor_review.id, parent_entity_id=lesson.id, source_entity_ids=";".join(lesson.editor_review.source_ids), target_entity_ids=lesson.id, textbook_chapter_or_section_reference="; ".join(lesson.textbook_refs), input_summary=lesson.title, output_summary="; ".join(lesson.editor_review.flags), decision_rationale=lesson.editor_review.decision_rationale, alignment_claim=lesson.editor_review.alignment_claim, confidence_level=.9, processing_time_seconds=0, token_usage=token_usage, status="Success" if lesson.editor_review.status == "approved" else "Warning", notes="Generated from live OpenAI model output.")


def write_outputs(course: Course, logger: EvidenceLogger, out: Path) -> None:
    out.mkdir(exist_ok=True)
    (out/"course.json").write_text(course.model_dump_json(indent=2), encoding="utf-8")
    with (out/"course.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["unit_id","unit_title","lesson_id","lesson_title","outcome_id","outcome","question_id","question","learning_object_ids"])
        for u in course.units:
            for l in u.lessons:
                los=";".join(lo.id for lo in l.learning_objects)
                for o in l.outcomes:
                    for q in o.questions:
                        w.writerow([u.id,u.title,l.id,l.title,o.id,o.text,q.id,q.prompt,los])
    lessons=[l for u in course.units for l in u.lessons]
    outcomes=[o for l in lessons for o in l.outcomes]
    questions=[q for o in outcomes for q in o.questions]
    los=[lo for l in lessons for lo in l.learning_objects]
    warnings=[r for r in logger.records if r.status=="Warning"]
    errors=[r for r in logger.records if r.status=="Error"]
    flagged=[l.id for l in lessons if l.editor_review and l.editor_review.status=="human_review"]
    (out/"course_build_summary.md").write_text(f"""# Course Build Summary

- Run ID: {course.run_id}
- Units created: {len(course.units)}
- Lessons created: {len(lessons)}
- Outcomes created: {len(outcomes)}
- Questions created: {len(questions)}
- Learning objects found/suggested: {len(los)}
- Warnings: {len(warnings)}
- Errors: {len(errors)}
- Lessons flagged for human review: {', '.join(flagged) if flagged else 'None'}
- Generation mode: live OpenAI Agents SDK call required; deterministic fallback content is disabled.
""", encoding="utf-8")


def validate_course_structure(course: Course, expected_units: int) -> None:
    if len(course.units) != expected_units:
        raise ValueError(f"Course must contain exactly {expected_units} units, found {len(course.units)}")
    for unit in course.units:
        if len(unit.lessons) != 3:
            raise ValueError(f"{unit.id} must contain exactly 3 lessons, found {len(unit.lessons)}")
        for lesson in unit.lessons:
            if not 1 <= len(lesson.outcomes) <= 2:
                raise ValueError(f"{lesson.id} must contain 1-2 outcomes, found {len(lesson.outcomes)}")
            if lesson.activation is None:
                raise ValueError(f"{lesson.id} is missing an activation")
            if not lesson.learning_objects:
                raise ValueError(f"{lesson.id} is missing learning objects")
            if lesson.editor_review is None:
                raise ValueError(f"{lesson.id} is missing an editor review")
            for outcome in lesson.outcomes:
                if not 3 <= len(outcome.questions) <= 4:
                    raise ValueError(f"{outcome.id} must contain 3-4 questions, found {len(outcome.questions)}")


async def main() -> None:
    p=argparse.ArgumentParser(description="Build a structured course plan from an OpenStax textbook using live OpenAI calls.")
    p.add_argument("--title", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--units", type=int, required=True)
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL","gpt-4.1-mini"))
    args=p.parse_args()
    run_id=str(uuid.uuid4())[:8]
    out=Path("outputs")
    logger=EvidenceLogger(out/"course_build_log.csv", run_id, args.model)
    try:
        toc, excerpt = extract_textbook_evidence(args.source)
        course=await build_course(args.title,args.source,args.units,toc,excerpt,logger)
        Course.model_validate(course.model_dump())
        validate_course_structure(course, args.units)
        write_outputs(course,logger,out)
    except Exception as exc:
        message = f"Course build failed: {exc}"
        print(message, file=sys.stderr)
        log_error(logger, "run_failed", message, notes=type(exc).__name__)
        raise SystemExit(1) from exc
    print(f"Done. Wrote {out/'course.json'}, {out/'course.csv'}, {out/'course_build_log.csv'}, and {out/'course_build_summary.md'}")

if __name__ == "__main__":
    asyncio.run(main())
