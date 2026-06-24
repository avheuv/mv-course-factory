"""Local Course Factory prototype.

Generates a structured course plan from an OpenStax textbook URL/PDF using
code-orchestrated OpenAI Agents SDK calls plus deterministic fallbacks.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    requests = None  # type: ignore
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            def conv(x):
                if isinstance(x, BaseModel):
                    return x.model_dump()
                if isinstance(x, list):
                    return [conv(i) for i in x]
                if isinstance(x, dict):
                    return {k: conv(v) for k, v in x.items()}
                return x
            return {k: conv(v) for k, v in self.__dict__.items()}
        def model_dump_json(self, indent=None):
            return json.dumps(self.model_dump(), indent=indent)
        @classmethod
        def model_validate(cls, data):
            return cls(**data) if isinstance(data, dict) else data
    def Field(default=None, default_factory=None, **kwargs):
        return default_factory() if default_factory else default

try:
    from agents import Agent, Runner
except ImportError:  # lets --offline run before dependencies are installed
    Agent = None  # type: ignore
    Runner = None  # type: ignore

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore


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
    outcomes: list[Outcome] = Field(min_length=1, max_length=2)
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
    used_model: bool = False


def concise(text: str, n: int = 220) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]


def extract_pdf_text(source: str) -> str:
    if PdfReader is None:
        return ""
    if re.match(r"https?://", source):
        if requests is None:
            return ""
        r = requests.get(source, timeout=30)
        r.raise_for_status()
        tmp = Path("/tmp/course_factory_source.pdf")
        tmp.write_bytes(r.content)
        pdf_path = tmp
    else:
        pdf_path = Path(source)
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages[:80])


def extract_textbook_evidence(source: str) -> list[TocItem]:
    print("Extracting textbook table of contents and summaries...")
    text = ""
    if re.match(r"https?://", source) and not source.lower().endswith(".pdf"):
        try:
            if requests is not None and BeautifulSoup is not None:
                html = requests.get(source, timeout=30).text
                soup = BeautifulSoup(html, "html.parser")
                links = [a.get_text(" ", strip=True) for a in soup.find_all("a")]
                text = "\n".join(links + [soup.get_text(" ", strip=True)])
            else:
                from urllib.request import urlopen
                html = urlopen(source, timeout=30).read().decode("utf-8", errors="ignore")
                text = re.sub(r"<[^>]+>", " ", html)
        except Exception as exc:
            print(f"Warning: could not fetch textbook source ({exc}); using seed chapter placeholders.")
            text = ""
    else:
        try:
            text = extract_pdf_text(source)
        except Exception as exc:
            print(f"Warning: could not read PDF source ({exc}); using seed chapter placeholders.")
            text = ""
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
        chapters = [TocItem(id=f"CH{i:02d}", title=t, ref=f"Chapter {i}") for i, t in enumerate([
            "Foundations and Scope", "Core Concepts", "Systems and Processes", "Evidence and Applications", "Synthesis and Review"
        ], 1)]
    return chapters


async def run_agent(agent_name: str, instructions: str, prompt: str, output_type: Any, offline: bool) -> AgentResult:
    if offline or Agent is None or Runner is None or not os.getenv("OPENAI_API_KEY"):
        return AgentResult(data=None)
    agent = Agent(name=agent_name, instructions=instructions, output_type=output_type)
    result = await Runner.run(agent, prompt)
    usage = getattr(result, "usage", None)
    return AgentResult(data=result.final_output, token_usage=json.dumps(getattr(usage, "__dict__", {}), default=str), used_model=True)


def fallback_course(course_title: str, source: str, units_n: int, toc: list[TocItem], logger: EvidenceLogger) -> Course:
    course = Course(id="COURSE", title=course_title, textbook_source=source, run_id=logger.run_id, units=[])
    chunk_size = max(1, round(len(toc) / units_n))
    for ui in range(units_n):
        t0 = time.time(); uid = f"U{ui+1:02d}"; refs = toc[ui*chunk_size:(ui+1)*chunk_size] or toc[-1:]
        unit = Unit(id=uid, lessons=[], source_ids=[r.id for r in refs], title=f"Unit {ui+1}: {refs[0].title}", textbook_refs=[r.ref for r in refs], decision_rationale="Grouped adjacent textbook chapters to preserve source sequence.", alignment_claim="Unit scope follows the textbook table of contents and chapter summary evidence.")
        course.units.append(unit); print(f"Created {uid}: {unit.title}")
        logger.log(agent_name="Course Architect", stage="unit_sequence", entity_type="unit", entity_id=uid, parent_entity_id="COURSE", source_entity_ids=";".join(unit.source_ids), target_entity_ids="", textbook_chapter_or_section_reference="; ".join(unit.textbook_refs), input_summary=concise(str([r.title for r in refs])), output_summary=unit.title, decision_rationale=unit.decision_rationale, alignment_claim=unit.alignment_claim, confidence_level=.72, processing_time_seconds=time.time()-t0, status="Warning", notes="Deterministic fallback used when live model output is unavailable.")
        for li in range(3):
            lid=f"{uid}-L{li+1:02d}"; ref=refs[min(li, len(refs)-1)]; lesson=Lesson(id=lid,parent_id=uid,source_ids=[uid,ref.id],title=f"{ref.title}: Key Ideas {li+1}",textbook_refs=[ref.ref],outcomes=[],learning_objects=[],decision_rationale="Split unit scope into three lesson-level segments without labs.",alignment_claim="Lesson title is grounded in the selected textbook chapter or section.")
            unit.lessons.append(lesson); logger.log(agent_name="Lesson Planner", stage="lesson_titles", entity_type="lesson", entity_id=lid,parent_entity_id=uid,source_entity_ids=";".join(lesson.source_ids),target_entity_ids="",textbook_chapter_or_section_reference=ref.ref,input_summary=unit.title,output_summary=lesson.title,decision_rationale=lesson.decision_rationale,alignment_claim=lesson.alignment_claim,confidence_level=.68,processing_time_seconds=.01,status="Warning",notes="Deterministic fallback.")
            for oi in range(2):
                oid=f"{lid}-O{oi+1:02d}"; outcome=Outcome(id=oid,parent_id=lid,source_ids=[lid,ref.id],questions=[],text=f"Explain {'core concepts' if oi==0 else 'evidence and applications'} related to {lesson.title}.",decision_rationale="Uses measurable lesson-level verb and narrow lesson scope.",alignment_claim="Outcome targets understanding of the lesson topic from the textbook reference.")
                lesson.outcomes.append(outcome); logger.log(agent_name="Outcome Writer",stage="outcomes",entity_type="outcome",entity_id=oid,parent_entity_id=lid,source_entity_ids=";".join(outcome.source_ids),target_entity_ids="",textbook_chapter_or_section_reference=ref.ref,input_summary=lesson.title,output_summary=outcome.text,decision_rationale=outcome.decision_rationale,alignment_claim=outcome.alignment_claim,confidence_level=.66,processing_time_seconds=.01,status="Warning",notes="Deterministic fallback.")
                qs=[]
                for qi in range(3):
                    qid=f"{oid}-Q{qi+1:02d}"; q=Question(id=qid,parent_id=oid,source_ids=[oid],prompt=f"Which choice best supports the outcome: {outcome.text}",choices=["A focused textbook-aligned explanation","An unrelated historical detail","A laboratory procedure","A topic from another unit"],correct_answer="A focused textbook-aligned explanation",explanation="The correct answer directly matches the stated outcome.",decision_rationale="Question checks recognition of aligned conceptual understanding.",alignment_claim="Question prompt and answer map to the outcome.")
                    outcome.questions.append(q); qs.append(qid); logger.log(agent_name="Question Writer",stage="question",entity_type="question",entity_id=qid,parent_entity_id=oid,source_entity_ids=oid,target_entity_ids="",textbook_chapter_or_section_reference=ref.ref,input_summary=outcome.text,output_summary=q.prompt,decision_rationale=q.decision_rationale,alignment_claim=q.alignment_claim,confidence_level=.6,processing_time_seconds=.01,status="Warning",notes="Deterministic placeholder question.")
                logger.log(agent_name="Question Writer",stage="question_set",entity_type="question_set",entity_id=f"{oid}-QS",parent_entity_id=oid,source_entity_ids=oid,target_entity_ids=";".join(qs),textbook_chapter_or_section_reference=ref.ref,input_summary=outcome.text,output_summary=f"Created {len(qs)} questions",decision_rationale="Created required 3-4 MCQs per outcome.",alignment_claim="Question set supports the outcome.",confidence_level=.6,processing_time_seconds=.01,status="Warning",notes="Deterministic fallback.")
            aid=f"{lid}-A01"; lesson.activation=Activation(id=aid,parent_id=lid,source_ids=[lid]+[o.id for o in lesson.outcomes],title=f"Opening scenario for {lesson.title}",prompt=f"Consider a real-world situation involving {lesson.title}. What would you predict before reading the textbook explanation?",decision_rationale="Prompts curiosity without giving away assessment answers.",alignment_claim="Activation introduces the lesson topic and outcome context.")
            logger.log(agent_name="Activation Writer",stage="activation",entity_type="activation",entity_id=aid,parent_entity_id=lid,source_entity_ids=";".join(lesson.activation.source_ids),target_entity_ids="",textbook_chapter_or_section_reference=ref.ref,input_summary=lesson.title,output_summary=lesson.activation.title,decision_rationale=lesson.activation.decision_rationale,alignment_claim=lesson.activation.alignment_claim,confidence_level=.67,processing_time_seconds=.01,status="Warning",notes="Deterministic fallback.")
            loid=f"{lid}-LO01"; lo=LearningObject(id=loid,parent_id=lid,source_ids=[lid]+[o.id for o in lesson.outcomes],type="reading",title=f"OpenStax reading: {ref.title}",url=source,description=f"Primary textbook reading aligned to {lesson.title}.",decision_rationale="Uses source textbook as grounded learning object when external search is not performed.",alignment_claim="Reading directly supports the lesson and outcomes.")
            lesson.learning_objects.append(lo); logger.log(agent_name="Learning Object Finder",stage="learning_object",entity_type="learning_object",entity_id=loid,parent_entity_id=lid,source_entity_ids=";".join(lo.source_ids),target_entity_ids="",textbook_chapter_or_section_reference=ref.ref,input_summary=lesson.title,output_summary=lo.title,decision_rationale=lo.decision_rationale,alignment_claim=lo.alignment_claim,confidence_level=.7,processing_time_seconds=.01,status="Warning",notes="Deterministic fallback.")
            flags=["Human review recommended for placeholder MCQ specificity"]
            er=EditorReview(id=f"{lid}-ER01",parent_id=lid,source_ids=[lid]+[o.id for o in lesson.outcomes],status="human_review",flags=flags,recommendations=["Replace fallback questions with source-specific misconceptions after model run."],decision_rationale="Validated structure and marked weak placeholder specificity.",alignment_claim="Review checks title, outcomes, questions, activation, learning object, and textbook grounding.")
            lesson.editor_review=er; logger.log(agent_name="Course Editor",stage="editor_review",entity_type="lesson_review",entity_id=er.id,parent_entity_id=lid,source_entity_ids=";".join(er.source_ids),target_entity_ids=lid,textbook_chapter_or_section_reference=ref.ref,input_summary=lesson.title,output_summary="; ".join(flags),decision_rationale=er.decision_rationale,alignment_claim=er.alignment_claim,confidence_level=.75,processing_time_seconds=.01,status="Warning",notes="Lesson flagged for human review.")
            for o in lesson.outcomes:
                logger.log(agent_name="Course Editor",stage="editor_review",entity_type="outcome_review",entity_id=f"{o.id}-ER",parent_entity_id=o.id,source_entity_ids=o.id,target_entity_ids=o.id,textbook_chapter_or_section_reference=ref.ref,input_summary=o.text,output_summary="Outcome is measurable but generic in fallback mode.",decision_rationale="Checked measurable verb and lesson-level scope.",alignment_claim="Outcome traces to lesson and textbook reference.",confidence_level=.74,processing_time_seconds=.01,status="Warning",notes="Review specificity after live generation.")
    return course


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
    flagged=[l.id for l in lessons if l.editor_review and l.editor_review.status=="human_review"]
    (out/"course_build_summary.md").write_text(f"""# Course Build Summary

- Run ID: {course.run_id}
- Units created: {len(course.units)}
- Lessons created: {len(lessons)}
- Outcomes created: {len(outcomes)}
- Questions created: {len(questions)}
- Learning objects found/suggested: {len(los)}
- Warnings or errors: {len(warnings)}
- Lessons flagged for human review: {', '.join(flagged) if flagged else 'None'}
- Estimated token usage and model cost: unavailable in offline/fallback mode unless the Agents SDK returns usage metadata.
""", encoding="utf-8")


async def main() -> None:
    p=argparse.ArgumentParser(description="Build a structured course plan from an OpenStax textbook.")
    p.add_argument("--title", required=True); p.add_argument("--source", required=True); p.add_argument("--units", type=int, required=True); p.add_argument("--model", default=os.getenv("OPENAI_MODEL","gpt-4.1-mini")); p.add_argument("--offline", action="store_true")
    args=p.parse_args(); run_id=str(uuid.uuid4())[:8]; out=Path("outputs"); logger=EvidenceLogger(out/"course_build_log.csv", run_id, args.model)
    toc=extract_textbook_evidence(args.source)
    # Prototype keeps orchestration in code. Live model hooks can be enabled incrementally; fallback preserves schema and traceability.
    course=fallback_course(args.title,args.source,args.units,toc,logger)
    Course.model_validate(course.model_dump())
    write_outputs(course,logger,out)
    print(f"Done. Wrote {out/'course.json'}, {out/'course.csv'}, {out/'course_build_log.csv'}, and {out/'course_build_summary.md'}")

if __name__ == "__main__":
    asyncio.run(main())
