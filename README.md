# Course Factory

Course Factory is a local Python prototype that creates a structured course plan from an OpenStax textbook PDF or URL. It is designed around the OpenAI Agents SDK, Pydantic models, code orchestration, and a detailed evidence trail.

## What it builds

Given a course title, textbook source, and desired number of units, the app writes:

- `outputs/course.json` — nested course data
- `outputs/course.csv` — flattened course rows for review
- `outputs/course_build_log.csv` — detailed evidence trail for every agent action
- `outputs/course_build_summary.md` — concise build summary, counts, warnings, and review flags

The generated course contains:

- Units based on textbook table-of-contents and chapter-summary evidence
- Exactly 3 lessons per unit
- 1–2 measurable lesson-level outcomes per lesson
- 3–4 multiple-choice questions per outcome
- One lesson activation per lesson
- Learning objects aligned to each lesson and outcome
- No labs
- Course Editor review records for every lesson and every outcome

## Agent workflow

The prototype uses code orchestration rather than autonomous handoffs. Each stage consumes the previous stage's structured output and logs how that input was used.

1. **Course Architect** creates the unit sequence from textbook table-of-contents and chapter-summary evidence.
2. **Lesson Planner** turns each unit into exactly 3 lesson titles.
3. **Outcome Writer** creates 1–2 measurable lesson-level learning outcomes for each lesson.
4. **Question Writer** creates a 3–4 item multiple-choice question set for each learning outcome and logs each individual question.
5. **Activation Writer** writes one compelling activation for each lesson.
6. **Learning Object Finder** locates or suggests videos, simulations, interactives, datasets, or readings aligned to the lesson and outcome.
7. **Course Editor** reviews each completed lesson object for alignment, quality, duplication, measurable outcomes, assessment alignment, activation quality, learning-object fit, and textbook grounding.

## Data models

`course_factory.py` defines Pydantic models for:

- `Unit`
- `Lesson`
- `Outcome`
- `Question`
- `Activation`
- `LearningObject`
- `EditorReview`
- `Course`
- `EvidenceRecord`

All generated objects include stable IDs such as `U01`, `U01-L01`, `U01-L01-O01`, and `U01-L01-O01-Q01`. Objects also include `parent_id`, `source_ids`, `decision_rationale`, and `alignment_claim` fields so the build can be audited.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set an OpenAI API key to enable live Agents SDK calls as the prototype evolves:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4.1-mini"
```

The current prototype also supports deterministic fallback generation. This is useful for local schema checks, logging checks, and development without an API key.

## Usage

Run against an OpenStax book URL:

```bash
python course_factory.py \
  --title "Astronomy" \
  --source "https://openstax.org/details/books/astronomy-2e" \
  --units 4
```

Run against a PDF URL or local PDF path:

```bash
python course_factory.py \
  --title "Astronomy" \
  --source "./astronomy-2e.pdf" \
  --units 4
```

Run in deterministic offline/fallback mode:

```bash
python course_factory.py \
  --title "Astronomy" \
  --source "https://openstax.org/details/books/astronomy-2e" \
  --units 4 \
  --offline
```

## Evidence trail

Every stage writes records to `outputs/course_build_log.csv` with these columns:

- Timestamp
- Run ID
- Agent Name
- Stage
- Entity Type
- Entity ID
- Parent Entity ID
- Source Entity IDs
- Target Entity IDs
- Textbook Chapter or Section Reference
- Input Summary
- Output Summary
- Decision Rationale
- Alignment Claim
- Confidence Level
- Processing Time in Seconds
- Model Name
- Token Usage, when available
- Status: Success, Warning, Error, Retry, Skipped
- Notes

The log explicitly captures trace chains such as:

- Unit → Lesson
- Lesson → Outcome
- Outcome → Question Set
- Lesson + Outcome → Activation
- Lesson + Outcome → Learning Object
- Completed Lesson Object → Editor Review

## Notes and limitations

- The script extracts table-of-contents-like chapter references from OpenStax HTML, PDF URLs, or local PDFs.
- If dependencies, API credentials, or model calls are unavailable, deterministic fallback generation preserves the required schema and evidence trail while marking records with warning statuses.
- Generated fallback questions are intentionally marked for human review because they are schema-valid placeholders rather than fully source-specific assessment items.
