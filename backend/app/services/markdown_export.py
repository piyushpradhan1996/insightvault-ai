from app.schemas import InsightResponse


def render_document_summary_markdown(title: str, insights: InsightResponse) -> str:
    lines = [
        f"# {title}",
        "",
        "## Summary",
        insights.summary,
        "",
        "## Action Items",
    ]
    for item in insights.action_items:
        lines.append(f"- {item.task} (Owner: {item.owner}; Deadline: {item.deadline})")
    lines.extend(["", "## Risks"])
    lines.extend([f"- {risk}" for risk in insights.risks] or ["- None identified."])
    lines.extend(["", "## Decisions"])
    lines.extend([f"- {decision}" for decision in insights.decisions] or ["- None identified."])
    lines.extend(["", "## Follow-up Questions"])
    lines.extend([f"- {question}" for question in insights.follow_up_questions])
    lines.append("")
    return "\n".join(lines)

