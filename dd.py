"""M&A due diligence mode: one focused AI review per risk, then a structured Board report."""

import io
import re
from collections import Counter
from datetime import datetime

import streamlit as st

import agent
from ingest import ENTITIES

CASE_CONTEXT = (
    "We act for the purchaser (our client) in the proposed acquisition of Canvassian Pty Ltd, a company that "
    "sells cybersecurity software. Our client is concerned about liabilities or risks that have not been "
    "accounted for and that could mean Canvassian is worth less than the price being considered. The evidence is "
    "a library of emails, contracts and board papers; it is the only source of facts. The Board needs an opinion "
    "this afternoon, so be efficient and precise. Absence of evidence must be reported as 'not found', never "
    "assumed to mean there is no risk. Do not guess."
)

ANSWER_FORMAT = (
    "\n\nWhen you call finish, `answer` must use exactly this format:\n"
    "RATING: <High | Medium | Low | Insufficient evidence>\n"
    "SUMMARY: <2-3 sentences>\n"
    "FINDINGS:\n- <each finding, with [S#] citations; quote key contract wording and give clause numbers where you can>\n"
    "IMPLICATION FOR THE DEAL: <price, conditions or protections to consider>\n"
    "GAPS: <what you could not find or verify>"
)

CLIENTS = [
    ("paywise", "PayWise"),
    ("alphabear", "Alphabear"),
    ("bravocat", "Bravocat"),
    ("charlemont", "Charlemont"),
    ("deltaforce", "Deltaforce"),
    ("echona", "Echona"),
]


def _coc_objective(key, name):
    return (
        f"Change-of-control review for the {name} contract(s) with Canvassian. Find them (try doc_type contract with "
        f"entity {key}; if nothing is found, loosen the filters) and read the termination, assignment, change of "
        f"control, consent, exclusivity, notice, term and renewal provisions. Report whether Canvassian being acquired "
        f"gives {name} any right to terminate, renegotiate, withhold consent or change pricing, citing clause numbers "
        f"and the key wording. Note the contract term and renewal or expiry dates. If the contract cannot be found, say so."
    )


PASSES = [
    {
        "id": "founder",
        "group": "critical",
        "title": "Critical 1: Founder dependency (Jane Wu)",
        "objective": (
            "Assess the risk that Jane Wu's continued involvement and motivation cannot be relied on after the "
            "acquisition. Look for signs she may leave or reduce her role, disagreements with the board or "
            "co-founders, burnout or morale, her employment, retention, non-compete or earn-out terms, "
            "key-person clauses in client contracts, and heavy reliance on her for client relationships or "
            "technology. Try entity jane_wu and topic key_person."
        ),
    },
    {
        "id": "paywise_health",
        "group": "critical",
        "title": "Critical 2: PayWise financial health",
        "objective": (
            "Verify the rumours that PayWise Pty Ltd is in financial difficulty. Look for late, disputed or "
            "partial payments, payment plans, requests to extend terms, insolvency or administration talk, "
            "covenant or funding problems, reduced usage, renewal doubts, and board or management commentary. "
            "PayWise is about 20% of Canvassian's revenue, so quantify the exposure where the documents allow "
            "(amounts, overdue invoices, contract value). Try entity paywise and topic financial."
        ),
    },
] + [
    {
        "id": f"coc_{key}",
        "group": "coc",
        "title": f"Critical 3: Change of control, {name}",
        "objective": _coc_objective(key, name),
    }
    for key, name in CLIENTS
] + [
    {
        "id": "other_legal",
        "group": "other",
        "title": "Other risk: Legal disputes and regulatory",
        "objective": (
            "Find other legal and regulatory risks: litigation or threatened claims, disputes with customers or "
            "suppliers, regulatory investigations, breaches of contract, indemnities, penalties, SLA failures and "
            "service credits. Try topic legal."
        ),
    },
    {
        "id": "other_security",
        "group": "other",
        "title": "Other risk: Security incidents and certifications",
        "objective": (
            "Canvassian sells cybersecurity software, so its own security record matters to reputation and "
            "liability. Find data breaches, security incidents, vulnerabilities, customer notifications, "
            "regulator contact, insurance, and the status of certifications (for example ISO 27001 or SOC 2). "
            "Try topic security."
        ),
    },
    {
        "id": "other_ip",
        "group": "other",
        "title": "Other risk: Intellectual property and licensing",
        "objective": (
            "Find IP and licensing risks: open-source licence obligations, third-party code, IP ownership and "
            "assignment from employees or contractors, source code escrow arrangements, and infringement "
            "claims. Try topic ip."
        ),
    },
    {
        "id": "other_financial",
        "group": "other",
        "title": "Other risk: Financial and commercial health",
        "objective": (
            "Look beyond PayWise at Canvassian's commercial health: revenue concentration, churn or cancellations, "
            "pricing disputes, unpaid invoices, forecasts versus actuals, cash runway, debts, undisclosed "
            "commitments and tax issues. Board papers are usually the best source (try doc_type board_paper)."
        ),
    },
    {
        "id": "other_people",
        "group": "other",
        "title": "Other risk: People, governance and operations",
        "objective": (
            "Find people and governance risks beyond Jane Wu: departures of other key staff, employee disputes, "
            "unpaid entitlements, reliance on contractors, board conflict, related-party dealings, and "
            "governance weaknesses."
        ),
    },
    {
        "id": "other_sweep",
        "group": "other",
        "title": "Other risk: Open-ended sweep",
        "objective": (
            "Do an open-ended sweep for anything else a purchaser's expert deal-maker would want to know: "
            "surprises, undisclosed liabilities, exclusivity or most-favoured-customer terms, unusual discounts, "
            "side letters, unresolved board actions, and urgent deadlines. Use several different searches."
        ),
    },
]
PASS_BY_ID = {p["id"]: p for p in PASSES}
RATINGS = ["High", "Medium", "Low", "Insufficient evidence"]
RATING_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢", "Insufficient evidence": "⚪", "Error": "⚠️"}


def parse_rating(answer: str) -> str:
    match = re.search(r"RATING:\s*\**\s*(High|Medium|Low|Insufficient evidence)", answer or "", re.I)
    if not match:
        return "Insufficient evidence"
    return next(r for r in RATINGS if r.lower() == match.group(1).lower())


def run_pass(client, model, collection, p, context, registry, max_turns, limit):
    s = agent._new_state(
        p["objective"] + ANSWER_FORMAT,
        context,
        sources=registry,
        allow_ask=False,
        allow_report=False,
        force_finish=True,
    )
    try:
        agent.run_agent(client, model, collection, max_turns, limit, s=s)
    except Exception as error:  # never lose the whole run because one review failed
        s["status"] = "error"
        s["final"] = None
        agent._trace(s, "stop", f"Unexpected error: {error}")
    answer = s["final"] or "No conclusion was reached."
    return {
        "title": p["title"],
        "answer": answer,
        "rating": "Error" if s["status"] == "error" else parse_rating(answer),
        "state": s,
        "turns": s["turn"],
        "passages": len(s["seen"]),
    }


REPORT_SYSTEM = (
    "You are a senior M&A lawyer writing a report for the Board of the purchaser. Write ONLY from the findings supplied. "
    "Keep every [S#] citation exactly as given; never invent facts, figures, clause numbers, dates or citations. "
    "If a review found insufficient evidence, say so plainly rather than filling the gap. Be concise and decisive; "
    "the Board needs to sign off this afternoon. Use markdown with exactly these sections:\n"
    "# Board Report: Proposed Acquisition of Canvassian Pty Ltd\n"
    "## 1. Recommendation (one of: Proceed / Proceed subject to conditions / Do not proceed on current terms / "
    "Cannot recommend without more information; then 3-5 sentences of reasoning)\n"
    "## 2. Risk Summary (a markdown table with columns Risk | Rating | Key point | Suggested protection; use the "
    "ratings supplied, one row per review in the order given)\n"
    "## 3. Critical Risk 1: Founder Dependency\n"
    "## 4. Critical Risk 2: PayWise\n"
    "## 5. Critical Risk 3: Change-of-Control Terms (a markdown table with columns Client | Trigger or restriction? | "
    "Consequence | Source, then any short commentary)\n"
    "## 6. Other Risks Identified (group related points; most serious first)\n"
    "## 7. Recommended Conditions and Next Steps (for example price adjustment, warranties and indemnities, "
    "holdback or escrow, conditions precedent such as client consents and founder retention terms, further due diligence)\n"
    "## 8. Information Gaps and Limitations\n"
    "Do not add a source list; it is appended separately."
)


def build_board_report(client, model, results, registry, prepared_for, context, turns):
    findings = "\n\n".join(
        f"### {p['title']} (rating: {results[p['id']]['rating']})\n{results[p['id']]['answer']}"
        for p in PASSES
        if p["id"] in results
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": f"Case context: {context}\n\nFindings from each review:\n\n{findings}"},
        ],
    )
    body = (resp.choices[0].message.content or "").strip()
    if not body:
        raise ValueError("The model returned an empty report.")

    cited = sorted({int(n) for n in re.findall(r"\bS(\d+)\b", body)})
    by_label = {src["label"]: src for src in registry.values()}
    unknown = [n for n in cited if f"S{n}" not in by_label]

    lines = [body, "", "## Appendix A: Source Register", ""]
    lines.append("Passages cited above (excerpts are reproduced from the documents as indexed).")
    for n in cited:
        src = by_label.get(f"S{n}")
        if not src:
            continue
        excerpt = re.sub(r"\s+", " ", src["text"]).strip()
        excerpt = excerpt[:500] + ("…" if len(excerpt) > 500 else "")
        meta = " · ".join(x for x in (src["doc_type"], src["date"], f"chunk {src['chunk']}", f"chars {src['chars']}") if x)
        lines += ["", f"- **[S{n}]** {src['file']} ({meta})", f"  \"{excerpt}\""]
    if unknown:
        lines += ["", "**Warning:** the report cites " + ", ".join(f"S{n}" for n in unknown) +
                  ", which do not match any retrieved passage. Treat those statements as unverified."]

    lines += ["", "## Appendix B: Review Coverage", "", "| Review | Rating | Turns used | Passages reviewed |", "| --- | --- | --- | --- |"]
    for p in PASSES:
        r = results.get(p["id"])
        if r:
            lines.append(f"| {p['title']} | {r['rating']} | {r['turns']} of {turns} | {r['passages']} |")
        else:
            lines.append(f"| {p['title']} | Not run | 0 | 0 |")
    lines += [
        "",
        "## Appendix C: How This Report Was Produced",
        "",
        f"Prepared for: {prepared_for}. Generated {datetime.now().strftime('%d %B %Y')}. "
        "Each risk was reviewed by an AI agent that searched an indexed library of the transaction documents "
        "(semantic search, exact-phrase search and metadata filters), then the findings were compiled into this report. "
        "This is a preliminary, AI-assisted analysis. A lawyer must check each cited passage against the original "
        "document before the Board relies on it. A finding of 'not found' means the search did not surface the "
        "material, not that it does not exist.",
    ]
    return "\n".join(lines)


def docx_bytes(markdown_text: str) -> bytes:
    lines = markdown_text.splitlines()
    title = "Board Report"
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]
    buf = io.BytesIO()
    agent._write_docx(buf, title, "\n".join(lines))
    return buf.getvalue()


def _coverage(collection):
    metas = collection.get(include=["metadatas"])["metadatas"]
    docs = {}
    for m in metas:
        docs[m.get("source_id")] = m
    counts = Counter(m.get("doc_type", "other") for m in docs.values())
    st.caption("Library: " + " · ".join(f"{n} {kind.replace('_', ' ')}" for kind, n in sorted(counts.items())))
    missing = [name for key, name in CLIENTS if not any(m.get(f"ent_{key}") for m in docs.values())]
    if "contract" not in counts:
        st.warning("No documents were classified as contracts. Check that the contracts were indexed.")
    if missing:
        st.warning("No documents mention: " + ", ".join(missing) + ". Those reviews will find nothing.")
    if not any(m.get("ent_jane_wu") for m in docs.values()):
        st.warning("No documents mention Jane Wu.")


def render_dd_tab(collection, client, model, limit):
    st.subheader("M&A due diligence")
    st.write(
        "Runs one focused AI review per risk (founder, PayWise, change-of-control for each of the six largest "
        "clients, and other risks), then compiles a Board report with a source register."
    )
    if client is None:
        st.info("Add an OpenAI API key in the sidebar to run the due diligence.")
        return
    if collection.count() == 0:
        st.info("Index the transaction documents in the Library tab first (a zip file works).")
        return

    _coverage(collection)
    dd = st.session_state.setdefault("dd", {"registry": {}, "results": {}, "report": None})

    with st.expander("Case context and settings", expanded=False):
        context = st.text_area("Case context given to the AI", value=CASE_CONTEXT, height=190, key="dd_context")
        prepared_for = st.text_input("Prepared for", value="The Board of the Purchaser", key="dd_prepared_for")
        turns = st.slider("Maximum turns per review", 3, 10, 6, key="dd_turns")
        report_model = st.text_input(
            "Model for the final Board report (leave blank to use the same model)", value="", key="dd_report_model"
        )
    titles = {p["id"]: p["title"] for p in PASSES}
    chosen = st.multiselect(
        "Reviews to run", [p["id"] for p in PASSES], default=[p["id"] for p in PASSES],
        format_func=lambda pid: titles[pid], key="dd_chosen",
    )

    todo = [PASS_BY_ID[pid] for pid in chosen if pid not in dd["results"]]
    label = "Run due diligence" if not dd["results"] else f"Run remaining reviews ({len(todo)})"
    if st.button(label, type="primary", disabled=not todo):
        bar = st.progress(0.0, text="Starting…")
        for n, p in enumerate(todo):
            bar.progress(n / len(todo), text=f"Review {n + 1} of {len(todo)}: {p['title']}")
            dd["results"][p["id"]] = run_pass(client, model, collection, p, context, dd["registry"], turns, limit)
            dd["report"] = None
        bar.empty()
        st.rerun()

    if dd["results"]:
        st.markdown("#### Findings by risk")
        for p in PASSES:
            r = dd["results"].get(p["id"])
            if not r:
                continue
            with st.expander(f"{RATING_ICON.get(r['rating'], '•')} {p['title']} · {r['rating']}"):
                st.markdown(r["answer"])
                st.caption(f"{r['turns']} turns · {r['passages']} passages reviewed")
                agent.render_trace(r["state"], compact=True)

        done = [p for p in PASSES if p["id"] in dd["results"]]
        if st.button("Generate Board report", type="primary", disabled=not done):
            try:
                with st.spinner("Compiling the Board report…"):
                    dd["report"] = build_board_report(
                        client, report_model.strip() or model, dd["results"], dd["registry"],
                        prepared_for, context, turns,
                    )
            except Exception as error:
                st.error(f"Could not compile the report: {error}")

    if dd["report"]:
        st.markdown("#### Board report")
        st.markdown(dd["report"])
        left, right = st.columns(2)
        left.download_button(
            "Download as Word (.docx)", data=docx_bytes(dd["report"]), file_name="board-report-canvassian.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        right.download_button(
            "Download as Markdown", data=dd["report"], file_name="board-report-canvassian.md", mime="text/markdown"
        )

    if dd["results"] and st.button("Reset due diligence"):
        st.session_state.pop("dd", None)
        st.rerun()
