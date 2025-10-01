import streamlit as st
import random

# Questions dictionary covering all content in intro.md
questions_dict = {
    "q1": {
        "id": "q1",
        "question": "Q1: What are data pipelines essential for in modern data computation?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Only storing data",
            "Manual data entry",
            "Systematically ingesting, transforming, and delivering data",
            "Hardware optimization only"
        ],
        "correct": "Systematically ingesting, transforming, and delivering data",
        "exp_correct": "Pipelines ensure reproducibility, scalability, and reliability by handling data flow from sources to insights.",
        "exp_wrong": "They bridge raw data to actionable insights, often using tools like Spark or Docker.",
        "hint": "Focus on the backbone of data workflows."
    },
    "q2": {
        "id": "q2",
        "question": "Q2: What key learning outcome involves modeling pipelines using graphs?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Circular graphs",
            "Database schemas",
            "Linear scripts",
            "Directed Acyclic Graphs (DAGs)"
        ],
        "correct": "Directed Acyclic Graphs (DAGs)",
        "exp_correct": "DAGs model tasks and dependencies for ordered execution in pipelines.",
        "exp_wrong": "DAGs are fundamental for visualizing and orchestrating data workflows.",
        "hint": "It's a graph without cycles."
    },
    "q3": {
        "id": "q3",
        "question": "Q3: From what did data pipelines evolve in the 1970s?",
        "type": "text",
        "input_label": "Term:",
        "placeholder": "Enter the early concept",
        "check_func": lambda x: "unix pipes" in x.lower() or "pipes" in x.lower(),
        "exp_correct": "Unix pipes were simple sequential data flows that inspired modern pipelines.",
        "exp_wrong": "Early Unix pipes laid the foundation, evolving to tools like Airflow.",
        "hint": "Simple command chaining in Unix."
    },
    "q4": {
        "id": "q4",
        "question": "Q4: What is a data pipeline primarily?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "A random collection of scripts",
            "A visualization tool",
            "A hardware configuration",
            "A directed sequence of data processing steps"
        ],
        "correct": "A directed sequence of data processing steps",
        "exp_correct": "It ingests raw data, applies transformations, and outputs refined data acyclically.",
        "exp_wrong": "Emphasis on directed, acyclic flow for deterministic execution.",
        "hint": "Focus on data flow without loops."
    },
    "q5": {
        "id": "q5",
        "question": "Q5: In a DAG, what do nodes represent?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Data storage",
            "User interfaces",
            "Hardware resources",
            "Atomic tasks"
        ],
        "correct": "Atomic tasks",
        "exp_correct": "Nodes are tasks like 'clean data' or 'join tables'; edges show dependencies.",
        "exp_wrong": "DAGs use nodes for tasks and directed edges for precedence without cycles.",
        "hint": "Individual processing steps."
    },
    "q6": {
        "id": "q6",
        "question": "Q6: What guarantees termination in a DAG?",
        "type": "text",
        "input_label": "Property:",
        "placeholder": "Enter the key property",
        "check_func": lambda x: "acyclicity" in x.lower() or "no cycles" in x.lower(),
        "exp_correct": "Acyclicity ensures no loops, allowing topological sorting for execution order.",
        "exp_wrong": "No cycles prevent infinite loops and enable parallel independent tasks.",
        "hint": "No loops in the graph."
    },
    "q7": {
        "id": "q7",
        "question": "Q7: In ETL, where is the schema enforced?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "On read (during query)",
            "During extraction only",
            "No schema enforcement",
            "On write (during load)"
        ],
        "correct": "On write (during load)",
        "exp_correct": "ETL transforms before loading, enforcing schema early to reject malformed data.",
        "exp_wrong": "ETL: Extract → Transform (schema-on-write) → Load; contrasts with ELT's on-read.",
        "hint": "Schema during transformation."
    },
    "q8": {
        "id": "q8",
        "question": "Q8: Which storage paradigm uses schema-on-read for raw data?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Data Warehouse",
            "Relational Database",
            "Data Mart",
            "Data Lake"
        ],
        "correct": "Data Lake",
        "exp_correct": "Data lakes store raw, multi-format data flexibly without early rejection.",
        "exp_wrong": "Lakes for ingestion-first; warehouses enforce schema-on-write for queries.",
        "hint": "Vast repository for semistructured data."
    },
    "q9": {
        "id": "q9",
        "question": "Q9: What is the first step in the pipeline workflow?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Loading data",
            "Monitoring",
            "Visualization",
            "Planning: Identify sources/sinks and define DAG"
        ],
        "correct": "Planning: Identify sources/sinks and define DAG",
        "exp_correct": "Planning sets up tasks and dependencies before ingestion.",
        "exp_wrong": "Start with planning the DAG structure for the entire workflow.",
        "hint": "Before any data handling."
    },
    "q10": {
        "id": "q10",
        "question": "Q10: How is latency calculated in a parallel DAG path?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Sum of all tasks",
            "Min of parallel tasks",
            "Average of paths",
            "Max of parallel tasks + sequential"
        ],
        "correct": "Max of parallel tasks + sequential",
        "exp_correct": "Latency is the longest path: sum sequential, max for parallel branches.",
        "exp_wrong": "For two 7s parallel tasks after 5s extract, latency = 5s + max(7s) + load.",
        "hint": "Critical path determines end-to-end time."
    },
    "q11": {
        "id": "q11",
        "question": "Q11: What transformation example is aggregating data?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Storing raw data",
            "Extracting from source",
            "Loading to sink",
            "Summarizing with GROUP BY"
        ],
        "correct": "Summarizing with GROUP BY",
        "exp_correct": "Aggregation reduces data via summaries like averages or counts.",
        "exp_wrong": "Transformations include cleaning, enriching, and aggregating.",
        "hint": "Reducing data volume."
    },
    "q12": {
        "id": "q12",
        "question": "Q12: What makes a pipeline graph invalid?",
        "type": "multiselect",
        "input_label": "Select all:",
        "options": [
            "Cyclic dependencies",
            "Linear sequence",
            "Undirected edges",
            "Branched parallelism"
        ],
        "correct_set": {"Cyclic dependencies", "Undirected edges"},
        "exp_correct": "Cycles cause non-termination; undirected edges create ambiguity.",
        "exp_wrong": "Valid graphs are acyclic and directed; linear/branched are fine if no cycles.",
        "hint": "Things that prevent topological order."
    },
    "q13": {
        "id": "q13",
        "question": "Q13: What advantage do DAGs provide for parallelism?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "All tasks sequential",
            "Increased cycles",
            "No execution order",
            "Independent nodes run concurrently"
        ],
        "correct": "Independent nodes run concurrently",
        "exp_correct": "DAGs reveal parallel opportunities for multi-core optimization.",
        "exp_wrong": "No incoming edges between independents allow simultaneous execution.",
        "hint": "Natural parallelism in branches."
    },
    "q14": {
        "id": "q14",
        "question": "Q14: What hardware limit affects parallelism on laptops?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Screen size",
            "Battery life",
            "CPU Cores",
            "Keyboard type"
        ],
        "correct": "CPU Cores",
        "exp_correct": "Number of cores limits concurrent tasks (e.g., 4 cores → 4 max).",
        "exp_wrong": "CPU cores dictate how many parallel branches can run.",
        "hint": "Processing power for tasks."
    },
    "q15": {
        "id": "q15",
        "question": "Q15: Why use Docker in pipelines?",
        "type": "text",
        "input_label": "Reason:",
        "placeholder": "Enter reproducibility",
        "check_func": lambda x: "reproducibility" in x.lower() or "isolation" in x.lower(),
        "exp_correct": "Docker ensures 'once built, runs anywhere' by isolating dependencies.",
        "exp_wrong": "It solves 'it works on my machine' via containerization.",
        "hint": "Encapsulates environments."
    },
    "q16": {
        "id": "q16",
        "question": "Q16: What Docker command starts an interactive container with port mapping?",
        "type": "text",
        "input_label": "Command:",
        "placeholder": "Enter docker run",
        "check_func": lambda x: "docker run --rm -it -p" in x.lower(),
        "exp_correct": "docker run --rm -it -p 8888:8888 myimage for Jupyter access.",
        "exp_wrong": "Use -it for interactive, -p for ports, --rm for cleanup.",
        "hint": "Includes -it and -p flags."
    },
    "q17": {
        "id": "q17",
        "question": "Q17: In best practices, what should tasks be for resilience?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Cyclic",
            "Manual",
            "Non-deterministic",
            "Idempotent"
        ],
        "correct": "Idempotent",
        "exp_correct": "Idempotent tasks allow reruns without side effects or duplicates.",
        "exp_wrong": "Rerunnable without changing results on repeat execution.",
        "hint": "Safe to retry."
    },
    "q18": {
        "id": "q18",
        "question": "Q18: What is a common pitfall in pipelines and its solution?",
        "type": "textarea",
        "input_label": "Pitfall and Solution:",
        "placeholder": "Enter resource exhaustion and Docker limits",
        "height": 100,
        "check_func": lambda x: "resource" in x.lower() and "docker" in x.lower(),
        "exp_correct": "Resource exhaustion: Set Docker limits like --cpus=2 --memory=4g.",
        "exp_wrong": "Monitor with docker stats; cap resources to prevent OOM errors.",
        "hint": "Hardware limits and container controls."
    },
    "q19": {
        "id": "q19",
        "question": "Q19: Which use case involves ELT to a data lake for ML?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Manual reporting",
            "Hardware testing",
            "UI design",
            "Machine Learning feature stores"
        ],
        "correct": "Machine Learning feature stores",
        "exp_correct": "ELT loads raw to lake, then transforms for ML features.",
        "exp_wrong": "Ideal for exploratory ML with flexible schema-on-read.",
        "hint": "Big data for models."
    },
    "q20": {
        "id": "q20",
        "question": "Q20: What benefit of pipelines involves cutting processing time?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Increased complexity",
            "More errors",
            "Slower execution",
            "Efficiency via parallelism (30-70% reduction)"
        ],
        "correct": "Efficiency via parallelism (30-70% reduction)",
        "exp_correct": "DAG branches enable concurrent tasks, reducing latency significantly.",
        "exp_wrong": "Parallelism in DAGs optimizes for multi-core systems.",
        "hint": "Time savings from concurrency."
    },
    "q21": {
        "id": "q21",
        "question": "Q21: According to core principles, what is a data warehouse optimized for?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Raw storage only",
            "Real-time streaming",
            "No structure",
            "OLAP queries with schema-on-write"
        ],
        "correct": "OLAP queries with schema-on-write",
        "exp_correct": "Warehouses store curated data for analytics with enforced integrity.",
        "exp_wrong": "Centralized for structured OLAP; contrasts with flexible lakes.",
        "hint": "Query-optimized storage."
    },
    "q22": {
        "id": "q22",
        "question": "Q22: What tool is recommended for DAG scheduling?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Excel",
            "Apache Airflow",
            "Notepad",
            "Paint"
        ],
        "correct": "Apache Airflow",
        "exp_correct": "Airflow orchestrates DAGs with scheduling and monitoring.",
        "exp_wrong": "Advanced orchestration for production pipelines.",
        "hint": "Workflow engine."
    },
    "q23": {
        "id": "q23",
        "question": "Q23: In the workflow diagram, what follows dependency check if met?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Plan DAG",
            "Validate outputs",
            "Ingest data",
            "Transform data"
        ],
        "correct": "Transform data",
        "exp_correct": "After checking dependencies, proceed to transformations like clean/enrich.",
        "exp_wrong": "Dependency met → Transform; pending → parallel branches.",
        "hint": "Data processing step."
    },
    "q24": {
        "id": "q24",
        "question": "Q24: What is a limitation of large DAGs?",
        "type": "text",
        "input_label": "Issue:",
        "placeholder": "Enter debugging difficulty",
        "check_func": lambda x: "debug" in x.lower() or "complexity" in x.lower(),
        "exp_correct": "Hard to debug; solution: modular sub-DAGs.",
        "exp_wrong": "Complexity in large graphs; break into smaller components.",
        "hint": "Maintenance challenge."
    },
    "q25": {
        "id": "q25",
        "question": "Q25: How many core principles are summarized in the takeaways table? (Numeric)",
        "type": "number",
        "input_label": "Enter number:",
        "min_value": 0,
        "max_value": 10,
        "value": 5,
        "correct_value": 5,
        "exp_correct": "5 principles: DAGs, ETL/ELT, Latency, Storage, Docker.",
        "exp_wrong": "The table lists 5 key insights for data pipelines.",
        "hint": "Count the rows in the table."
    }
}

# Compute total questions
total_questions = len(questions_dict)

# Page config
st.set_page_config(page_title="Data Pipelines Quiz", page_icon="🔄", layout="wide")

st.title("🔄 Data Pipelines Introduction Quiz")
st.markdown(
    f"**Interactive Assessment**: Test your knowledge with {total_questions} questions covering the intro to data pipelines summary. Detailed feedback provided!"
)

# CSS styling (adapted from docker_quiz.py)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    body { font-family: 'Roboto', sans-serif; }
    .main-header { color: #1f2937; font-size: 2.5em; text-align: center; margin-bottom: 0.5em; font-weight: 700; }
    .section-header { color: #374151; font-size: 1.8em; font-weight: 600; border-bottom: 2px solid #d1d5db; padding-bottom: 0.5em; margin: 2em 0 1em 0; }
    .expander-header { font-weight: 500; color: #374151; }
    .feedback-success { background-color: #f9fafb; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981; margin-top: 10px; color: #065f46; }
    .feedback-error { background-color: #fef7f7; padding: 12px; border-radius: 8px; border-left: 4px solid #dc2626; margin-top: 10px; color: #991b1b; }
    .submit-btn { background-color: #374151; color: white; border-radius: 6px; padding: 8px 16px; font-weight: 500; border: none; }
    .submit-btn:hover { background-color: #4b5563; }
    .submit-btn:disabled { background-color: #9ca3af; color: #6b7280; }
    .progress-bar { background-color: #e5e7eb; border-radius: 10px; height: 20px; margin: 10px 0; overflow: hidden; border: 1px solid #d1d5db; width: 100%; }
    .progress-fill { background: linear-gradient(90deg, #10b981, #34d399); height: 100%; transition: width 0.3s ease; border-radius: 9px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1); }
    .hint-box { background-color: #fffbeb; padding: 8px 12px; border-radius: 6px; border-left: 3px solid #f0b90b; margin: 8px 0; font-size: 0.95em; color: #a16207; }
    .hint-btn { background-color: #f0b90b; color: #92400e; border-radius: 4px; padding: 4px 8px; font-size: 0.85em; margin-left: 10px; }
    .stRadio > label { font-weight: 400; padding: 6px 12px; margin: 2px; border-radius: 4px; }
    .stTextInput > div > label { display: none; }
    .stTextArea > div > label { display: none; }
    .stNumberInput > div > label { display: none; }
    .stMultiSelect > div > label { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state
if "score" not in st.session_state:
    st.session_state.score = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = set()
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = {}
if "hints_shown" not in st.session_state:
    st.session_state.hints_shown = {}
if "shuffled_options" not in st.session_state:
    st.session_state.shuffled_options = {}

# Progress bar
progress = len(st.session_state.submitted) / total_questions * 100
st.markdown(
    f'<div class="progress-bar"><div class="progress-fill" style="width: {progress}%;"></div></div>',
    unsafe_allow_html=True,
)

# Metrics
st.markdown(
    f"""
    <div style='display: flex; justify-content: space-around; align-items: center; margin: 20px 0; padding: 15px;'>
      <div style='text-align: center; flex: 1;'>
        <div style='font-size: 1.2em; font-weight: bold;'>Progress</div>
        <div style='font-size: 1.5em;'>{len(st.session_state.submitted)} / {total_questions}</div>
      </div>
      <div style='text-align: center; flex: 1;'>
        <div style='font-size: 1.2em; font-weight: bold;'>Score</div>
        <div style='font-size: 1.5em;'>{st.session_state.score} / {len(st.session_state.submitted)}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar reset
with st.sidebar:
    if st.button("🔄 Reset Quiz"):
        st.session_state.score = 0
        st.session_state.submitted = set()
        st.session_state.feedbacks = {}
        st.session_state.hints_shown = {}
        st.session_state.shuffled_options = {}
        st.rerun()

def submit_button(q_key, correct_condition, explanation_correct, explanation_wrong, disabled=False):
    if disabled:
        st.button("✅ Submitted", disabled=True, key=f"sub_{q_key}_dis")
        return
    if st.button("Check Answer", key=f"sub_{q_key}"):
        st.session_state.submitted.add(q_key)
        if correct_condition:
            st.session_state.score += 1
            st.session_state.feedbacks[q_key] = f"Correct! {explanation_correct}"
        else:
            st.session_state.feedbacks[q_key] = f"Incorrect. {explanation_wrong}"
        st.rerun()

def show_feedback(q_key):
    if q_key in st.session_state.submitted:
        fb = st.session_state.feedbacks.get(q_key, "")
        if "Correct" in fb:
            st.markdown(f'<div class="feedback-success">{fb}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="feedback-error">{fb}</div>', unsafe_allow_html=True)

def toggle_hint(q_key):
    if q_key not in st.session_state.hints_shown:
        st.session_state.hints_shown[q_key] = False
    button_text = "Hide Hint" if st.session_state.hints_shown[q_key] else "Show Hint"
    if st.button(button_text, key=f"hint_btn_{q_key}"):
        st.session_state.hints_shown[q_key] = not st.session_state.hints_shown[q_key]
        st.rerun()
    if st.session_state.hints_shown[q_key]:
        hint_text = questions_dict[q_key]["hint"]
        st.markdown(f'<div class="hint-box">{hint_text}</div>', unsafe_allow_html=True)

def render_question(q):
    toggle_hint(q["id"])
    condition = False
    if q["type"] == "radio":
        input_label = q.get("input_label", "Your answer:")
        if q["id"] not in st.session_state.shuffled_options:
            shuffled = q["options"][:]
            random.shuffle(shuffled)
            st.session_state.shuffled_options[q["id"]] = shuffled
        shuffled_options = st.session_state.shuffled_options[q["id"]]
        ans = st.radio(input_label, shuffled_options, key=q["id"], horizontal=True)
        condition = ans == q["correct"]
    elif q["type"] == "text":
        input_label = q["input_label"]
        placeholder = q.get("placeholder", "")
        ans = st.text_input(input_label, placeholder=placeholder, key=q["id"])
        condition = q["check_func"](ans)
    elif q["type"] == "textarea":
        input_label = q["input_label"]
        placeholder = q.get("placeholder", "")
        height = q.get("height", 100)
        ans = st.text_area(
            input_label, placeholder=placeholder, key=q["id"], height=height
        )
        condition = q["check_func"](ans)
    elif q["type"] == "multiselect":
        input_label = q["input_label"]
        selected = st.multiselect(input_label, q["options"], key=q["id"])
        condition = set(selected) == q["correct_set"]
    elif q["type"] == "number":
        input_label = q["input_label"]
        min_value = q.get("min_value", 0)
        max_value = q.get("max_value", 10)
        value = q.get("value", 1)
        ans = st.number_input(
            input_label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            key=q["id"],
        )
        condition = ans == q["correct_value"]
    col1, col2 = st.columns([3, 1])
    with col1:
        submit_button(q["id"], condition, q["exp_correct"], q["exp_wrong"])
    if q["id"] in st.session_state.submitted:
        show_feedback(q["id"])

# Sections based on TOC
sections = [
    {"title": "Introduction to Data Pipelines", "question_ids": ["q1", "q2", "q3"]},
    {"title": "Fundamental Concepts", "question_ids": ["q4", "q5", "q6", "q7", "q8"]},
    {"title": "Designing and Building Pipelines", "question_ids": ["q9", "q10", "q11"]},
    {"title": "Practical Examples and Visualizations", "question_ids": ["q12", "q13"]},
    {"title": "Resource Management and Tools", "question_ids": ["q14", "q15", "q16"]},
    {"title": "Best Practices and Configuration", "question_ids": ["q17", "q18"]},
    {"title": "Real-World Applications", "question_ids": ["q19", "q20"]},
    {"title": "Key Takeaways and Next Steps", "question_ids": ["q21", "q22", "q23", "q24", "q25"]}
]

# Render sections
for section in sections:
    st.markdown(f'<div class="section-header">{section["title"]}</div>', unsafe_allow_html=True)
    for q_id in section["question_ids"]:
        q = questions_dict[q_id]
        with st.expander(q["question"], expanded=True):
            render_question(q)

# Final score
st.markdown("---")
if len(st.session_state.submitted) == total_questions:
    pct = (st.session_state.score / total_questions) * 100
    st.markdown(
        f'<div class="metric-container"><strong>Final Score: {st.session_state.score}/{total_questions} ({pct:.1f}%)</strong></div>',
        unsafe_allow_html=True,
    )
    if pct >= 80:
        st.success("🎉 Excellent mastery of data pipeline basics!")
    elif pct >= 60:
        st.info("👍 Solid understanding – practice more!")
    else:
        st.warning("📚 Review the summary for improvement.")
    if st.button("🔄 Reset & Restart Quiz", use_container_width=True):
        st.session_state.score = 0
        st.session_state.submitted = set()
        st.session_state.feedbacks = {}
        st.session_state.hints_shown = {}
        st.session_state.shuffled_options = {}
        st.rerun()
else:
    st.info(f"💡 Complete all {total_questions} questions to view your final score! Current: {len(st.session_state.submitted)}/{total_questions}")

st.markdown("---")
st.caption("*Run with:* `streamlit run quizzes/intro_quiz.py` | *Comprehensive coverage of intro.md: pipelines, DAGs, ETL/ELT, storage, design, tools, practices, applications.*")
