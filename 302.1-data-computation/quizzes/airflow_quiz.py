import streamlit as st
import random

# Central questions dictionary (renumbered sequentially 1-33 based on airflow.md coverage)
questions_dict = {
    "q1": {
        "id": "q1",
        "question": "Q1: True or False: Operators describe how to do work; Tasks are instances in a DAG.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["False", "True"],
        "correct": "True",
        "exp_correct": "Operators define task logic (e.g., PythonOperator); Tasks are runtime instances within a DAG, enabling execution of the workflow.",
        "exp_wrong": "True. Operators encapsulate 'how' (e.g., run Python code); Tasks are instantiated from them in the DAG for scheduling.",
        "hint": "Operators are blueprints; Tasks are executions.",
    },
    "q2": {
        "id": "q2",
        "question": "Q2: True or False: The Webserver decides what to run in Airflow.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["False", "True"],
        "correct": "False",
        "exp_correct": "False. The Scheduler decides what and when to run; Webserver only serves the UI for monitoring.",
        "exp_wrong": "False. Scheduler parses DAGs and queues tasks; Webserver visualizes but does not execute.",
        "hint": "Webserver is for UI; execution is handled elsewhere.",
    },
    "q3": {
        "id": "q3",
        "question": "Q3: True or False: The Executor submits tasks to workers.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "True",
        "exp_correct": "True. Executor (e.g., CeleryExecutor) dispatches queued tasks from Scheduler to Workers for parallel execution.",
        "exp_wrong": "True. Executor bridges Scheduler and Workers, enabling scalability like in distributed data pipelines.",
        "hint": "Executor handles task submission to execution slots.",
    },
    "q4": {
        "id": "q4",
        "question": "Q4: True or False: XComs enable task communication despite isolation.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["False", "True"],
        "correct": "True",
        "exp_correct": "True. XComs (Cross-Communication) allow tasks to exchange small data (e.g., paths) via Metadata DB, despite process isolation.",
        "exp_wrong": "True. Use ti.xcom_pull() to fetch from prior tasks; ideal for metadata in ETL chains.",
        "hint": "XComs for small inter-task data sharing.",
    },
    "q5": {
        "id": "q5",
        "question": "Q5: True or False: XComs are suitable for large data blobs.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "False",
        "exp_correct": "False. XComs are for small data (e.g., <1MB refs); use external storage like S3 for large blobs to avoid DB overload.",
        "exp_wrong": "False. Limit XComs to metadata; Hooks for external data transfer in data workflows.",
        "hint": "XComs are lightweight; not for big data.",
    },
    "q6": {
        "id": "q6",
        "question": "Q6: True or False: schedule_interval=None means manual triggers only.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["False", "True"],
        "correct": "True",
        "exp_correct": "True. None disables automatic scheduling; trigger via UI or CLI for testing/on-demand ETL runs.",
        "exp_wrong": "True. Use '@daily' or cron for scheduled; None for manual control.",
        "hint": "None for no auto-runs.",
    },
    "q7": {
        "id": "q7",
        "question": "Q7: True or False: a >> b means B depends on A (B waits for A).",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "True",
        "exp_correct": "True. The >> operator sets dependency: task b runs after a succeeds, ensuring ordered execution in DAGs.",
        "exp_wrong": "True. Use >> for linear deps; [t2, t3] for parallel after t1.",
        "hint": ">> defines task ordering.",
    },
    "q8": {
        "id": "q8",
        "question": "Q8: True or False: Web UI shows DAGs, execution, timings, logs.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["False", "True"],
        "correct": "True",
        "exp_correct": "True. UI provides Graph view (deps), Gantt (timings), logs, and run history for monitoring data pipelines.",
        "exp_wrong": "True. Essential for debugging failures and optimizing slow tasks.",
        "hint": "UI for visualization and logs.",
    },
    "q9": {
        "id": "q9",
        "question": "Q9: True or False: apache-airflow-providers-slack enables Slack notifications.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "True",
        "exp_correct": "True. Providers add Operators/Hooks like SlackOperator for alerts on task failures in data workflows.",
        "exp_wrong": "True. Install via pip; use with Connections for secure integration.",
        "hint": "Providers extend Airflow with extras like notifications.",
    },
    "q10": {
        "id": "q10",
        "question": "Q10: Where are DAG files located in Airflow?",
        "type": "text",
        "input_label": "Directory:",
        "placeholder": "Enter the directory path",
        "check_func": lambda x: "dags" in x.lower(),
        "exp_correct": "DAG files are Python scripts in the dags/ directory, parsed by the Scheduler for workflow definition.",
        "exp_wrong": "Expected: dags/ (e.g., /opt/airflow/dags in Docker); place ETL scripts here.",
        "hint": "Standard folder for workflow files.",
    },
    "q11": {
        "id": "q11",
        "question": "Q11: What does the Metadata DB store in Airflow?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "What is stored?",
        "check_func": lambda x: "dag" in x.lower() or "state" in x.lower() or "run" in x.lower(),
        "exp_correct": "Metadata DB (e.g., PostgreSQL) stores DAG definitions, run history, task instances, and states for retries and UI.",
        "exp_wrong": "Stores DAGs, runs, task states; query for audits like failed tasks in data pipelines.",
        "hint": "Central storage for workflow metadata.",
    },
    "q12": {
        "id": "q12",
        "question": "Q12: What does the Graph view in the UI show?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "What is visualized?",
        "check_func": lambda x: "dependency" in x.lower() or "graph" in x.lower(),
        "exp_correct": "Graph view visualizes task dependencies (>> edges) in the DAG, helping understand execution order.",
        "exp_wrong": "Graph for deps; use Gantt for timings. Key for debugging ETL flows.",
        "hint": "Visualizes task relationships.",
    },
    "q13": {
        "id": "q13",
        "question": "Q13: Which is NOT an Airflow component? (Select the odd one out)",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Executor",
            "Hypervisor",
            "Webserver",
            "Scheduler"
        ],
        "correct": "Hypervisor",
        "exp_correct": "Hypervisor is for VMs, not Airflow; core components are Scheduler, Executor, Webserver, Workers, DB.",
        "exp_wrong": "Hypervisor (VM tech); Airflow uses Executors like Celery for task distribution.",
        "hint": "Airflow is orchestration, not virtualization.",
    },
    "q14": {
        "id": "q14",
        "question": "Q14: What does PythonOperator do?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Its purpose",
        "check_func": lambda x: "python" in x.lower() or "callable" in x.lower(),
        "exp_correct": "PythonOperator executes Python callables (functions), ideal for data transforms and custom logic in DAGs.",
        "exp_wrong": "Runs python_callable=func; pass **context for dates like {{ ds }}.",
        "hint": "For running Python code as tasks.",
    },
    "q15": {
        "id": "q15",
        "question": "Q15: True or False: a >> b means b waits for a.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["False", "True"],
        "correct": "True",
        "exp_correct": "True. >> sets b as dependent on a, ensuring sequential execution in pipelines.",
        "exp_wrong": "True. Core for ordering tasks like extract >> transform.",
        "hint": "Dependency operator.",
    },
    "q16": {
        "id": "q16",
        "question": "Q16: What are good DAG hygiene practices? (Multi-Select)",
        "type": "multiselect",
        "input_label": "Select all:",
        "options": [
            "Log I/O and latency",
            "Explicit dependencies",
            "Small, idempotent tasks",
            "No linting"
        ],
        "correct_set": {"Log I/O and latency", "Explicit dependencies", "Small, idempotent tasks"},
        "exp_correct": "Hygiene: Log metrics, use >> for deps, keep tasks atomic/re-runnable; lint with ruff for clean code.",
        "exp_wrong": "Expected: logging, explicit deps, small tasks; avoid large/monolithic tasks in data flows.",
        "hint": "Practices for maintainable DAGs.",
    },
    "q17": {
        "id": "q17",
        "question": "Q17: List the core Airflow components. (Multi-Select)",
        "type": "multiselect",
        "input_label": "Select all:",
        "options": [
            "Scheduler",
            "Executor",
            "Worker",
            "Metadata DB",
            "Webserver",
            "Docker"
        ],
        "correct_set": {"Scheduler", "Executor", "Worker", "Metadata DB", "Webserver"},
        "exp_correct": "Core: Scheduler (triggers), Executor (submits), Worker (executes), DB (states), Webserver (UI).",
        "exp_wrong": "All except Docker (containerization, not core); essential for scalable orchestration.",
        "hint": "Main architectural pieces.",
    },
    "q18": {
        "id": "q18",
        "question": "Q18: Best practices for XComs in data workflows?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Guidelines",
        "check_func": lambda x: "small" in x.lower() or "external" in x.lower(),
        "exp_correct": "XComs for small data/refs (e.g., paths); use external storage for large; pull with ti.xcom_pull(task_ids='prev').",
        "exp_wrong": "Small only; e.g., return row_count, not full DF; Hooks for big data transfer.",
        "hint": "Limit size; use for metadata.",
    },
    "q19": {
        "id": "q19",
        "question": "Q19: Why use Airflow for data pipelines? (Multi-Select)",
        "type": "multiselect",
        "input_label": "Select all:",
        "options": [
            "Gantt/Graph views",
            "Scheduling and retries",
            "Monitoring",
            "No multi-step support"
        ],
        "correct_set": {"Gantt/Graph views", "Scheduling and retries", "Monitoring"},
        "exp_correct": "Benefits: UI views for deps/timings, auto-retries, monitoring logs; orchestrates complex ETL.",
        "exp_wrong": "All except no multi-step (DAGs handle it); key for reliable data flows.",
        "hint": "Advantages over simple scripts.",
    },
    "q20": {
        "id": "q20",
        "question": "Q20: What is a pipeline in Airflow?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Definition",
        "check_func": lambda x: "dag" in x.lower(),
        "exp_correct": "A pipeline is a DAG: Directed Acyclic Graph of tasks with dependencies for workflow orchestration.",
        "exp_wrong": "DAG models the pipeline; acyclic to prevent loops in execution.",
        "hint": "Graph-based workflow.",
    },
    "q21": {
        "id": "q21",
        "question": "Q21: Who decides what and when to run in Airflow?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Component",
        "check_func": lambda x: "scheduler" in x.lower(),
        "exp_correct": "Scheduler parses DAGs, checks schedules, and queues tasks for execution.",
        "exp_wrong": "Scheduler; monitors dags/ and handles dependencies/retries.",
        "hint": "The triggering component.",
    },
    "q22": {
        "id": "q22",
        "question": "Q22: Where are DAG files placed?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Location",
        "check_func": lambda x: "dags" in x.lower(),
        "exp_correct": "In the dags/ directory; Scheduler watches for Python files defining workflows.",
        "exp_wrong": "dags/; e.g., basic_etl.py with DAG constructor.",
        "hint": "Folder monitored by Scheduler.",
    },
    "q23": {
        "id": "q23",
        "question": "Q23: What does >> declare in a DAG?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Its meaning",
        "check_func": lambda x: "depend" in x.lower() or "wait" in x.lower(),
        "exp_correct": ">> declares dependency: right task waits for left to complete.",
        "exp_wrong": "B depends on A in a >> b; ensures order like extract >> transform.",
        "hint": "Task ordering operator.",
    },
    "q24": {
        "id": "q24",
        "question": "Q24: Ways to declare dependencies in Airflow? (Multi-Select)",
        "type": "multiselect",
        "input_label": "Select all:",
        "options": [
            "Lists [t2, t3]",
            "Tuples (t2, t3)",
            "chain()",
            "No >> operator"
        ],
        "correct_set": {"Lists [t2, t3]", "Tuples (t2, t3)", "chain()"},
        "exp_correct": "Deps via t1 >> [t2, t3] (parallel), (t2, t3) (same), or chain(t1, t2, t3) for linear.",
        "exp_wrong": "All except no >> (it's primary); flexible for complex DAGs.",
        "hint": "Multiple syntaxes for deps.",
    },
    "q25": {
        "id": "q25",
        "question": "Q25: True or False: chain_linear(t0, t1, [t2,t3,t4], [t5,t6,t7], t8) chains parallel groups.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["False", "True"],
        "correct": "True",
        "exp_correct": "True. chain_linear connects tasks/groups sequentially, allowing parallel within groups for efficient data processing.",
        "exp_wrong": "True. Utility for building linear deps with branches.",
        "hint": "For sequential chaining with parallels.",
    },
    "q26": {
        "id": "q26",
        "question": "Q26: True or False: TaskGroup groups tasks like t0 >> tg1 >> t3.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "True",
        "exp_correct": "True. TaskGroup creates sub-DAGs (e.g., with TaskGroup(group_id='tg1')), treated as single tasks in UI Graph.",
        "exp_wrong": "True. For modularity: with TaskGroup() as tg: t1 >> t2; then external >> tg.",
        "hint": "For nesting tasks visually.",
    },
    "q27": {
        "id": "q27",
        "question": "Q27: Command to initialize the Metadata DB?",
        "type": "text",
        "input_label": "Command:",
        "placeholder": "Enter the init command",
        "check_func": lambda x: "db init" in x.lower(),
        "exp_correct": "airflow db init sets up the DB schema for storing DAG states and runs.",
        "exp_wrong": "airflow db init; run after install for core functionality.",
        "hint": "Setup command for DB.",
    },
    "q28": {
        "id": "q28",
        "question": "Q28: Command to list DAGs?",
        "type": "text",
        "input_label": "Command:",
        "placeholder": "Enter the list command",
        "check_func": lambda x: "dags list" in x.lower(),
        "exp_correct": "airflow dags list shows parsed DAGs from dags/ for verification.",
        "exp_wrong": "airflow dags list; check if your ETL DAG loaded.",
        "hint": "CLI to view workflows.",
    },
    "q29": {
        "id": "q29",
        "question": "Q29: Command to test a full DAG run?",
        "type": "text",
        "input_label": "Command:",
        "placeholder": "Enter the test command",
        "check_func": lambda x: "dags test" in x.lower(),
        "exp_correct": "airflow dags test <dag_id> <date> runs the DAG for debugging without scheduling.",
        "exp_wrong": "airflow dags test etl 2023-01-01; validates deps and tasks.",
        "hint": "For manual execution testing.",
    },
    "q30": {
        "id": "q30",
        "question": "Q30: What is schedule_interval=None used for?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Daily runs",
            "No execution",
            "Manual triggers only",
            "Hourly"
        ],
        "correct": "Manual triggers only",
        "exp_correct": "None disables auto-scheduling; use UI/CLI triggers for on-demand or testing.",
        "exp_wrong": "Manual only; set '@daily' for scheduled data ingestion.",
        "hint": "For non-automatic runs.",
    },
    "q31": {
        "id": "q31",
        "question": "Q31: What does start_date define in a DAG?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Its role",
        "check_func": lambda x: "anchor" in x.lower() or "date" in x.lower(),
        "exp_correct": "start_date is the scheduling anchor (immutable datetime); first run after this date.",
        "exp_wrong": "datetime(2023,1,1); set once, affects backfill and intervals.",
        "hint": "Fixed point for scheduling.",
    },
    "q32": {
        "id": "q32",
        "question": "Q32: Default retries in default_args?",
        "type": "number",
        "input_label": "Enter number:",
        "min_value": 0,
        "max_value": 10,
        "value": 0,
        "correct_value": 0,
        "exp_correct": "Default is 0 (no retries); set {'retries': 3} for fault tolerance in data tasks.",
        "exp_wrong": "0 by default; configure for ETL resilience.",
        "hint": "Standard value without config.",
    },
    "q33": {
        "id": "q33",
        "question": "Q33: Best practice for large data in tasks?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Use XComs",
            "In-memory only",
            "No persistence",
            "External storage like S3"
        ],
        "correct": "External storage like S3",
        "exp_correct": "For large data, use Hooks to external storage (S3/DB); XComs for small refs only.",
        "exp_wrong": "External (e.g., S3Hook); prevents DB bloat in scalable pipelines.",
        "hint": "Avoid overloading internal comms.",
    },
}

# Compute total questions from dict
total_questions = len(questions_dict)

# Page config
st.set_page_config(page_title="Apache Airflow Mastery Quiz", page_icon="📊", layout="wide")

st.title("📊 Apache Airflow Mastery Quiz")
st.markdown(
    "**Professional Interactive Assessment**: Test your knowledge with {} carefully crafted questions covering all aspects of Apache Airflow from airflow.md. Receive detailed feedback!".format(
        total_questions
    )
)

# Sober Professional CSS (adapted from Docker quiz)
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

# Full-width Progress Bar
progress = len(st.session_state.submitted) / total_questions * 100
st.markdown(
    f'<div class="progress-bar"><div class="progress-fill" style="width: {progress}%;"></div></div>',
    unsafe_allow_html=True,
)

# Centered Metrics using HTML for precise control
st.markdown(
    """
    <div style='display: flex; justify-content: space-around; align-items: center; margin: 20px 0; padding: 15px; background-color: inherit;'>
      <div style='text-align: center; flex: 1;'>
        <div style='font-size: 1.2em; font-weight: bold;'>Progress</div>
        <div style='font-size: 1.5em;'>{}</div>
      </div>
      <div style='text-align: center; flex: 1;'>
        <div style='font-size: 1.2em; font-weight: bold;'>Score</div>
        <div style='font-size: 1.5em;'>{}</div>
      </div>
    </div>
    """.format(
        f"{len(st.session_state.submitted)} / {total_questions}",
        f"{st.session_state.score} / {len(st.session_state.submitted)}",
    ),
    unsafe_allow_html=True,
)

# Sidebar - Only reset button
with st.sidebar:
    if st.button("🔄 Reset Quiz"):
        st.session_state.score = 0
        st.session_state.submitted = set()
        st.session_state.feedbacks = {}
        st.session_state.hints_shown = {}
        st.session_state.shuffled_options = {}
        st.rerun()

def submit_button(
    q_key, correct_condition, explanation_correct, explanation_wrong, disabled=False
):
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
            st.markdown(
                f'<div class="feedback-success">{fb}</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="feedback-error">{fb}</div>', unsafe_allow_html=True
            )

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

# Sections configuration (questions in sequential order q1-q33, grouped logically like docker_quiz.py)
sections = [
    {"title": "Section 1: Introduction & Basics", "question_ids": ["q1", "q2", "q3", "q4", "q5"]},
    {"title": "Section 2: Core Components", "question_ids": ["q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17"]},
    {"title": "Section 3: DAG Workflow & Dependencies", "question_ids": ["q18", "q19", "q20", "q21", "q22", "q23", "q24", "q25", "q26"]},
    {"title": "Section 4: UI & Monitoring", "question_ids": ["q27", "q28"]},
    {"title": "Section 5: Operators & Providers", "question_ids": ["q29", "q30"]},
    {"title": "Section 6: Commands & Configuration", "question_ids": ["q31", "q32"]},
    {"title": "Section 7: Best Practices & XComs", "question_ids": ["q33"]},
]

# Render sections and questions
for section in sections:
    st.markdown(
        f'<div class="section-header">{section["title"]}</div>',
        unsafe_allow_html=True,
    )
    for q_id in section["question_ids"]:
        q = questions_dict[q_id]
        with st.expander(q["question"], expanded=True):
            render_question(q)

# Final Score
st.markdown("---")
if len(st.session_state.submitted) == total_questions:
    pct = (st.session_state.score / total_questions) * 100
    st.markdown(
        f'<div class="metric-container"><strong>Final Score: {st.session_state.score}/{total_questions} ({pct:.1f}%)</strong></div>',
        unsafe_allow_html=True,
    )
    if pct >= 80:
        st.success("🎉 Excellent mastery of Airflow concepts!")
    elif pct >= 60:
        st.info("👍 Solid understanding – continue practicing!")
    else:
        st.warning("📚 Good start – review the feedback for improvement.")
    if st.button("🔄 Reset & Restart Quiz", use_container_width=True):
        st.session_state.score = 0
        st.session_state.submitted = set()
        st.session_state.feedbacks = {}
        st.session_state.hints_shown = {}
        st.session_state.shuffled_options = {}
        st.rerun()
else:
    st.info(
        f"💡 Complete all {total_questions} questions to view your final score! Current: {len(st.session_state.submitted)}/{total_questions}"
    )

st.markdown("---")
st.caption(
    "*Run with:* `streamlit run quizzes/airflow_quiz.py` | *Comprehensive coverage of airflow.md: components, DAGs, operators, best practices.*"
)
