import streamlit as st

# Page config
st.set_page_config(page_title="Docker Mastery Quiz", page_icon="🐳", layout="wide")

st.title("🐳 Docker Mastery Quiz")
st.markdown(
    "**Professional Interactive Assessment**: Test your knowledge with 22 carefully crafted questions based on Docker fundamentals, Dockerfile best practices, commands, Compose, and optimization. Receive detailed feedback to enhance your skills!"
)

# Sober Professional CSS
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

# Full-width Progress Bar
progress = len(st.session_state.submitted) / 22 * 100
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
        len(st.session_state.submitted),
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
        st.rerun()

# Hints dictionary
hints = {
    "q1": "Focus on the OS level sharing.",
    "q2": "Docker caches layers from top to bottom.",
    "q3": "The flag references the stage by name.",
    "q4": "Use something dynamic in the command.",
    "q5": "It's the starting image layer.",
    "q6": "Changes early in the file affect more layers.",
    "q7": "It's a subcommand of docker image.",
    "q8": "It's for data persistence, not networking.",
    "q9": "It's not 'run', which starts new containers.",
    "q10": "Default behavior for missing images.",
    "q11": "It's a lightweight operation.",
    "q12": "Multi-container management.",
    "q13": "Format is host:container.",
    "q14": "Fast, in-memory store.",
    "q15": "Builds and starts in background.",
    "q16": "ps shows status and ports.",
    "q17": "Top-level volumes with driver key.",
    "q18": "It affects the build context.",
    "q19": "--cpus for CPU limits.",
    "q20": "Multi-stage keeps only the final stage.",
    "q21": "Build and compose with build flag.",
    "q22": "Single by default.",
}


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
        st.markdown(
            f'<div class="hint-box">{hints[q_key]}</div>', unsafe_allow_html=True
        )


# Section 1: Docker Fundamentals
st.markdown(
    '<div class="section-header">Section 1: Docker Fundamentals</div>',
    unsafe_allow_html=True,
)

with st.expander(
    "Q1: What key component do containers share with the host operating system, making them more lightweight than virtual machines?",
    expanded=True,
):
    toggle_hint("q1")
    options = [
        "The full guest operating system",
        "The host kernel",
        "Dedicated hardware resources",
        "A separate boot loader",
    ]
    ans = st.radio("Your answer:", options, key="q1", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q1",
            ans == "The host kernel",
            "Sharing the kernel allows containers to leverage the host's OS while providing isolation, resulting in faster startup and lower overhead compared to VMs.",
            "Containers share the **host kernel**. VMs run a full guest OS, increasing resource use.",
        )
    if "q1" in st.session_state.submitted:
        show_feedback("q1")

with st.expander(
    "Q2: True or False: Modifying a Dockerfile instruction only invalidates the cache for that layer and all subsequent layers, leaving earlier layers intact.",
    expanded=True,
):
    toggle_hint("q2")
    tf = st.radio("True or False:", ["True", "False"], key="q2", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q2",
            tf == "True",
            "Yes, Docker builds layers sequentially. Changes to a layer invalidate it and all following layers, but earlier ones remain cached for efficiency.",
            "True. This caching mechanism speeds up iterative builds by reusing unchanged layers.",
        )
    if "q2" in st.session_state.submitted:
        show_feedback("q2")

# Section 2: Dockerfile
st.markdown(
    '<div class="section-header">Section 2: Dockerfile Best Practices</div>',
    unsafe_allow_html=True,
)

with st.expander(
    "Q3: In a multi-stage Dockerfile, what flag and syntax allow you to copy files from a previous stage named 'builder' to the current stage?",
    expanded=True,
):
    toggle_hint("q3")
    q3 = st.text_input(
        "Enter the command syntax:", placeholder="Type the full instruction", key="q3"
    )
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q3",
            "--from=builder" in q3.lower(),
            "The --from flag lets you select artifacts from earlier stages, discarding unnecessary build tools in the final image for size and security benefits.",
            "Use COPY --from=builder /path/in/builder /path/in/current to transfer files between stages.",
        )
    if "q3" in st.session_state.submitted:
        show_feedback("q3")

with st.expander(
    "Q4: Which RUN instruction pattern is often used to deliberately invalidate the build cache for testing or dependency updates?",
    expanded=True,
):
    toggle_hint("q4")
    options = [
        "RUN apt-get update && apt-get install -y package",
        'RUN echo "$(date)" > /tmp/cache-bust.txt',
        "RUN --mount=type=cache ...",
        "RUN FROM scratch",
    ]
    ans = st.radio("Your answer:", options, key="q4", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q4",
            'RUN echo "$(date)" > /tmp/cache-bust.txt' in ans,
            "Appending a dynamic value like $(date) to a RUN command changes the layer's content each time, forcing a rebuild of that and subsequent layers.",
            'RUN echo "$(date)" (or similar dynamic output) busts the cache by making the layer unique.',
        )
    if "q4" in st.session_state.submitted:
        show_feedback("q4")

with st.expander(
    "Q5: The FROM instruction is the foundation of a Dockerfile because it specifies the initial environment. What is this initial environment called?",
    expanded=True,
):
    toggle_hint("q5")
    q5 = st.text_input("Term:", placeholder="Enter the key term", key="q5")
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q5",
            "base image" in q5.lower(),
            "The base image provides the OS, libraries, and tools that your application builds upon, forming the first cacheable layer.",
            "The **base image**, e.g., FROM node:18-alpine establishes the runtime foundation.",
        )
    if "q5" in st.session_state.submitted:
        show_feedback("q5")

with st.expander(
    "Q6: In a Dockerfile with layers ordered as FROM > RUN (install deps) > COPY (app code), which modification would result in the shortest rebuild time?",
    expanded=True,
):
    toggle_hint("q6")
    options = [
        "Updating a dependency in the RUN layer",
        "Changing the base image in FROM",
        "Editing source code in the final COPY",
        "Adding a new environment variable",
    ]
    ans = st.radio("Your answer:", options, key="q6", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q6",
            "Editing source code in the final COPY" in ans,
            "Changes to the last COPY only invalidate that layer, reusing all previous cached layers for quick iterations during development.",
            "Editing the final COPY, as it only rebuilds the application layer without touching dependencies.",
        )
    if "q6" in st.session_state.submitted:
        show_feedback("q6")

with st.expander(
    "Q7: What command reveals the history of an image's layers, including the command that created each and their sizes?",
    expanded=True,
):
    toggle_hint("q7")
    q7 = st.text_input("Command:", placeholder="docker image ...", key="q7")
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q7",
            "history" in q7.lower(),
            "docker image history shows each layer's details, helping optimize image size by spotting bloated steps.",
            "docker image history <image> lists layers with sizes and commands.",
        )
    if "q7" in st.session_state.submitted:
        show_feedback("q7")

with st.expander(
    "Q8: To persist data beyond the lifecycle of a container, which flag mounts a host directory into the container?",
    expanded=True,
):
    toggle_hint("q8")
    options = [
        "-p for ports",
        "-v for volumes",
        "--network for connectivity",
        "-e for environment variables",
    ]
    ans = st.radio("Your answer:", options, key="q8", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q8",
            "-v for volumes" in ans,
            "The -v flag creates a bind mount, linking host and container filesystems for durable storage.",
            "-v (or --volume) /host/path:/container/path mounts directories.",
        )
    if "q8" in st.session_state.submitted:
        show_feedback("q8")

# Section 3: Commands & Components
st.markdown(
    '<div class="section-header">Section 3: Commands & Components</div>',
    unsafe_allow_html=True,
)

with st.expander("Q9: Interactive shell in a running container?", expanded=True):
    toggle_hint("q9")
    options9 = [
        "docker run -it container bash",
        "docker exec -it container bash",
        "docker attach container",
        "docker logs -f",
    ]
    ans9 = st.radio("Your answer:", options9, key="q9", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q9",
            ans9 == "docker exec -it container bash",
            "Correct! exec allows shell access to running containers.",
            "Answer: docker exec -it container bash",
        )
    if "q9" in st.session_state.submitted:
        show_feedback("q9")

with st.expander(
    "Q10: docker run image always pulls latest from Hub if missing? (T/F)",
    expanded=True,
):
    toggle_hint("q10")
    tf10 = st.radio("True or False:", ["True", "False"], key="q10", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q10",
            tf10 == "True",
            "Correct! It pulls if the image isn't local.",
            "Answer: True",
        )
    if "q10" in st.session_state.submitted:
        show_feedback("q10")

with st.expander("Q11: docker tag creates? (short)", expanded=True):
    toggle_hint("q11")
    q11 = st.text_input("Enter:", placeholder="alias same ID", key="q11")
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q11",
            "alias" in q11.lower() and "id" in q11.lower(),
            "Correct! Tags are aliases pointing to the same image ID.",
            "Expected: alias to same ID",
        )
    if "q11" in st.session_state.submitted:
        show_feedback("q11")

with st.expander("Q12: Docker Compose primarily for?", expanded=True):
    toggle_hint("q12")
    options12 = [
        "a) Single-image builds",
        "b) Multi-service orchestration",
        "c) VM management",
        "d) Cache only",
    ]
    ans12 = st.radio("Your answer:", options12, key="q12", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q12",
            ans12 == "b) Multi-service orchestration",
            "Correct! Compose orchestrates multi-container apps.",
            "Answer: b)",
        )
    if "q12" in st.session_state.submitted:
        show_feedback("q12")

# Section 4: Docker Compose
st.markdown(
    '<div class="section-header">Section 4: Docker Compose</div>',
    unsafe_allow_html=True,
)

with st.expander("Q13: Compose ports mapping (partial)", expanded=True):
    toggle_hint("q13")
    q13 = st.text_input("Enter format:", placeholder="8000:5000", key="q13")
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q13",
            ":" in q13,
            "Correct! Ports are mapped as host:container.",
            "Expected: host:container (e.g., 8000:5000)",
        )
    if "q13" in st.session_state.submitted:
        show_feedback("q13")

with st.expander("Q14: Redis is a ... (short)", expanded=True):
    toggle_hint("q14")
    q14 = st.text_input("Enter:", placeholder="in-memory key-value store", key="q14")
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q14",
            "in-memory" in q14.lower()
            and ("key-value" in q14.lower() or "nosql" in q14.lower()),
            "Correct! Redis is an in-memory NoSQL key-value store for caching.",
            "Expected: in-memory NoSQL key-value store",
        )
    if "q14" in st.session_state.submitted:
        show_feedback("q14")

with st.expander("Q15: docker compose up -d --build does?", expanded=True):
    toggle_hint("q15")
    options15 = [
        "a) Only pulls",
        "b) Builds & starts detached",
        "c) Stops services",
        "d) Lists status",
    ]
    ans15 = st.radio("Your answer:", options15, key="q15", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q15",
            ans15 == "b) Builds & starts detached",
            "Correct! Builds images if needed and starts services in detached mode.",
            "Answer: b)",
        )
    if "q15" in st.session_state.submitted:
        show_feedback("q15")

with st.expander(
    "Q16: docker compose ps shows ports like 0.0.0.0:443->8043/tcp? (T/F)",
    expanded=True,
):
    toggle_hint("q16")
    tf16 = st.radio("True or False:", ["True", "False"], key="q16", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q16",
            tf16 == "True",
            "Correct! It shows service status including port mappings.",
            "Answer: True",
        )
    if "q16" in st.session_state.submitted:
        show_feedback("q16")

with st.expander(
    "Q17: Compose top-level for named volumes with driver (YAML)", expanded=True
):
    toggle_hint("q17")
    q17 = st.text_area(
        "Enter:",
        placeholder="volumes:\n  db-data:\n    driver: flocker",
        key="q17",
        height=100,
    )
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q17",
            "driver" in q17.lower() and "flocker" in q17.lower(),
            "Correct! Defines named volumes with custom drivers like flocker.",
            "Expected: volumes section with driver: flocker",
        )
    if "q17" in st.session_state.submitted:
        show_feedback("q17")

with st.expander("Q18: .dockerignore effect on cache?", expanded=True):
    toggle_hint("q18")
    options18 = [
        "a) No impact",
        "b) Excludes files from context (no cache pollution)",
        "c) Forces rebuild",
        "d) Adds layers",
    ]
    ans18 = st.radio("Your answer:", options18, key="q18", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q18",
            ans18 == "b) Excludes files from context (no cache pollution)",
            "Correct! It reduces build context size and avoids unnecessary layers.",
            "Answer: b)",
        )
    if "q18" in st.session_state.submitted:
        show_feedback("q18")

# Section 5: Optimization & Advanced
st.markdown(
    '<div class="section-header">Section 5: Optimization & Advanced</div>',
    unsafe_allow_html=True,
)

with st.expander("Q19: Flag to limit CPU in docker run? (short)", expanded=True):
    toggle_hint("q19")
    q19 = st.text_input("Enter flag:", placeholder="--cpus=2", key="q19")
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q19",
            "--cpus" in q19.lower(),
            "Correct! Use --cpus for CPU limits and --memory for RAM.",
            "Expected: --cpus",
        )
    if "q19" in st.session_state.submitted:
        show_feedback("q19")

with st.expander(
    "Q20: Multi-stage builds discard build tools for smaller images? (T/F)",
    expanded=True,
):
    toggle_hint("q20")
    tf20 = st.radio("True or False:", ["True", "False"], key="q20", horizontal=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q20",
            tf20 == "True",
            "Correct! Multi-stage builds copy only runtime artifacts, discarding tools.",
            "Answer: True",
        )
    if "q20" in st.session_state.submitted:
        show_feedback("q20")

with st.expander(
    "Q21: Select all Docker commands that can build images (Multi-Select)",
    expanded=True,
):
    toggle_hint("q21")
    options21 = [
        "docker build",
        "docker run",
        "docker compose up --build",
        "docker push",
    ]
    selected21 = st.multiselect("Select all:", options21, key="q21")
    col1, col2 = st.columns([3, 1])
    with col2:
        correct21 = {"docker build", "docker compose up --build"}
        submit_button(
            "q21",
            set(selected21) == correct21,
            "Correct! These commands trigger image builds.",
            "Expected: docker build and docker compose up --build",
        )
    if "q21" in st.session_state.submitted:
        show_feedback("q21")

with st.expander(
    "Q22: Default number of replicas in Docker Compose services? (Numeric)",
    expanded=True,
):
    toggle_hint("q22")
    q22 = st.number_input(
        "Enter number:", min_value=0, max_value=10, value=1, key="q22"
    )
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button(
            "q22",
            q22 == 1,
            "Correct! Services default to 1 replica unless specified.",
            "Expected: 1",
        )
    if "q22" in st.session_state.submitted:
        show_feedback("q22")

# Final Score
st.markdown("---")
if len(st.session_state.submitted) == 22:
    pct = (st.session_state.score / 22) * 100
    st.markdown(
        f'<div class="metric-container"><strong>Final Score: {st.session_state.score}/22 ({pct:.1f}%)</strong></div>',
        unsafe_allow_html=True,
    )
    if pct >= 80:
        st.success("🎉 Excellent mastery of Docker concepts!")
    elif pct >= 60:
        st.info("👍 Solid understanding – continue practicing!")
    else:
        st.warning("📚 Good start – review the feedback for improvement.")
    if st.button("🔄 Reset & Restart Quiz", use_container_width=True):
        reset_quiz()
else:
    st.info(
        f"💡 Complete all 22 questions to view your final score! Current: {len(st.session_state.submitted)}/22"
    )

st.markdown("---")
st.caption(
    "*Run with:* `streamlit run quizzes/docker_quiz.py` | *Covers all docker.md sections with interactive, varied questions.*"
)
