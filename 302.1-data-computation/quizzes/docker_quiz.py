import streamlit as st

# Central questions dictionary (renumbered sequentially 1-37 for logical order)
questions_dict = {
    "q1": {
        "id": "q1",
        "question": "Q1: What Docker command verifies a successful installation by pulling and running a test image?",
        "type": "text",
        "input_label": "Command:",
        "placeholder": "Enter the command to test Docker",
        "check_func": lambda x: "hello-world" in x.lower(),
        "exp_correct": "docker run hello-world pulls the test image from Docker Hub and runs it, printing a success message confirming Docker Engine, daemon, and networking work correctly.",
        "exp_wrong": "docker run hello-world: Tests the full stack (pull, create, run, output); if it succeeds, Docker is ready for data apps like Spark containers.",
        "hint": "It's a simple test image that outputs a welcome message.",
    },
    "q2": {
        "id": "q2",
        "question": "Q2: What key component do containers share with the host operating system, making them more lightweight than virtual machines?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "The full guest operating system",
            "The host kernel",
            "Dedicated hardware resources",
            "A separate boot loader",
        ],
        "correct": "The host kernel",
        "exp_correct": "Containers share the host kernel for process isolation, enabling fast startup (seconds) and low resource use (MBs RAM), ideal for scaling data workloads like Spark clusters.",
        "exp_wrong": "Containers share the **host kernel** (lightweight isolation via namespaces/cgroups). VMs include a full guest OS (slower boot in minutes, GBs RAM/disk) for stronger hardware isolation.",
        "hint": "Containers leverage the host's OS kernel for efficiency, unlike VMs with full guest OS.",
    },
    "q3": {
        "id": "q3",
        "question": "Q3: Which Linux kernel features does Docker use for container isolation (process/filesystem/network) and resource limits?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Namespaces and cgroups",
            "Hypervisors and VMs",
            "Chroot and SELinux only",
            "iptables and AppArmor",
        ],
        "correct": "Namespaces and cgroups",
        "exp_correct": "Namespaces provide isolation (e.g., PID for processes, NET for networks); cgroups limit resources (CPU/memory)—enabling lightweight, secure data tasks without full OS overhead.",
        "exp_wrong": "Namespaces (isolation) and cgroups (limits) are core to Docker; unlike VMs (hypervisors) or simpler tools (chroot).",
        "hint": "Focus on process isolation and resource control mechanisms.",
    },
    "q4": {
        "id": "q4",
        "question": "Q4: What is the primary benefit of Docker's layered image structure for data applications?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "a) Increases image size",
            "b) Enables layer sharing and efficient caching",
            "c) Requires full OS per image",
            "d) Disables portability",
        ],
        "correct": "b) Enables layer sharing and efficient caching",
        "exp_correct": "Layers are diffs; common bases (e.g., python:3.9) share across images, reducing storage/pull time; changes only rebuild affected layers for fast data app iterations.",
        "exp_wrong": "Answer: b) Layered diffs allow reuse (e.g., shared Ubuntu base) and cache hits, optimizing for data pipelines with heavy deps like PySpark.",
        "hint": "Layers allow reuse of unchanged parts across builds/images.",
    },
    "q5": {
        "id": "q5",
        "question": "Q5: In Docker, what distinguishes bind mounts from named volumes for data persistence?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Bind mounts map host paths directly; named volumes are Docker-managed",
            "Bind mounts are always anonymous; named are persistent",
            "Named volumes require host paths; bind mounts are managed",
            "Both are identical in usage",
        ],
        "correct": "Bind mounts map host paths directly; named volumes are Docker-managed",
        "exp_correct": "Bind mounts (-v /host/data:/container/data) link specific host dirs for dev; named volumes (docker volume create mydata) persist independently, shareable across containers for prod data like DBs.",
        "exp_wrong": "Bind: direct host mapping (real-time edits); Named: managed storage (e.g., /var/lib/docker/volumes/mydata), inspectable via docker volume inspect.",
        "hint": "One ties to host filesystem; the other is abstracted by Docker.",
    },
    "q6": {
        "id": "q6",
        "question": "Q6: To access an interactive shell (e.g., bash) in an already running container, which command is correct?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "docker run -it container bash",
            "docker exec -it container bash",
            "docker attach container",
            "docker logs -f",
        ],
        "correct": "docker exec -it container bash",
        "exp_correct": "docker exec -it <container> bash starts a new interactive shell in a running container; -it must precede the container name for TTY allocation.",
        "exp_wrong": "docker exec -it <container> bash ( -it before name); docker run starts a new container, attach joins the main process (e.g., logs), logs views output.",
        "hint": "Use exec for new processes in running containers, not run (new container).",
    },
    "q7": {
        "id": "q7",
        "question": "Q7: When running docker run image, does it automatically pull the latest from Docker Hub if the image is missing locally? (T/F)",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "True",
        "exp_correct": "docker run pulls the image (e.g., alpine:latest from Docker Hub) if not local; use --pull always to force fresh pull even if local exists, ensuring reproducibility.",
        "exp_wrong": "True. By default, it pulls if missing; specify tags like :3.9 for versions, avoiding 'latest' changes.",
        "hint": "Default behavior fetches from registry if image not found locally.",
    },
    "q8": {
        "id": "q8",
        "question": "Q8: Which flag in docker run enables auto-removal of the container after it exits (useful for one-off data tasks)?",
        "type": "text",
        "input_label": "Flag:",
        "placeholder": "Enter the flag for auto-cleanup",
        "check_func": lambda x: "--rm" in x.lower(),
        "exp_correct": "--rm auto-removes the container on exit, preventing clutter from temp tasks (e.g., docker run --rm my-etl process.py); combine with -v for data persistence.",
        "exp_wrong": "Expected: --rm (e.g., docker run -it --rm alpine sh); ideal for testing without leftovers.",
        "hint": "It's for cleanup after non-daemon runs.",
    },
    "q9": {
        "id": "q9",
        "question": "Q9: Which command cleans up unused Docker objects (containers, images, networks, volumes) to free disk space?",
        "type": "text",
        "input_label": "Command:",
        "placeholder": "Enter the cleanup command",
        "check_func": lambda x: "prune" in x.lower(),
        "exp_correct": "docker system prune -a --volumes removes all unused items (dangling images, stopped containers, etc.); run after experiments to reclaim space from test data images.",
        "exp_wrong": "docker system prune (add -a for all unused images, --volumes for data); prevents buildup in iterative data workflows.",
        "hint": "It's for system-wide cleanup of dangling resources.",
    },
    "q10": {
        "id": "q10",
        "question": "Q10: True or False: Modifying a file used in a later COPY instruction in a Dockerfile only invalidates that COPY layer and subsequent layers, leaving earlier layers (e.g., RUN installs) intact and cached.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "True",
        "exp_correct": "Docker builds layers sequentially; changes to a later COPY (e.g., source code) only invalidate from that point onward, reusing earlier cached layers like dependency installs for faster rebuilds.",
        "exp_wrong": "True. This forward-only invalidation optimizes iterative builds by reusing unchanged earlier layers (e.g., RUN apt install remains cached).",
        "hint": "Cache invalidation is forward-only: changes affect the layer and all after it.",
    },
    "q11": {
        "id": "q11",
        "question": 'Q11: In a multi-stage Dockerfile, what flag allows copying files from a previous stage named "builder" to the current stage?',
        "type": "text",
        "input_label": "Enter the flag and syntax:",
        "placeholder": "Type the COPY flag for multi-stage",
        "check_func": lambda x: "--from=builder" in x.lower(),
        "exp_correct": "COPY --from=builder /source/path /dest/path copies artifacts from the 'builder' stage, discarding build tools for a slim final image (e.g., runtime Python without compilers).",
        "exp_wrong": "Use COPY --from=builder /path/in/builder /path/in/current to transfer only necessary files between stages, reducing image size and enhancing security.",
        "hint": "The --from flag references a named stage (e.g., AS builder) for selective copying.",
    },
    "q12": {
        "id": "q12",
        "question": "Q12: Which RUN instruction is commonly used as a cache buster to force invalidation of subsequent layers during builds (e.g., in CI/CD)?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "RUN apt-get update && apt-get install -y package",
            'RUN echo "$(date)" > /tmp/cache-bust.txt',
            "RUN --mount=type=cache ...",
            "RUN FROM scratch",
        ],
        "correct": 'RUN echo "$(date)" > /tmp/cache-bust.txt',
        "exp_correct": "Dynamic output like $(date) or $(git rev-parse HEAD) changes the layer hash each build, invalidating it and all subsequent layers to ensure fresh code/dependencies without full --no-cache.",
        "exp_wrong": 'RUN echo "$(date)" (or git commit hash) creates a unique layer to bust cache strategically, placed after stable deps but before variable code COPY.',
        "hint": "Use a dynamic command (e.g., date or git hash) to generate changing output.",
    },
    "q13": {
        "id": "q13",
        "question": "Q13: The FROM instruction in a Dockerfile specifies the initial environment. What is this initial environment called?",
        "type": "text",
        "input_label": "Term:",
        "placeholder": "Enter the key term for the starting image",
        "check_func": lambda x: "base image" in x.lower(),
        "exp_correct": "The base image (e.g., FROM python:3.9-slim) provides the OS, libraries, and tools as the first layer; pin tags for reproducibility and cache stability.",
        "exp_wrong": "The **base image** (e.g., FROM node:18-alpine) establishes the runtime foundation; changing its tag invalidates all subsequent layers.",
        "hint": "It's the foundational image that all layers build upon.",
    },
    "q14": {
        "id": "q14",
        "question": "Q14: In a Dockerfile with layers FROM > RUN (install deps) > COPY (app code like docker.cow), which modification results in the shortest rebuild time due to optimal cache reuse?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "Updating a dependency in the RUN layer",
            "Changing the base image in FROM",
            "Editing source code in the final COPY (e.g., docker.cow)",
            "Adding a new environment variable",
        ],
        "correct": "Editing source code in the final COPY (e.g., docker.cow)",
        "exp_correct": "Changes to the last COPY only invalidate that layer and CMD, reusing all prior cached layers (e.g., deps install), ideal for frequent code iterations in development.",
        "exp_wrong": "Editing the final COPY (e.g., docker.cow file) only rebuilds from there, as earlier layers like RUN deps remain cached—fastest for dev changes.",
        "hint": "Place frequently changing elements (like code) in later layers to minimize rebuild scope.",
    },
    "q15": {
        "id": "q15",
        "question": "Q15: What command reveals the history of an image's layers, including the instruction that created each and their sizes?",
        "type": "text",
        "input_label": "Command:",
        "placeholder": "Enter the image history command",
        "check_func": lambda x: "history" in x.lower(),
        "exp_correct": "docker image history <image> shows layer details (ID, CREATED BY instruction like RUN pip install, SIZE), helping optimize by identifying bloated steps or verifying cache hits.",
        "exp_wrong": "docker image history <image> lists layers with commands and sizes; use to debug multi-stage efficiency or combine RUNs for fewer layers.",
        "hint": "It's a subcommand under docker image for layer inspection.",
    },
    "q16": {
        "id": "q16",
        "question": "Q16: To persist data beyond a container's lifecycle (e.g., mounting a host directory to /app/data), which flag is used in docker run?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "-p for ports",
            "-v for volumes",
            "--network for connectivity",
            "-e for environment variables",
        ],
        "correct": "-v for volumes",
        "exp_correct": "The -v flag creates bind mounts (e.g., -v /host/data:/app/data) or named volumes for durable storage, essential for datasets in data pipelines surviving container restarts.",
        "exp_wrong": "-v (or --volume) /host/path:/container/path mounts directories; named volumes (docker volume create) are Docker-managed for sharing across containers.",
        "hint": "Use -v for bind mounts in host:container format to link filesystems.",
    },
    "q17": {
        "id": "q17",
        "question": "Q17: What does the WORKDIR instruction in a Dockerfile set for subsequent commands?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "What does it change?",
        "check_func": lambda x: "workdir" in x.lower() or "directory" in x.lower(),
        "exp_correct": "WORKDIR /app sets the current directory for RUN, COPY, CMD, etc. (creates if missing); use relative paths like WORKDIR subdir to organize builds without full paths.",
        "exp_wrong": "WORKDIR establishes the context for later instructions (e.g., WORKDIR /app before COPY . /app); avoids repeated cd in RUN.",
        "hint": "It changes the default path for file operations in the image.",
    },
    "q18": {
        "id": "q18",
        "question": "Q18: Which Dockerfile instruction sets environment variables that persist in the running container (overridable with -e)?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "ARG",
            "ENV",
            "LABEL",
            "EXPOSE",
        ],
        "correct": "ENV",
        "exp_correct": "ENV KEY=value sets vars for build and runtime (e.g., ENV SPARK_HOME=/opt/spark); ARG is build-only; use for data configs like DB_URL=postgres://db:5432.",
        "exp_wrong": "ENV for persistent vars (e.g., PATH=/app/bin:$PATH); ARG for build-time only (not in image); LABEL is metadata.",
        "hint": "Unlike ARG (build-only), this one carries to runtime.",
    },
    "q19": {
        "id": "q19",
        "question": "Q19: In Dockerfile, what does the VOLUME instruction declare for persistent data directories?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "What does it create?",
        "check_func": lambda x: "volume" in x.lower() or "mount" in x.lower(),
        "exp_correct": 'VOLUME ["/data"] declares directories as volume mount points; at runtime, Docker auto-mounts named volumes if not overridden, ensuring data survives restarts (e.g., for DBs).',
        "exp_wrong": 'VOLUME creates persistent mount points (e.g., VOLUME ["/var/lib/postgres"]); use -v to bind at run.',
        "hint": "It signals directories intended for external mounting.",
    },
    "q20": {
        "id": "q20",
        "question": "Q20: What is the difference between CMD and ENTRYPOINT in a Dockerfile? (short)",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Briefly describe their roles",
        "check_func": lambda x: ("cmd" in x.lower() and "entrypoint" in x.lower())
        or ("default" in x.lower() and "executable" in x.lower()),
        "exp_correct": 'ENTRYPOINT sets the main executable (e.g., ENTRYPOINT ["python"]); CMD provides default args (CMD ["app.py"])—together, makes container run like a command; CMD overridable, ENTRYPOINT not easily.',
        "exp_wrong": "ENTRYPOINT: fixed command (e.g., spark-submit); CMD: args/default (overridable with docker run args); use exec form [] for direct execution.",
        "hint": "One is the primary command; the other supplies arguments.",
    },
    "q21": {
        "id": "q21",
        "question": "Q21: What is the recommended Dockerfile order for optimal caching in data apps with heavy dependencies?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "a) COPY code first, then RUN installs",
            "b) FROM base, COPY deps file, RUN install, then COPY code",
            "c) All RUN commands first, then FROM",
            "d) ENV vars before FROM",
        ],
        "correct": "b) FROM base, COPY deps file, RUN install, then COPY code",
        "exp_correct": "Stable elements first (FROM pinned, COPY requirements.txt, RUN pip install) cache deps; variable code last (COPY .) minimizes rebuilds for iterations on ETL scripts.",
        "exp_wrong": "Answer: b) Deps early for cache hits; code late to avoid busting installs—key for efficient data builds with libs like PySpark.",
        "hint": "Put unchanging parts (deps) before changing ones (code).",
    },
    "q22": {
        "id": "q22",
        "question": "Q22: What is Docker Compose primarily used for in Docker workflows?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "a) Single-image builds",
            "b) Multi-service orchestration",
            "c) VM management",
            "d) Cache optimization only",
        ],
        "correct": "b) Multi-service orchestration",
        "exp_correct": "Compose defines and runs multi-container apps via YAML (services like web + db), automating networking/volumes/dependencies for local dev of complex data stacks.",
        "exp_wrong": "Answer: b) Multi-service orchestration (e.g., app + Redis + Postgres); not for single builds (use docker build) or VMs.",
        "hint": "It's for defining and managing multi-container applications declaratively.",
    },
    "q23": {
        "id": "q23",
        "question": "Q23: In docker run, what is the correct format for port mapping (e.g., expose container port 5000 on host 8000)?",
        "type": "text",
        "input_label": "Enter format:",
        "placeholder": "Enter the port syntax",
        "check_func": lambda x: ":" in x
        and ("host" in x.lower() or "8000:5000" in x.lower()),
        "exp_correct": "Ports map as host:container (e.g., -p 8000:5000); host optional for auto-assign. Internal communication uses service names, no -p needed.",
        "exp_wrong": "Expected: host:container (e.g., -p 8000:5000); protocol like /tcp optional. Use for host access only.",
        "hint": "The syntax is host_port:container_port for publishing.",
    },
    "q24": {
        "id": "q24",
        "question": "Q24: What does docker compose up -d --build accomplish?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "a) Only pulls images",
            "b) Builds images if needed and starts services detached",
            "c) Stops all services",
            "d) Lists service status",
        ],
        "correct": "b) Builds images if needed and starts services detached",
        "exp_correct": "up -d runs detached (background); --build forces rebuild of services with build: in YAML (not default—caches otherwise), pulling others; starts in dependency order.",
        "exp_wrong": "Answer: b) Builds (explicit --build) and starts detached (-d); without --build, uses cached images.",
        "hint": "--build triggers local Dockerfile rebuilds; -d for background mode.",
    },
    "q25": {
        "id": "q25",
        "question": "Q25: Does docker compose ps display port mappings in a format like 0.0.0.0:443->8043/tcp for running services? (T/F)",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "True",
        "exp_correct": "docker compose ps shows service status (Up/Exited), names (e.g., project_web_1), and ports (e.g., 0.0.0.0:443->8043/tcp for published mappings), confirming networking.",
        "exp_wrong": "Answer: True. It provides a snapshot of the stack, including bound ports for host access.",
        "hint": "ps lists services with their status and exposed ports.",
    },
    "q26": {
        "id": "q26",
        "question": "Q26: In Docker Compose YAML, how do you define a top-level named volume with a custom driver and size option? (YAML snippet)",
        "type": "textarea",
        "input_label": "Enter YAML:",
        "placeholder": "Enter the structure for defining a named volume with driver and options",
        "height": 150,
        "check_func": lambda x: "driver" in x.lower()
        and "flocker" in x.lower()
        and "size" in x.lower(),
        "exp_correct": "Top-level volumes: db-data: driver: flocker, driver_opts: size: '10GiB'—provisions cluster-aware storage; reference in services as - db-data:/var/lib/postgres/data.",
        "exp_wrong": "Expected: volumes section with driver: flocker and driver_opts: size: '10GiB'; enables scalable, persistent storage for data like databases.",
        "hint": "Define under top-level volumes with driver and driver_opts keys.",
    },
    "q27": {
        "id": "q27",
        "question": "Q27: What is the primary effect of .dockerignore on the Docker build process and cache?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "a) No impact on build or cache",
            "b) Excludes files from build context, preventing cache pollution",
            "c) Forces full rebuild every time",
            "d) Adds extra layers to the image",
        ],
        "correct": "b) Excludes files from build context, preventing cache pollution",
        "exp_correct": ".dockerignore filters the build context (files sent to daemon), excluding irrelevant items like .git/logs to reduce transfer time and avoid unnecessary cache invalidations from changing files.",
        "exp_wrong": "Answer: b) Reduces context size and prevents cache misses from excluded files (e.g., node_modules changes don't bust COPY layers).",
        "hint": "It acts like .gitignore for the build context to optimize transfers and cache.",
    },
    "q28": {
        "id": "q28",
        "question": "Q28: In Docker Compose YAML, what top-level key defines custom networks for service isolation (e.g., front-tier, back-tier)?",
        "type": "text",
        "input_label": "Key:",
        "placeholder": "Enter the top-level key for networks",
        "check_func": lambda x: "networks" in x.lower(),
        "exp_correct": "networks: front-tier: {} back-tier: {} creates isolated subnets; services join via networks: - front-tier, enabling secure tiered access (e.g., UI to DB but not external).",
        "exp_wrong": "Top-level networks define custom bridges; default is shared; use for data security (e.g., backend in private net).",
        "hint": "It's for defining isolated communication subnets.",
    },
    "q29": {
        "id": "q29",
        "question": "Q29: What does docker compose config do to validate a Compose file?",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "What does it perform?",
        "check_func": lambda x: "config" in x.lower() or "validate" in x.lower(),
        "exp_correct": "docker compose config merges multiple YAML files (e.g., base + override), validates syntax, and outputs the effective configuration—catches errors before up.",
        "exp_wrong": "Expected: merges/validates YAML (e.g., docker compose -f base.yml -f dev.yml config); ensures data stacks like app + DB are correctly defined.",
        "hint": "It checks and resolves the full configuration without running.",
    },
    "q30": {
        "id": "q30",
        "question": "Q30: In Docker Compose, what does depends_on specify for services?",
        "type": "radio",
        "input_label": "Your answer:",
        "options": [
            "a) Resource limits",
            "b) Startup order/dependencies",
            "c) Port mappings",
            "d) Volume mounts",
        ],
        "correct": "b) Startup order/dependencies",
        "exp_correct": "depends_on: - db ensures db starts before app; list of service names; pair with healthchecks for readiness (e.g., wait for DB healthy).",
        "exp_wrong": "Answer: b) Defines startup sequence (e.g., redis before web); doesn't wait for ready—use condition: service_healthy.",
        "hint": "It controls the order in which services are launched.",
    },
    "q31": {
        "id": "q31",
        "question": "Q31: What does docker tag primarily create for an existing image? (short)",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "What does it produce?",
        "check_func": lambda x: "alias" in x.lower()
        and ("id" in x.lower() or "layers" in x.lower()),
        "exp_correct": "docker tag creates an additional repository:tag alias pointing to the same image ID/layers without duplication, useful for versioning before push.",
        "exp_wrong": "Expected: alias to the same image ID (no new layers copied); e.g., tag local-app as repo/app:v1.",
        "hint": "It creates a lightweight reference without copying data.",
    },
    "q32": {
        "id": "q32",
        "question": "Q32: What is Redis, and why is it useful in data applications? (short)",
        "type": "text",
        "input_label": "Enter:",
        "placeholder": "Describe Redis briefly",
        "check_func": lambda x: "in-memory" in x.lower()
        and ("key-value" in x.lower() or "redis" in x.lower()),
        "exp_correct": "Redis is an open-source, in-memory NoSQL key-value store with super-low latency (sub-ms) for caching query results or sessions in data pipelines, supporting strings/lists/sets/hashes/pub-sub.",
        "exp_wrong": "Expected: fast, low-latency in-memory NoSQL key-value store; use for caching in ETL (e.g., store intermediate results to avoid recomputes).",
        "hint": "It's a high-speed, in-memory data structure server for caching.",
    },
    "q33": {
        "id": "q33",
        "question": "Q33: Which flag in docker run limits CPU usage (e.g., to 2 cores) for resource control in data jobs?",
        "type": "text",
        "input_label": "Enter flag:",
        "placeholder": "Enter the CPU limit flag",
        "check_func": lambda x: "--cpus" in x.lower(),
        "exp_correct": "--cpus=2.0 limits to 2 CPU cores; pair with --memory=1g for RAM caps to prevent one container (e.g., Spark job) from hogging host resources in multi-task setups.",
        "exp_wrong": "Expected: --cpus (e.g., --cpus=2); use for fair sharing in data clusters; monitor with docker stats.",
        "hint": "Use --cpus for CPU limits and --memory for RAM in docker run.",
    },
    "q34": {
        "id": "q34",
        "question": "Q34: True or False: Multi-stage builds allow discarding build tools (e.g., compilers) from the final image by copying only runtime artifacts, resulting in smaller images.",
        "type": "radio",
        "input_label": "True or False:",
        "options": ["True", "False"],
        "correct": "True",
        "exp_correct": "Multi-stage uses multiple FROM with COPY --from=<stage> to transfer only essentials (e.g., binaries) to a slim runtime stage, excluding heavy build deps for secure, efficient data images.",
        "exp_wrong": "Answer: True. Example: Build in 'builder' stage, copy artifact to 'runtime' FROM scratch/python-slim; discards tools, reducing size from GBs to MBs.",
        "hint": "Stages separate build (tools) from runtime (app only) via selective copying.",
    },
    "q35": {
        "id": "q35",
        "question": "Q35: Which Docker commands can trigger image builds? (Multi-Select)",
        "type": "multiselect",
        "input_label": "Select all:",
        "options": [
            "docker build",
            "docker run",
            "docker compose up --build",
            "docker push",
        ],
        "correct_set": {"docker build", "docker compose up --build"},
        "exp_correct": "docker build directly builds from Dockerfile; docker compose up --build rebuilds services with build: in YAML; run/pull/push use existing images (no build).",
        "exp_wrong": "Expected: docker build (direct) and docker compose up --build (orchestration with rebuild); --build explicit in Compose.",
        "hint": "Look for commands that explicitly invoke Dockerfile processing.",
    },
    "q36": {
        "id": "q36",
        "question": "Q36: In Docker Compose, what is the default number of replicas (instances) for a service unless specified otherwise? (Numeric)",
        "type": "number",
        "input_label": "Enter number:",
        "min_value": 0,
        "max_value": 10,
        "value": 1,
        "correct_value": 1,
        "exp_correct": "Services default to 1 replica; scale with docker compose up --scale service=3 for multiple instances (e.g., load-balanced data workers).",
        "exp_wrong": "Expected: 1 (single instance by default); use deploy.replicas in YAML for explicit scaling in orchestration.",
        "hint": "Compose starts one container per service unless scaled.",
    },
    "q37": {
        "id": "q37",
        "question": "Q37: In Redis (used for caching in data apps), what persistence options does it support? (Multi-Select)",
        "type": "multiselect",
        "input_label": "Select all:",
        "options": [
            "RDB snapshots",
            "AOF logs",
            "In-memory only (no persistence)",
            "SQL transactions",
        ],
        "correct_set": {"RDB snapshots", "AOF logs", "In-memory only (no persistence)"},
        "exp_correct": "Redis supports RDB (periodic snapshots), AOF (append-only logs for durability), or pure in-memory (fastest, volatile); configure in redis.conf or command line (e.g., --appendonly yes).",
        "exp_wrong": "Expected: RDB (snapshots), AOF (logs), in-memory (default, no persistence); not SQL (NoSQL key-value). Use RDB/AOF for durable caching in ETL.",
        "hint": "Redis offers snapshotting, logging, and volatile modes.",
    },
}

# Compute total questions from dict
total_questions = len(questions_dict)

# Page config
st.set_page_config(page_title="Docker Mastery Quiz", page_icon="🐳", layout="wide")

st.title("🐳 Docker Mastery Quiz")
st.markdown(
    "**Professional Interactive Assessment**: Test your knowledge with {} carefully crafted questions covering all aspects of Docker and Compose from docker.md. Receive detailed feedback!".format(
        total_questions
    )
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
        ans = st.radio(input_label, q["options"], key=q["id"], horizontal=True)
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


# Sections configuration (questions now in sequential order q1-q37)
sections = [
    {"title": "Section 1: Introduction & Installation", "question_ids": ["q1"]},
    {
        "title": "Section 2: Core Components (Images, Containers, Volumes, Networks)",
        "question_ids": ["q2", "q3", "q4", "q5"],
    },
    {
        "title": "Section 3: Workflow & Essential Commands",
        "question_ids": ["q6", "q7", "q8", "q9"],
    },
    {
        "title": "Section 4: Dockerfile Deep Dive (Caching, Keywords, Multi-Stage)",
        "question_ids": [
            "q10",
            "q11",
            "q12",
            "q13",
            "q14",
            "q15",
            "q16",
            "q17",
            "q18",
            "q19",
            "q20",
            "q21",
        ],
    },
    {
        "title": "Section 5: Docker Compose Fundamentals & Implementation",
        "question_ids": ["q22", "q23", "q24", "q25", "q26", "q27", "q28", "q29", "q30"],
    },
    {
        "title": "Section 6: Optimization, Parameters, Best Practices & Use Cases",
        "question_ids": ["q31", "q32", "q33", "q34", "q35", "q36", "q37"],
    },
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
        st.success("🎉 Excellent mastery of Docker concepts!")
    elif pct >= 60:
        st.info("👍 Solid understanding – continue practicing!")
    else:
        st.warning("📚 Good start – review the feedback for improvement.")
    if st.button("🔄 Reset & Restart Quiz", use_container_width=True):
        st.session_state.score = 0
        st.session_state.submitted = set()
        st.session_state.feedbacks = {}
        st.session_state.hints_shown = {}
        st.rerun()
else:
    st.info(
        f"💡 Complete all {total_questions} questions to view your final score! Current: {len(st.session_state.submitted)}/{total_questions}"
    )

st.markdown("---")
st.caption(
    "*Run with:* `streamlit run quizzes/docker_quiz.py` | *Comprehensive coverage of docker.md: installation, components, workflow, Dockerfile, Compose, optimization, use cases.*"
)
