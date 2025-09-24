import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from sentence_mastermind import SentenceMastermind
    from deap import creator, base, tools
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    import math
    
    # Unified color scheme for consistent styling
    COLORS = {
        'primary': '#2563eb',      # blue-600
        'secondary': '#7c3aed',    # violet-600
        'accent': '#0d9488',       # teal-600
        'success': '#16a34a',      # green-600
        'warning': '#d97706',      # amber-600
        'danger': '#dc2626',       # red-600
        'info': '#0ea5e9',         # sky-500
        'background': '#f8fafc',   # slate-50
        'card': '#ffffff',         # white
        'text': '#1e293b',         # slate-800
        'text_secondary': '#64748b' # slate-500
    }
    
    return (
        COLORS,
        SentenceMastermind,
        base,
        creator,
        mo,
        np,
        plt,
        random,
        sns,
        time,
        tools,
    )


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.md(
        rf"""
    <div style="background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); padding: 30px; border-radius: 15px; color: white; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
    <h1 style="margin-top: 0; font-size: 2.5em; text-align: center;">🧬 Mastermind Solver with Genetic Algorithms</h1>
    <p style="font-size: 1.2em; text-align: center; max-width: 800px; margin: 0 auto;">
    This interactive notebook demonstrates how to solve the <strong>Mastermind puzzle</strong> using <strong>Genetic Algorithms (GA)</strong>. 
    Mastermind is a code-breaking game where one player creates a secret code and the other player tries to guess it.
    </p>
    </div>

    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 30px 0;">
        <div style="flex: 1; min-width: 300px; background-color: {COLORS['card']}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h2 style="color: {COLORS['primary']}; margin-top: 0;">🎯 Learning Objectives</h2>
            <ul style="padding-left: 20px;">
                <li>Understand how genetic algorithms work</li>
                <li>Learn how to apply GAs to optimization problems</li>
                <li>Visualize the evolution process</li>
                <li>Experiment with different parameters and see their effects</li>
            </ul>
        </div>
        
        <div style="flex: 1; min-width: 300px; background-color: {COLORS['card']}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h2 style="color: {COLORS['secondary']}; margin-top: 0;">🔬 How Genetic Algorithms Work</h2>
            <ol style="padding-left: 20px;">
                <li><strong>Initialization</strong> 🧬: Creating a population of random "guesses" (candidate solutions)</li>
                <li><strong>Evaluation</strong> 📊: Evaluating each guess against the hidden sentence using fitness functions</li>
                <li><strong>Selection</strong> 🏆: Selecting the best performers to "breed" based on their fitness</li>
                <li><strong>Crossover</strong> 🔀: Creating new guesses by combining parts of two parent solutions</li>
                <li><strong>Mutation</strong> 🧬: Introducing random changes to maintain diversity</li>
                <li><strong>Replacement</strong> 🔁: Forming a new generation and repeating the process</li>
            </ol>
        </div>
    </div>

    <div style="background-color: {COLORS['card']}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 30px 0;">
        <h2 style="color: {COLORS['accent']}; margin-top: 0;">🛠️ Key Components of This Implementation</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div style="padding: 15px; background-color: rgba(37, 99, 235, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['primary']};">
                <strong>DEAP Framework</strong>: Using the Distributed Evolutionary Algorithms in Python library for efficient implementation
            </div>
            <div style="padding: 15px; background-color: rgba(124, 58, 237, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['secondary']};">
                <strong>Multi-objective Fitness</strong>: Evaluating solutions based on both correct positions and correct characters
            </div>
            <div style="padding: 15px; background-color: rgba(13, 148, 136, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['accent']};">
                <strong>Tournament Selection</strong>: Selecting parents through competitive tournaments
            </div>
            <div style="padding: 15px; background-color: rgba(22, 163, 74, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['success']};">
                <strong>Uniform Crossover</strong>: Combining parent solutions with uniform probability
            </div>
            <div style="padding: 15px; background-color: rgba(217, 119, 6, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['warning']};">
                <strong>Random Mutation</strong>: Introducing diversity by randomly changing characters
            </div>
            <div style="padding: 15px; background-color: rgba(220, 38, 38, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['danger']};">
                <strong>Diversity Monitoring</strong>: Tracking population diversity to prevent premature convergence
            </div>
        </div>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎨 Genetic Algorithm Pipeline Visualization

    Below is a visualization of the genetic algorithm pipeline we'll be implementing:
    """
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    # Create a diagram of the GA pipeline
    pipeline_diagram = mo.mermaid(
        f"""
        flowchart TD
            A[Initialize Population<br/>Random Guesses] --> B[Evaluate Fitness<br/>Compare with Hidden Sentence]
            B --> C{{Convergence?<br/>Solution Found?}}
            C -->|No| D[Select Parents<br/>Tournament Selection]
            D --> E[Apply Crossover<br/>Uniform Mixing]
            E --> F[Apply Mutation<br/>Random Changes]
            F --> G[Create New Generation]
            G --> B
            C -->|Yes| H[Solution Found!<br/>Return Best Individual]

            style A fill:{COLORS['primary']}20,stroke:{COLORS['primary']},stroke-width:2px
            style B fill:{COLORS['secondary']}20,stroke:{COLORS['secondary']},stroke-width:2px
            style C fill:{COLORS['warning']}20,stroke:{COLORS['warning']},stroke-width:2px
            style D fill:{COLORS['success']}20,stroke:{COLORS['success']},stroke-width:2px
            style E fill:{COLORS['accent']}20,stroke:{COLORS['accent']},stroke-width:2px
            style F fill:{COLORS['danger']}20,stroke:{COLORS['danger']},stroke-width:2px
            style G fill:{COLORS['info']}20,stroke:{COLORS['info']},stroke-width:2px
            style H fill:{COLORS['success']}40,stroke:{COLORS['success']},stroke-width:2px
        """
    ).center()

    pipeline_diagram
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.md(
        rf"""
    <div style="background: linear-gradient(135deg, {COLORS['warning']}10 0%, {COLORS['accent']}10 100%); border-left: 6px solid {COLORS['warning']}; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <h2 style="color: {COLORS['warning']}; margin-top: 0;">🎮 Setting up the Game</h2>

    <p>First, we need to define the hidden sentence that our genetic algorithm will try to guess. The algorithm will attempt to discover this sentence through evolutionary processes.</p>

    <h3 style="color: {COLORS['accent']};">🧠 Game Mechanics</h3>

    <p>In Mastermind:</p>

    <ul>
    <li><strong>Correct Position</strong> 🟢: A character that matches both the value and position in the hidden sentence</li>
    <li><strong>Correct Character</strong> 🟡: A character that exists in the hidden sentence but is in the wrong position</li>
    <li><strong>Fitness Calculation</strong> 📈: Our algorithm optimizes for both metrics, with higher weight on correct positions</li>
    </ul>

    <h3 style="color: {COLORS['accent']};">🧮 Example</h3>

    <p>If the hidden sentence is "<strong>hello</strong>" and we guess "<strong>world</strong>":</p>

    <ul>
    <li>Position 4: 'l' matches both value and position → <strong>1 correct position</strong></li>
    <li>'o' exists in the hidden sentence but at position 5, not 5 → <strong>1 correct character</strong></li>
    </ul>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Create input for hidden sentence
    sentence_input = mo.ui.text(
        value="trotinette",
        placeholder="Enter your hidden sentence here...",
        label="🔤 Hidden Sentence"
    )

    # Create a run button to start the game and run the algorithm
    run_button = mo.ui.run_button(
        label="🚀 Run Genetic Algorithm"
    )
    return run_button, sentence_input


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.md(
        rf"""
    <div style="background: linear-gradient(135deg, {COLORS['secondary']}10 0%, {COLORS['primary']}10 100%); border-left: 6px solid {COLORS['secondary']}; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <h2 style="color: {COLORS['secondary']}; margin-top: 0;">⚙️ Genetic Algorithm Parameters</h2>

    <p>Adjust the parameters to control how the genetic algorithm behaves. These parameters significantly affect the algorithm's performance and convergence speed:</p>

    <h3 style="color: {COLORS['primary']};">🧩 Population Parameters</h3>
    <ul>
    <li><strong>Population Size</strong> 👥: Number of candidate solutions in each generation. Larger populations increase diversity but require more computational resources.</li>
    <li><strong>Generations</strong> 🔄: Maximum number of generations to run. The algorithm may converge before reaching this limit.</li>
    </ul>

    <h3 style="color: {COLORS['primary']};">🧬 Genetic Operators</h3>
    <ul>
    <li><strong>Crossover Probability</strong> 🔀: Chance that two parent solutions will combine to create offspring. Higher values increase exploration.</li>
    <li><strong>Mutation Probability</strong> 🧬: Chance that a character in a solution will be randomly changed. Higher values increase diversity.</li>
    </ul>

    <h3 style="color: {COLORS['primary']};">🏆 Selection Mechanism</h3>
    <ul>
    <li><strong>Tournament Size</strong> 🏟️: Number of individuals competing in each selection tournament (fixed at 3 in this implementation).</li>
    </ul>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo, sentence_input):
    mo.md(
        rf"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 25px;">
        <div style="background: linear-gradient(135deg, {COLORS['primary']}10 0%, {COLORS['info']}10 100%); border-left: 6px solid {COLORS['primary']}; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h3 style="color: {COLORS['primary']}; margin-top: 0;">🖋️ Enter Your Hidden Sentence</h3>

        <p>Type the sentence you want the genetic algorithm to guess. The algorithm will try to discover this sentence through evolutionary processes.</p>

        <p><strong>Current hidden sentence</strong>: <code style="background-color: {COLORS['background']}; padding: 4px 8px; border-radius: 4px;">{sentence_input.value}</code></p>
        </div>

        <div style="background: linear-gradient(135deg, {COLORS['warning']}10 0%, {COLORS['danger']}10 100%); border-left: 6px solid {COLORS['warning']}; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h3 style="color: {COLORS['warning']}; margin-top: 0;">🔤 Character Set</h3>

        <p>The algorithm can use any of the following characters:</p>

        <ul>
        <li><strong>Lowercase letters</strong>: a-z</li>
        <li><strong>Accented letters</strong>: áéíóúüñ</li>
        <li><strong>Digits</strong>: 0-9</li>
        <li><strong>Punctuation</strong>: .,;:!¿?()[]'\"-_/\\|@#$%^&*+=~`&lt;&gt;</li>
        <li><strong>Space character</strong></li>
        </ul>
        </div>
    </div>
        """
    )

    # Create interactive sliders for GA parameters with numeric displays
    population_slider = mo.ui.slider(
        start=20,
        stop=500,
        step=10,
        value=100,
        label="👥 Population Size",
        show_value=True
    )

    generations_slider = mo.ui.slider(
        start=10,
        stop=1000,
        step=10,
        value=200,
        label="🔄 Generations",
        show_value=True
    )

    crossover_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.7,
        label="🔀 Crossover Probability",
        show_value=True
    )

    mutation_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.3,
        label="🧬 Mutation Probability",
        show_value=True
    )
    return (
        crossover_slider,
        generations_slider,
        mutation_slider,
        population_slider,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    crossover_slider,
    generations_slider,
    mo,
    mutation_slider,
    population_slider,
    sentence_input,
):
    mo.md(rf"""
    <div style="background: linear-gradient(135deg, {COLORS['secondary']}10 0%, {COLORS['accent']}10 100%); border-left: 6px solid {COLORS['secondary']}; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 25px 0;">
    <h3 style="color: {COLORS['secondary']}; margin-top: 0; text-align: center;">🎛️ Adjust Parameters</h3>
    </div>
    """)

    # Display the hidden sentence input and sliders with their current values next to them, centered
    _controls2 = mo.vstack([
        mo.md(f"<div style='background: linear-gradient(135deg, {COLORS['primary']}10 0%, {COLORS['info']}10 100%); padding: 15px; border-radius: 8px; margin: 10px 0; text-align: center;'><h4 style='color: {COLORS['primary']}; margin-top: 0;'>🔤 Hidden Sentence</h4></div>"),
        mo.center(sentence_input),
        mo.md(f"<div style='background: linear-gradient(135deg, {COLORS['secondary']}10 0%, {COLORS['primary']}10 100%); padding: 15px; border-radius: 8px; margin: 20px 0 10px 0; text-align: center;'><h4 style='color: {COLORS['secondary']}; margin-top: 0;'>🎚️ Algorithm Parameters</h4></div>"),
        mo.center(population_slider),
        mo.center(generations_slider),
        mo.center(crossover_slider),
        mo.center(mutation_slider),
    ])

    _controls2
    return


@app.cell(hide_code=True)
def _(COLORS, mo, run_button):
    mo.md(rf"""
    <div style="background: linear-gradient(135deg, {COLORS['success']}10 0%, {COLORS['accent']}10 100%); border-left: 6px solid {COLORS['success']}; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <h3 style="color: {COLORS['success']}; margin-top: 0;">▶️ Execute the Algorithm</h3>

    <p>Click the button below to run the genetic algorithm with your parameters:</p>

    <h4 style="color: {COLORS['accent']};">🔄 Execution Flow</h4>

    <ol>
    <li>The algorithm initializes a random population of candidate solutions</li>
    <li>It evaluates each solution against the hidden sentence</li>
    <li>It evolves the population through selection, crossover, and mutation</li>
    <li>Progress is displayed every 10 generations</li>
    <li>Execution stops when a solution is found or max generations is reached</li>
    <li>Detailed performance metrics and visualizations are displayed</li>
    </ol>
    </div>
    """)

    # Display the run button centered
    _run_control = mo.center(run_button)
    _run_control
    return


@app.cell(hide_code=True)
def _(base, creator):
    # Clear any previous definitions
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual

    # Position and correct score (position and correct character)
    # Weights: (1.5, 1.0) gives 1.5x importance to correct positions over correct characters
    creator.create("FitnessMulti", base.Fitness, weights=(1.5, 1.0))

    # An individual is a list(char) and we want it to have the fitness attribute
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    return


@app.cell(hide_code=True)
def _(np, random, tools):
    def run_mastermind_ga(toolbox, mastermind, sentence_length, population_size=100, generations=200, cxpb=0.7, mutpb=0.3):
        """
        Run the genetic algorithm to solve the Mastermind puzzle.

        Args:
            toolbox: DEAP toolbox with registered functions
            mastermind: SentenceMastermind instance
            sentence_length: Length of the hidden sentence
            population_size: Number of individuals in the population
            generations: Number of generations to run
            cxpb: Crossover probability
            mutpb: Mutation probability

        Returns:
            best_individual: The best solution found
            logbook: Statistics about the evolution process
            history: History of best individuals
        """

        # Storage for detailed history
        history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_individual': [],
            'diversity': [],  # Track population diversity
            'convergence_rate': [],  # Track convergence rate
            'correct_positions': [],  # Track correct positions
            'correct_characters': []  # Track correct characters (wrong position)
        }

        # Create initial population (random individuals, not all combinations)
        pop = toolbox.population(n=population_size)

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Create statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("std", np.std, axis=0)

        # Create logbook to store statistics
        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        # Evaluate initial population
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)

        # Store initial history
        best_ind = tools.selBest(pop, 1)[0]
        history['generation'].append(0)
        history['best_fitness'].append(best_ind.fitness.values[0])
        history['avg_fitness'].append(record['avg'][0])
        history['best_individual'].append(''.join([str(char) for char in best_ind]))
        history['correct_positions'].append(best_ind.fitness.values[0])
        history['correct_characters'].append(best_ind.fitness.values[1])

        # Calculate initial diversity
        unique_individuals = len(set([''.join(ind) for ind in pop]))
        history['diversity'].append(unique_individuals / population_size)
        history['convergence_rate'].append(0.0)

        print(f"Gen {0:3}: Max={record['max'][0]:.2f}, Avg={record['avg'][0]:.2f}, Best={''.join([str(char) for char in best_ind])}")

        # Begin the evolution
        for gen in range(1, generations + 1):
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the population with the offspring
            pop[:] = offspring

            # Record statistics
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            # Store history
            best_ind = tools.selBest(pop, 1)[0]
            history['generation'].append(gen)
            history['best_fitness'].append(best_ind.fitness.values[0])
            history['avg_fitness'].append(record['avg'][0])
            history['best_individual'].append(''.join([str(char) for char in best_ind]))
            history['correct_positions'].append(best_ind.fitness.values[0])
            history['correct_characters'].append(best_ind.fitness.values[1])

            # Calculate diversity
            unique_individuals = len(set([''.join(ind) for ind in pop]))
            history['diversity'].append(unique_individuals / population_size)

            # Calculate convergence rate (improvement from previous generation)
            if gen > 1:
                improvement = history['best_fitness'][gen] - history['best_fitness'][gen-1]
                history['convergence_rate'].append(max(0, improvement))
            else:
                history['convergence_rate'].append(0.0)

            # Print progress every 10 generations or when solution is found
            if gen % 10 == 0 or best_ind.fitness.values[0] == sentence_length:
                print(f"Gen {gen:3}: Max={record['max'][0]:.2f}, Avg={record['avg'][0]:.2f}, Best={''.join([str(char) for char in best_ind])}")

            # Check if we found the solution
            if best_ind.fitness.values[0] == sentence_length:  # All positions correct
                print(f"Solution found in generation {gen}")
                print(f"Best solution: {''.join([str(char) for char in best_ind])}")
                break

        # Get the best individual
        best_ind = tools.selBest(pop, 1)[0]
        return best_ind, logbook, history
    return (run_mastermind_ga,)


@app.cell(hide_code=True)
def _(
    COLORS,
    SentenceMastermind,
    base,
    creator,
    crossover_slider,
    generations_slider,
    mo,
    mutation_slider,
    np,
    plt,
    population_slider,
    random,
    run_button,
    run_mastermind_ga,
    sentence_input,
    sns,
    time,
    tools,
):
    # Only run when the button is clicked
    if run_button.value:
        try:
            # Initialize the game
            mastermind = SentenceMastermind(sentence_input.value)
            sentence_length = mastermind.get_sentence_length()
            genes = mastermind.get_all_possible_char()

            # Show game initialization
            print(f"🎮 Game initialized with sentence: {sentence_input.value} (length: {sentence_length})")

            # Create toolbox
            toolbox = base.Toolbox()

            # Register attribute generator (for generating individual characters)
            # This function randomly selects a character from the available gene set
            toolbox.register("attr_char", random.choice, genes)

            # Register a function to create individuals (lists of characters of the right length)
            # Creates individuals as lists of characters with length equal to the hidden sentence
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_char, n=sentence_length)

            # Register population generator
            # Creates a population as a list of individuals
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Selection function (Tournament)
            # Selects individuals through tournament selection with tournament size of 3
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Crossover (uniform)
            # Combines two individuals by uniformly swapping characters with 50% probability
            toolbox.register("mate", tools.cxUniform, indpb=0.5)

            # Mutation (randomly change characters)
            # Mutates individuals by randomly changing characters with given probability
            def mutate_individual(individual, indpb):
                for i in range(len(individual)):
                    if random.random() < indpb:
                        individual[i] = random.choice(genes)
                return individual,

            toolbox.register("mutate", mutate_individual, indpb=0.1)

            # Evaluation function
            # Evaluates individuals by comparing them to the hidden sentence
            def evaluate_mastermind(individual):
                """
                Evaluate an individual (potential solution) against the hidden sentence.

                Args:
                    individual: A list of characters representing a guess

                Returns:
                    tuple: A tuple containing both fitness values (correct_position, correct_character)
                """
                # Convert the individual (list of characters) to a string
                # Fix: Make sure all elements are strings
                guess = ''.join([str(char) for char in individual])

                # Use the Mastermind game logic to check the guess
                correct_position, correct_character = mastermind.check_guess(guess)

                # Return both values as a tuple for multi-objective optimization
                # DEAP will try to maximize both values
                return (correct_position, correct_character)

            toolbox.register("evaluate", evaluate_mastermind)

            # Start timing
            start_time = time.time()

            # Run the genetic algorithm with slider values
            best_solution, logbook, history = run_mastermind_ga(
                toolbox=toolbox,
                mastermind=mastermind,
                sentence_length=sentence_length,
                population_size=population_slider.value,
                generations=generations_slider.value,
                cxpb=crossover_slider.value,
                mutpb=mutation_slider.value
            )

            # End timing
            end_time = time.time()
            execution_time = end_time - start_time

            # Define variables for summary
            best_individual = ''.join([str(char) for char in best_solution])
            hidden_sentence = mastermind.get_sentence()
            solution_found = best_solution.fitness.values[0] == sentence_length
            fitness_achieved = best_solution.fitness.values[0]
            best_fitness_achieved = max(history['best_fitness'])

            # Show summary with improved styling
            print("\n" + "="*60)
            print("EXECUTION SUMMARY")
            print("="*60)
            print(f"Hidden sentence:     {mastermind.get_sentence()}")
            print(f"Best solution found: {best_individual}")
            print(f"Fitness achieved:    {fitness_achieved} / {sentence_length}")
            print(f"Solution status:     {'SOLUTION FOUND' if solution_found else 'Solution not found (max generations reached)'}")
            print(f"Execution time:      {execution_time:.2f} seconds")
            print(f"Generations completed: {len(history['generation'])-1}")
            print(f"Population size:     {population_slider.value}")
            print(f"Best fitness achieved: {max(history['best_fitness'])}")
            print("="*60)

            # Set up better plotting style with unified color scheme
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create a custom color palette based on our scheme
            custom_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                           COLORS['success'], COLORS['warning'], COLORS['danger'], COLORS['info']]

            # Create and display meaningful graphs with explanations

            # 1. Convergence Analysis
            print("Creating convergence analysis graphs...")

            fig1 = plt.figure(figsize=(15, 5))

            # Convergence to Solution
            ax1 = plt.subplot(1, 3, 1)
            ax1.plot(history['generation'], history['best_fitness'], label='Best Fitness', linewidth=2, marker='o', markersize=4, color=custom_colors[0])
            ax1.axhline(y=sentence_length, color=custom_colors[5], linestyle='--', linewidth=2, label=f'Target Fitness ({sentence_length})')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Convergence to Solution', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Population Diversity Over Time
            ax2 = plt.subplot(1, 3, 2)
            ax2.plot(history['generation'], history['diversity'], color=custom_colors[2], linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Diversity Ratio')
            ax2.set_title('Population Diversity', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Convergence Rate (Improvement per Generation)
            ax3 = plt.subplot(1, 3, 3)
            ax3.bar(history['generation'][1:], history['convergence_rate'][1:], color=custom_colors[1], alpha=0.7)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Fitness Improvement')
            ax3.set_title('Convergence Rate', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # 2. Fitness Components Analysis
            print("Creating fitness components analysis graphs...")

            fig2 = plt.figure(figsize=(15, 5))

            # Correct Positions vs Correct Characters
            ax4 = plt.subplot(1, 3, 1)
            ax4.plot(history['generation'], history['correct_positions'], label='Correct Positions', linewidth=2, marker='o', markersize=4, color=custom_colors[3])
            ax4.plot(history['generation'], history['correct_characters'], label='Correct Characters\n(Wrong Position)', linewidth=2, marker='^', markersize=4, color=custom_colors[4])
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Count')
            ax4.set_title('Fitness Components Evolution', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Fitness Distribution (Min, Avg, Max)
            ax5 = plt.subplot(1, 3, 2)
            ax5.plot(history['generation'], [logbook[i]['min'][0] for i in range(len(logbook))], label='Min Fitness', linewidth=2, color=custom_colors[5])
            ax5.plot(history['generation'], [logbook[i]['avg'][0] for i in range(len(logbook))], label='Avg Fitness', linewidth=2, color=custom_colors[0])
            ax5.plot(history['generation'], [logbook[i]['max'][0] for i in range(len(logbook))], label='Max Fitness', linewidth=2, color=custom_colors[3])
            ax5.set_xlabel('Generation')
            ax5.set_ylabel('Fitness')
            ax5.set_title('Population Fitness Distribution', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # Population Statistics
            ax6 = plt.subplot(1, 3, 3)
            if len(logbook) > 0:
                final_stats = logbook[-1]
                ax6.bar(['Min', 'Avg', 'Max'], [final_stats['min'][0], final_stats['avg'][0], final_stats['max'][0]], 
                       color=[custom_colors[5], custom_colors[0], custom_colors[3]])
                ax6.set_ylabel('Fitness')
                ax6.set_title('Final Generation Statistics', fontsize=12, fontweight='bold')
            else:
                ax6.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
            ax6.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # 3. Solution Quality Analysis
            print("Creating solution quality analysis graphs...")

            fig3 = plt.figure(figsize=(15, 5))

            # Cumulative Best Fitness
            ax7 = plt.subplot(1, 3, 1)
            cumulative_best = np.maximum.accumulate(history['best_fitness'])
            ax7.plot(history['generation'], cumulative_best, color=custom_colors[6], linewidth=2, marker='o', markersize=4)
            ax7.set_xlabel('Generation')
            ax7.set_ylabel('Best Fitness So Far')
            ax7.set_title('Cumulative Best Fitness', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)

            # Moving Average of Fitness
            ax8 = plt.subplot(1, 3, 2)
            if len(history['best_fitness']) > 5:
                window_size = min(5, len(history['best_fitness']) // 4)
                if window_size > 1:
                    moving_avg = np.convolve(history['best_fitness'], np.ones(window_size)/window_size, mode='valid')
                    ax8.plot(history['generation'][:len(moving_avg)], moving_avg, color=custom_colors[4], linewidth=2, marker='s', markersize=4)
                    ax8.set_xlabel('Generation')
                    ax8.set_ylabel('Moving Average Fitness')
                    ax8.set_title('Fitness Smoothing', fontsize=12, fontweight='bold')
                else:
                    ax8.text(0.5, 0.5, 'Not enough data\nfor moving average', horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
            else:
                ax8.text(0.5, 0.5, 'Not enough data\nfor convergence analysis', horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
            ax8.grid(True, alpha=0.3)

            # Solution Quality Indicator
            ax9 = plt.subplot(1, 3, 3)
            achieved_fitness = max(history['best_fitness'])
            target_fitness = sentence_length
            quality_percentage = (achieved_fitness / target_fitness) * 100
            bar_color = custom_colors[3] if quality_percentage >= 90 else custom_colors[4] if quality_percentage >= 70 else custom_colors[5]
            ax9.bar(['Solution Quality'], [quality_percentage], color=bar_color)
            ax9.set_ylabel('Percentage (%)')
            ax9.set_ylim(0, 100)
            ax9.set_title('Solution Quality', fontsize=12, fontweight='bold')
            ax9.text(0, quality_percentage + 2, f'{quality_percentage:.1f}%', ha='center')
            ax9.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error running genetic algorithm: {str(e)}")
    else:
        print("Click the '🚀 Run Genetic Algorithm' button to execute the solver with your parameters.")
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.md(
        rf"""
    <div style="background: linear-gradient(135deg, {COLORS['primary']}10 0%, {COLORS['secondary']}10 100%); padding: 30px; border-radius: 15px; margin: 30px 0;">
    <h2 style="color: {COLORS['primary']}; text-align: center; margin-top: 0;">📚 Conclusion and Further Exploration</h2>

    <p style="font-size: 1.1em; text-align: center; max-width: 800px; margin: 0 auto 25px;">
    This notebook demonstrated how genetic algorithms can be used to solve the Mastermind puzzle efficiently. 
    By evolving a population of candidate solutions over multiple generations, we can find the hidden sentence 
    without having to enumerate all possible combinations.
    </p>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
        <div style="background-color: {COLORS['card']}; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h3 style="color: {COLORS['accent']};">🧠 Key Takeaways</h3>
            <ul>
            <li><strong>Effectiveness</strong> ⚡: Genetic algorithms are effective for large search spaces where brute force is infeasible</li>
            <li><strong>Parameter Sensitivity</strong> 🎛️: Tuning parameters (population size, crossover/mutation rates) significantly affects performance</li>
            <li><strong>Diversity Importance</strong> 🌈: Monitoring diversity helps prevent premature convergence to suboptimal solutions</li>
            <li><strong>Visualization Value</strong> 📈: Graphs provide insights into the algorithm's behavior and convergence patterns</li>
            </ul>
        </div>
        
        <div style="background-color: {COLORS['card']}; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h3 style="color: {COLORS['warning']};">🔧 Potential Improvements</h3>
            <ul>
            <li><strong>Elitism</strong> 👑: Preserving best solutions across generations</li>
            <li><strong>Adaptive Parameters</strong> 📊: Dynamically adjusting crossover/mutation rates</li>
            <li><strong>Different Selection Methods</strong> 🎯: Trying roulette wheel or rank-based selection</li>
            <li><strong>Advanced Operators</strong> 🔬: Implementing problem-specific crossover and mutation operators</li>
            </ul>
        </div>
        
        <div style="background-color: {COLORS['card']}; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h3 style="color: {COLORS['success']};">🎓 Educational Value</h3>
            <ul>
            <li><strong>Algorithm Understanding</strong> 🧠: Visualizing how genetic algorithms work step by step</li>
            <li><strong>Parameter Exploration</strong> 🔍: Experimenting with different settings to see their effects</li>
            <li><strong>Performance Analysis</strong> 📊: Understanding trade-offs between speed and accuracy</li>
            </ul>
        </div>
    </div>
    </div>

    <div style="background: linear-gradient(135deg, {COLORS['secondary']} 0%, {COLORS['primary']} 100%); padding: 25px; border-radius: 15px; text-align: center; color: white; margin: 30px 0;">
    <h2 style="margin-top: 0;">🙏 Thank You!</h2>
    <p style="font-size: 1.1em; max-width: 800px; margin: 0 auto;">
    Thank you for exploring genetic algorithms with this interactive notebook. 
    Feel free to experiment with different parameters and hidden sentences to deepen your understanding of evolutionary computation!
    </p>
    </div>
    """
    )
    return


if __name__ == "__main__":
    app.run()
