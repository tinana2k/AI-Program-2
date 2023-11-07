import random
from itertools import product
from pprint import pprint
import numpy as np

# Define the facilitators, activities, rooms, and times
facilitators = ["Lock", "Glen", "Banks", "Richards", "Shaw", "Singer", "Uther", "Tyler", "Numen", "Zeldin"]

activities = {
    "SLA100A": {"enrollment": 50, "preferred_facilitators": ["Glen", "Lock", "Banks", "Zeldin"],
                "other_facilitators": ["Numen", "Richards"]},
    "SLA100B": {"enrollment": 50, "preferred_facilitators": ["Glen", "Lock", "Banks", "Zeldin"],
                "other_facilitators": ["Numen", "Richards"]},
    "SLA191A": {"enrollment": 50, "preferred_facilitators": ["Glen", "Lock", "Banks", "Zeldin"],
                "other_facilitators": ["Numen", "Richards"]},
    "SLA191B": {"enrollment": 50, "preferred_facilitators": ["Glen", "Lock", "Banks", "Zeldin"],
                "other_facilitators": ["Numen", "Richards"]},
    "SLA201": {"enrollment": 50, "preferred_facilitators": ["Glen", "Banks", "Zeldin", "Shaw"],
               "other_facilitators": ["Numen", "Richards", "Singer"]},
    "SLA291": {"enrollment": 50, "preferred_facilitators": ["Lock", "Banks", "Zeldin", "Singer"],
               "other_facilitators": ["Numen", "Richards", "Shaw", "Tyler"]},
    "SLA303": {"enrollment": 60, "preferred_facilitators": ["Glen", "Zeldin", "Banks"],
               "other_facilitators": ["Numen", "Singer", "Shaw"]},
    "SLA304": {"enrollment": 25, "preferred_facilitators": ["Glen", "Banks", "Tyler"],
               "other_facilitators": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"]},
    "SLA394": {"enrollment": 20, "preferred_facilitators": ["Tyler", "Singer"],
               "other_facilitators": ["Richards", "Zeldin"]},
    "SLA451": {"enrollment": 100, "preferred_facilitators": ["Tyler", "Singer", "Shaw"],
               "other_facilitators": ["Zeldin", "Uther", "Richards", "Banks"]},
    "SLA449": {"enrollment": 60, "preferred_facilitators": ["Tyler", "Singer", "Shaw"],
               "other_facilitators": ["Zeldin", "Uther"]},
}

rooms = {
    "Slater 003": 45,
    "Roman 216": 30,
    "Loft 206": 75,
    "Roman 201": 50,
    "Loft 310": 108,
    "Beach 201": 60,
    "Beach 301": 75,
    "Logos 325": 450,
    "Frank 119": 60,
}

times = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]


# Generate a random schedule
def generate_random_schedule(activities, rooms, times, facilitators):
    schedule = {}
    for activity in activities:
        room = random.choice(list(rooms.keys()))
        time = random.choice(times)
        facilitator = random.choice(facilitators)
        schedule[activity] = {"room": room, "time": time, "facilitator": facilitator}
    return schedule


# Calculate difference in time
def time_difference(start_time, end_time):
    start_time = times.index(start_time)
    end_time = times.index(end_time)
    return end_time - start_time


def fitness(schedule):
    fitness_scores = {}

    # Fitness for each activity
    for activity, details in schedule.items():
        fitness_scores[activity] = 0

        # Check for scheduling conflicts (same time and room)
        for other_activity, other_details in schedule.items():
            if activity != other_activity and \
                    details['time'] == other_details['time'] and \
                    details['room'] == other_details['room']:
                fitness_scores[activity] -= 0.5

        # Check room size
        if rooms[details['room']] < activities[activity]['enrollment']:
            fitness_scores[activity] -= 0.5
        elif rooms[details['room']] > 3 * activities[activity]['enrollment']:
            fitness_scores[activity] -= 0.2
        elif rooms[details['room']] > 6 * activities[activity]['enrollment']:
            fitness_scores[activity] -= 0.4
        else:
            fitness_scores[activity] += 0.3

        # Check facilitator
        if details['facilitator'] in activities[activity]['preferred_facilitators']:
            fitness_scores[activity] += 0.5
        elif details['facilitator'] in activities[activity]['other_facilitators']:
            fitness_scores[activity] += 0.2
        else:
            fitness_scores[activity] -= 0.1

        # Check facilitator load and activity-specific adjustments
        facilitator_load = 0
        for other_activity, other_details in schedule.items():
            if activity != other_activity and \
                    details['facilitator'] == other_details['facilitator']:
                facilitator_load += 1
                time_diff = time_difference(details['time'], other_details['time'])
                if time_diff <= 4:
                    fitness_scores[activity] -= 0.5
                if time_diff == 0:
                    fitness_scores[activity] -= 0.5
                if time_diff == 1:
                    fitness_scores[activity] += 0.25
                if time_diff == 0:
                    fitness_scores[activity] -= 0.25
        if facilitator_load == 1 or facilitator_load == 2:
            fitness_scores[activity] -= 0.4
        if details['facilitator'] == 'Tyler' and facilitator_load < 2:
            fitness_scores[activity] += 0.4

        # Activity-specific adjustments
        if activity == 'SLA 101':
            if details['section'] == 'A':
                section_b = schedule.get('SLA 101', {}).get('B')
                if section_b and time_difference(details['time'], section_b['time']) > 4:
                    fitness_scores[activity] += 0.5
                elif section_b and details['time'] == section_b['time']:
                    fitness_scores[activity] -= 0.5
            elif details['section'] == 'B':
                section_a = schedule.get('SLA 101', {}).get('A')
                if section_a and time_difference(details['time'], section_a['time']) > 4:
                    fitness_scores[activity] += 0.5
                elif section_a and details['time'] == section_a['time']:
                    fitness_scores[activity] -= 0.5

        if activity == 'SLA 191':
            if details['section'] == 'A':
                section_b = schedule.get('SLA 191', {}).get('B')
                if section_b and time_difference(details['time'], section_b['time']) > 4:
                    fitness_scores[activity] += 0.5
                elif section_b and details['time'] == section_b['time']:
                    fitness_scores[activity] -= 0.5
                elif section_b and time_difference(details['time'], section_b['time']) == 1:
                    fitness_scores[activity] += 0.25
            elif details['section'] == 'B':
                section_a = schedule.get('SLA 191', {}).get('A')
                if section_a and time_difference(details['time'], section_a['time']) > 4:
                    fitness_scores[activity] += 0.5
                elif section_a and details['time'] == section_a['time']:
                    fitness_scores[activity] -= 0.5
                elif section_a and time_difference(details['time'], section_a['time']) == 1:
                    fitness_scores[activity] += 0.25

            # Check for consecutive time slots
            section_101 = schedule.get('SLA 101', {})
            section_101_a = section_101.get('A')
            section_101_b = section_101.get('B')
            if (details['section'] == 'A' and section_101_b) or \
                    (details['section'] == 'B' and section_101_a):
                time_diff = time_difference(details['time'], section_101_b['time']) \
                    if details['section'] == 'A' else \
                    time_difference(details['time'], section_101_a['time'])
                if time_diff == 1:
                    fitness_scores[activity] += 0.25
                elif time_diff == 0:
                    fitness_scores[activity] -= 0.25
                if 'Roman' not in details['room'] and 'Beach' not in details['room'] and \
                        ('Roman' in section_101_a['room'] or 'Beach' in section_101_a['room'] or \
                         'Roman' in section_101_b['room'] or 'Beach' in section_101_b['room']):
                    fitness_scores[activity] -= 0.4
    return sum(fitness_scores.values())


def crossover(schedule1, schedule2):
    # Get a list of all course codes in both schedules
    course_codes = list(set(schedule1.keys()) | set(schedule2.keys()))

    # Choose a random course to perform crossover on
    crossover_course = random.choice(course_codes)

    # Swap the values of the chosen course between the two schedule

    temp = schedule1[crossover_course]
    schedule1[crossover_course] = schedule2[crossover_course]
    schedule2[crossover_course] = temp

    return schedule1, schedule2


def selection(population, fitness_scores):
    # Convert fitness scores to probabilities using softmax normalization
    probs = np.exp(fitness_scores) / np.sum(np.exp(fitness_scores))
    # Select parents using roulette wheel selection
    parent_indices = np.random.choice(len(population), size=2, p=probs, replace=False)
    return parent_indices[0], parent_indices[1]


def mutate(schedule, mutation_rate):
    # randomly select a course from the schedule
    course_id = random.choice(list(schedule.keys()))
    course = schedule[course_id]

    entrophy = random.random()

    if entrophy < mutation_rate:
        # randomly choose whether to change the facilitator, room, or time for the course
        mutation_type = random.choice(['facilitator', 'room', 'time'])

        # randomly select a new value for the selected attribute
        if mutation_type == 'facilitator':
            new_value = random.choice(
                ["Lock", "Glen", "Banks", "Richards", "Shaw", "Singer", "Uther", "Tyler", "Numen", "Zeldin"])
        elif mutation_type == 'room':
            new_value = random.choice(
                ["Frank 119", "Loft 310", "Roman 216", "Logos 325", "Beach 201", "Slater 003", "Beach 301", "Loft 206",
                 "Roman 201"])
        else:  # mutation_type == 'time'
            new_value = random.choice(["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"])
        # create a copy of the course with the mutated attribute
        new_course = dict(course)
        new_course[mutation_type] = new_value
        # create a copy of the schedule with the mutated course
        new_schedule = dict(schedule)
        new_schedule[course_id] = new_course
        return new_schedule
    else:
        return schedule


def main():
    # Define the parameters for the genetic algorithm
    POPULATION_SIZE = 510
    NUM_GENERATIONS = 100
    MUTATION_RATE = 0.05

    init_population = [generate_random_schedule(activities, rooms, times, facilitators) for i in range(POPULATION_SIZE)]
    fitness_scores = [fitness(schedule) for schedule in init_population]
    # Main loop for the genetic algorithm
    for i in range(NUM_GENERATIONS):
        # Select a subset of the population based on their fitness

        selected_indices = selection(init_population, fitness_scores)
        selected_population = [init_population[index] for index in selected_indices]
        # pprint(selected_population)
        # Apply genetic operators to the selected individuals to create a new population
        next_population = []
        while len(next_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected_population, 2)
            # print(parent1, parent2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)
            next_population.append(child1)
            next_population.append(child2)
        # Evaluate the fitness of each individual in the new population
        new_fitness = [fitness(schedule) for schedule in next_population]

        # Replace the old population with the new population
        init_population = next_population
        fitness_scores = new_fitness
        if max(fitness_scores) > 4.0:
            break
        else:
            NUM_GENERATIONS += 1
    # Print the best schedule and its fitness
    best_index = max(range(POPULATION_SIZE), key=lambda x: fitness_scores[x])
    best_schedule = init_population[best_index]
    best_fitness = fitness_scores[best_index]
    print("Best fitness in generation {}: {}".format(i, best_fitness))
    pprint(best_schedule)


main()