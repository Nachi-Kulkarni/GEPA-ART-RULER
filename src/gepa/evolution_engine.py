import random
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class PromptCandidate:
    text: str
    fitness_score: float
    generation: int
    parent_info: str
    test_results: List[Dict]

class GEPAEvolutionEngine:
    def __init__(self, model_interface, evaluator, population_size: int = 6):
        self.model = model_interface
        self.evaluator = evaluator
        self.population_size = population_size
        self.generation_count = 0
        self.evolution_log = []
    
    def create_initial_population(self, base_prompt: str) -> List[PromptCandidate]:
        """Create starting population with variations of base prompt"""
        population = []
        
        # Keep the original as one candidate
        population.append(PromptCandidate(
            text=base_prompt,
            fitness_score=0.0,
            generation=0,
            parent_info="original",
            test_results=[]
        ))
        
        # Create variations using the model itself
        for i in range(self.population_size - 1):
            variation = self.mutate_prompt(base_prompt, f"initial_variation_{i}")
            population.append(PromptCandidate(
                text=variation,
                fitness_score=0.0,
                generation=0,
                parent_info=f"variation_of_original_{i}",
                test_results=[]
            ))
        
        return population
    
    def mutate_prompt(self, prompt: str, mutation_id: str) -> str:
        """Use the model to create an improved version of the prompt"""
        
        mutation_instruction = f"""
Analyze this competitive programming prompt and create an improved version:

ORIGINAL PROMPT:
{prompt}

IMPROVEMENT GOALS:
- Make algorithmic thinking more systematic
- Improve C++ coding guidance
- Add better error handling instructions  
- Enhance constraint analysis
- Make instructions clearer and more actionable

Create an improved version of the prompt that keeps the same structure but enhances the quality of guidance:
"""
        
        # Use your model to generate the mutation
        response = self.model.generate(mutation_instruction, max_tokens=800)
        
        # Extract the improved prompt from the response
        # You'll need to implement this based on your model interface
        improved_prompt = self._extract_improved_prompt(response)
        
        return improved_prompt
    
    def crossover_prompts(self, parent1: PromptCandidate, parent2: PromptCandidate) -> str:
        """Combine two high-performing prompts"""
        
        crossover_instruction = f"""
Combine the best aspects of these two competitive programming prompts:

PROMPT A (Score: {parent1.fitness_score:.3f}):
{parent1.text}

PROMPT B (Score: {parent2.fitness_score:.3f}):
{parent2.text}

Create a hybrid prompt that incorporates the strongest elements from both:
- Keep the best instructional patterns from A
- Integrate the best guidance strategies from B
- Maintain coherent structure
- Focus on competitive programming effectiveness

Hybrid prompt:
"""
        
        response = self.model.generate(crossover_instruction, max_tokens=800)
        return self._extract_improved_prompt(response)
    
    def evaluate_population(self, population: List[PromptCandidate], 
                          test_problems: List[Dict]) -> None:
        """Test each prompt on the problem set and assign fitness scores"""
        
        for candidate in population:
            print(f"Evaluating candidate: {candidate.parent_info}")
            
            total_score = 0
            results = []
            
            for problem in test_problems:
                # Test this prompt on this problem
                success = self._test_prompt_on_problem(candidate.text, problem)
                results.append({
                    "problem_id": problem["id"],
                    "success": success
                })
                total_score += 1 if success else 0
            
            candidate.fitness_score = total_score / len(test_problems)
            candidate.test_results = results
            
            print(f"  → Score: {candidate.fitness_score:.3f}")
    
    def _test_prompt_on_problem(self, prompt: str, problem: Dict) -> bool:
        """Test a specific prompt on a specific problem"""
        
        # Create full prompt for this problem
        full_prompt = f"{prompt}\n\n{problem['prompt']}"
        
        try:
            # Generate solution
            response = self.model.generate(full_prompt, max_tokens=3200)
            
            # Extract code
            from utils.code_parser import CodeParser
            parser = CodeParser()
            language, code = parser.get_main_solution(response)
            
            # Evaluate with OJBench  
            result = self.evaluator.evaluate_solution(
                problem["id"], code, language
            )
            
            return result["success"]
            
        except Exception as e:
            print(f"    Error testing problem {problem['id']}: {e}")
            return False  # Failed to generate or evaluate solution
    
    def evolve_generation(self, population: List[PromptCandidate]) -> List[PromptCandidate]:
        """Create the next generation through selection, crossover, and mutation"""
        
        # Sort by fitness (best first)
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Log this generation
        gen_info = {
            "generation": self.generation_count,
            "best_score": population[0].fitness_score,
            "average_score": sum(p.fitness_score for p in population) / len(population),
            "population_diversity": len(set(p.text for p in population))
        }
        self.evolution_log.append(gen_info)
        
        # Keep top performers (elitism)
        next_generation = []
        elite_count = 2  # Keep best 2
        next_generation.extend(population[:elite_count])
        
        # Generate offspring
        while len(next_generation) < self.population_size:
            if random.random() < 0.6:  # 60% crossover
                parent1, parent2 = random.sample(population[:4], 2)  # Select from top 4
                offspring_text = self.crossover_prompts(parent1, parent2)
                parent_info = f"crossover_{parent1.parent_info}_{parent2.parent_info}"
            else:  # 40% mutation
                parent = random.choice(population[:3])  # Select from top 3
                offspring_text = self.mutate_prompt(parent.text, f"mutation_{parent.parent_info}")
                parent_info = f"mutated_{parent.parent_info}"
            
            offspring = PromptCandidate(
                text=offspring_text,
                fitness_score=0.0,
                generation=self.generation_count + 1,
                parent_info=parent_info,
                test_results=[]
            )
            next_generation.append(offspring)
        
        self.generation_count += 1
        return next_generation
    
    def _extract_improved_prompt(self, model_response: str) -> str:
        """Extract the actual prompt from the model's response"""
        # This is a simple extraction - you might need to make this more sophisticated
        lines = model_response.split('\n')
        
        # Look for lines that seem like prompt content
        prompt_lines = []
        in_prompt = False
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['prompt:', 'improved', 'version:']):
                in_prompt = True
                continue
            
            if in_prompt:
                if line.strip():
                    prompt_lines.append(line)
                elif len(prompt_lines) > 5:  # Stop after we have substantial content
                    break
        
        return '\n'.join(prompt_lines).strip()
    
    def run_evolution(self, base_prompt: str, test_problems: List[Dict], 
                     max_generations: int = 4) -> PromptCandidate:
        """Run the complete evolution process"""
        
        print(f"🧬 Starting GEPA evolution with {len(test_problems)} test problems")
        print(f"Population size: {self.population_size}, Max generations: {max_generations}")
        
        # Create initial population
        population = self.create_initial_population(base_prompt)
        
        for generation in range(max_generations):
            print(f"\n--- Generation {generation + 1} ---")
            
            # Evaluate current population
            self.evaluate_population(population, test_problems)
            
            # Show best performer
            best = max(population, key=lambda x: x.fitness_score)
            print(f"Best score this generation: {best.fitness_score:.3f}")
            
            # Evolve to next generation (except on last iteration)
            if generation < max_generations - 1:
                population = self.evolve_generation(population)
        
        # Return the best prompt found
        final_best = max(population, key=lambda x: x.fitness_score)
        
        print(f"\n🏆 Evolution complete!")
        print(f"Best final score: {final_best.fitness_score:.3f}")
        
        return final_best