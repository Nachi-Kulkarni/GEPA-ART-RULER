import argparse
from datetime import datetime
import json
from pathlib import Path

import litellm
from terminal_bench.agents.terminus_1 import Terminus
from terminal_bench.agents.terminus_1 import AgentResult, FailureMode
from terminal_bench.agents.terminus_1 import Chat
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.dataset.dataset import Dataset

from gepa.adapters.terminal_bench_adapter import TerminusAdapter, TerminalBenchTask

from gepa import optimize


INSTRUCTION_PROMPT_PATH = (
    Path(__file__).parent / "prompt-templates/instruction_prompt.txt"
)


class TerminusWrapper(Terminus):
    def __init__(
        self,
        model_name: str,
        max_episodes: int = 50,
        api_base: str | None = None,
        **kwargs,
    ):
        self.PROMPT_TEMPLATE_PATH = (
            Path(__file__).parent / "prompt-templates/terminus.txt"
        )
        self.instruction_prompt = INSTRUCTION_PROMPT_PATH.read_text()
        super().__init__(model_name, max_episodes, api_base, **kwargs)

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ):
        chat = Chat(self._llm)

        initial_prompt = self.instruction_prompt + self._prompt_template.format(
            response_schema=self._response_schema,
            instruction=instruction,
            history="",
            terminal_state=session.capture_pane(),
        )

        self._run_agent_loop(initial_prompt, session, chat, logging_dir)

        return AgentResult(
            total_input_tokens=chat.total_input_tokens,
            total_output_tokens=chat.total_output_tokens,
            failure_mode=FailureMode.NONE,
            timestamped_markers=self._timestamped_markers,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--n_concurrent", type=int, default=6)
    args = parser.parse_args()

    initial_prompt_from_terminus = """
You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task instruction and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

For each response:
1. Analyze the current state based on any terminal output provided
2. Determine the next set of commands needed to make progress
3. Decide if you need to see the output of these commands before proceeding

Don't include markdown formatting.

Note that you operate directly on the terminal from inside a tmux session. Use tmux keystrokes like `C-x` or `Escape` to interactively navigate the terminal. If you would like to execute a command that you have written you will need to append a newline character to the end of your command.

For example, if you write "ls -la" you will need to append a newline character to the end of your command like this: `ls -la\n`.

One thing to be very careful about is handling interactive sessions like less, vim, or git diff. In these cases, you should not wait for the output of the command. Instead, you should send the keystrokes to the terminal as if you were typing them.
"""

    terminal_bench_dataset = Dataset(name="terminal-bench-core", version="head")
    terminal_bench_dataset.sort_by_duration()

    terminal_bench_tasks = terminal_bench_dataset._tasks[::-1]
    trainset = [
        TerminalBenchTask(task_id=task.name, model_name=args.model_name)
        for task in terminal_bench_tasks[30:50]
    ]
    valset = [
        TerminalBenchTask(task_id=task.name, model_name=args.model_name)
        for task in terminal_bench_tasks[:30]
    ]

    reflection_lm_name = "openai/gpt-5"
    reflection_lm = (
        lambda prompt: litellm.completion(
            model=reflection_lm_name,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
        )
        .choices[0]
        .message.content
    )

    optimized_results = optimize(
        seed_candidate={"instruction_prompt": initial_prompt_from_terminus},
        trainset=trainset,
        valset=valset,
        adapter=TerminusAdapter(n_concurrent=args.n_concurrent),
        reflection_lm=reflection_lm,
        use_wandb=True,
        max_metric_calls=400,
        reflection_minibatch_size=3,
        perfect_score=1,
        skip_perfect_score=False,
    )

    print(optimized_results.to_dict())
    # save to json
    with open(
        f"optimized_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w"
    ) as f:
        json.dump(optimized_results.to_dict(), f, indent=4)
