import sys
import importlib

COMMAND_MODULES = [
    'sft',
    'generate',
    'openai_generate',
    'score',
]

if __name__ == '__main__':
    for command in COMMAND_MODULES:
        if command in sys.argv:
            sys.argv.remove(command)
            module_name = f'llm_scripts.{command}'
            module = importlib.import_module(module_name)
            module.main()
            break
    else:
        raise ValueError(
            f'no command found in args: {sys.argv}, '
            f'valid commands: {", ".join(COMMAND_MODULES.keys())}'
        )
