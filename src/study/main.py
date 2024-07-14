#!/usr/bin/env python
from study.crew import StudyCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'AI LLMs'
    }
    StudyCrew().crew().kickoff(inputs=inputs)