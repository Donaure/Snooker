"""Minimal protocol example: accept fouls and aim straight at the nearest Ball On."""
import json
import math
import sys

state = json.loads(sys.stdin.readline())
balls = {b['number']: b['position'] for b in state['balls'] if b['active']}
cue = balls[0]
target = min(state['legal_targets'], key=lambda n: math.dist(cue, balls[n]))
if state['stage'] == 'colour':
    print(json.dumps({'op': 'nominate_colour', 'ball': target}), flush=True)
    state = json.loads(sys.stdin.readline())
point = balls[target]
print(json.dumps([math.atan2(point[1]-cue[1], point[0]-cue[0]), 1.]), flush=True)
