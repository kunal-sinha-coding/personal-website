"""Read the saved chronological aggregate evaluations and plot their running maximum.
Mark each new maximum with its evaluation number and accuracy, then check that
all labels fit without overlap and save the PNG beside the source data.
Run this file with Python and Matplotlib to rebuild the blog figure.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the original 70 full test evaluations from the aggregate data snapshot.
directory = Path(__file__).resolve().parent
with (directory / 'when2call-full-test-accuracy.csv').open() as handle:
    records = list(csv.DictReader(handle))
assert len(records) == 70
assert all(int(record['examples']) == 3652 for record in records)

# Compute the cumulative maximum and retain each evaluation that improves it.
maxima = []
improvements = []
best = 0.0
for index, record in enumerate(records, 1):
    assert int(record['evaluation']) == index
    accuracy = float(record['accuracy']) * 100
    if accuracy > best:
        best = accuracy
        improvements.append((index, accuracy))
    maxima.append(best)
assert [index for index, _ in improvements] == [1, 2, 3, 6, 8, 25, 28, 30, 35, 61, 64, 70]

# Draw the original step curve and its twelve record markers.
fig, ax = plt.subplots(figsize=(12, 6.2), layout='constrained')
ax.step(range(1, 71), maxima, where='post', color='#2563eb', linewidth=2.4)
ax.scatter(*zip(*improvements), color='#dc2626', s=34, zorder=3, label='New record')

# Stagger labels and connect them to their points so nearby records stay distinct.
positions = [(8, 12), (12, -40), (8, 22), (26, -44), (42, 20),
             (-8, -42), (-24, 30), (40, -30), (10, 24), (-12, -42),
             (-8, 22), (-12, 18)]
labels = []
for (index, accuracy), (dx, dy) in zip(improvements, positions):
    label = ax.annotate(
        f'{index}: {accuracy:.2f}%', (index, accuracy), xytext=(dx, dy),
        textcoords='offset points', ha='right' if dx < 0 else 'left',
        va='bottom' if dy > 0 else 'top', fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9, 'pad': 2},
        arrowprops={'arrowstyle': '-', 'color': '#777777', 'linewidth': 0.7},
    )
    labels.append(label)

# Set the axis range and explain the meaning of the numeric labels.
ax.set_xlim(0, 72)
ax.set_ylim(50, 77)
ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70])
ax.set_xlabel('Completed full-test evaluation number')
ax.set_ylabel('Best accuracy so far (%)')
ax.set_title('Cumulative best accuracy on the full When2Call test set')
ax.grid(True, linestyle=':', alpha=0.35)
ax.legend(loc='lower right', frameon=True, fontsize=9)
fig.supxlabel('Each label shows the evaluation number and its record accuracy.', fontsize=9)

# Check the rendered text bounds before saving the figure.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bounds = [label.get_bbox_patch().get_window_extent(renderer) for label in labels]
for index, box in enumerate(bounds):
    assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
    assert all(not box.overlaps(other) for other in bounds[index + 1:])
fig.savefig(directory / 'when2call-full-test-accuracy.png', dpi=150, facecolor='white')
print(f'Saved {len(records)} evaluations with {len(labels)} visible record labels.')
