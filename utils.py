# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A module with util functions."""
import sys
import json
import math

from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams.update({
    # Set the plot left margin so that the labels are visible.
    'figure.subplot.left': 0.3,

    # Hide the bottom toolbar.
    'toolbar': 'None'
})


class Plotter(object):
  """An util class to display the classification results."""

  _PAUSE_TIME = 0.05
  """Time for matplotlib to wait for UI event."""

  def __init__(self) -> None:
    fig, self._axes = plt.subplots()
    fig.canvas.manager.set_window_title('Audio classification')

    # Stop the program when the ESC key is pressed.
    def event_callback(event):
      if event.key == 'escape':
        sys.exit(0)

    fig.canvas.mpl_connect('key_press_event', event_callback)

    plt.show(block=False)

  # TODO(khanhlvg): Add type hint for result once ClassificationResult added
  # to tflite_support.task.processor module
  def plot(self, result, ws) -> None:
    """Plot the audio classification result.

    Args:
      result: Classification results returned by an audio classification
        model.
    """
    # Clear the axes
    self._axes.cla()
    self._axes.set_title('Press ESC to exit.')
    self._axes.set_xlim((0, 1))

    # Plot the results so that the most probable category comes at the top.
    classification = result.classifications[0]
    print(classification)
    # transformation du tableau d'instance de Clategory en tableau d'objet {index, score || 0, category_name}
    classificationJson = []
    for i in range(len(classification.categories)):
      # if classification.categories[i].score is NaN, set it to 0
      if math.isnan(classification.categories[i].score):
        classificationJson.append({'index': classification.categories[i].index, 'score': 0, 'category_name': classification.categories[i].category_name})
      else:
        classificationJson.append({'index': classification.categories[i].index, 'score': classification.categories[i].score, 'category_name': classification.categories[i].category_name})
    print(classificationJson)
    message = {'type': 'new-alert', 'sender': 'beacon', 'payload': {'reportings': classificationJson, 'location': 'montagne-du-vuache'}}
    ws.send(json.dumps(message))
    label_list = [category.category_name for category in classification.categories]
    score_list = [category.score for category in classification.categories]
    self._axes.barh(label_list[::-1], score_list[::-1])

    # Wait for the UI event.
    plt.pause(self._PAUSE_TIME)
