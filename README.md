# Pseudo Package made for my master thesis
Contains a unique (for now) command line ``gw-setup-pipeline`` that performs coherent GW search based on data extracted from KNe.
***
## Intallation
Simply clone the repo, make sure to change your active directory to the cloned repo and execute
``pip install -e .``
## Documentation
### ``gw-setup-pipeline``
``gw-setup-pipeline`` has one *OBLIGATORY* argument, the path to the ``.yaml`` config file (examples of config files are provided in the ``example_file`` directory).
Additional (optionnal) arguments:
- ``--submit``: Automatically submit the pipeline to HTCondor after generation.
- ``--injection``: If true, will inject a fake signal inside the time windows to be searched, for testing purposes. The injection parameters will be read from the config file (under the 'Injection' section).
- ``--expected-trigger-time``: Expected trigger time to be searched, in gps format. Used only in the final trigger distribution plot.
- ``skip-search``: If true, will skip the search step and directly run the post-processing script. Only works if you already have triggers generated from a previous search run.
- ``--plot-spectrogram``: If true, will generate a spectrogram plot for the top trigger in the post-processing step. This can be useful for visually inspecting the trigger.
  - ``spectrogram-range``: vmin and vmax for the spectrogram plot. Only used if ``--plot-spectrogram`` is set.
- ``--monitor``: If true, will monitor the pipeline execution.
