# Audio-Visual Self-Other Distinction Task

This repository contains the experimental paradigm, source code, and design matrix for an **Audio-Visual Self-Other Distinction (SOD) Task**. The experiment was developed using **PsychoPy Builder**.

## Task Overview
The primary objective of this project is to investigate the cognitive mechanisms underlying Self-Other Distinction through cross-modal sensory processing. 

During the task, participants are simultaneously exposed to an auditory and a visual stimulus:
- **Auditory Stimulus (Self):** The sensory input directly heard and experienced by the participant.
- **Visual Stimulus (Other):** A visual cue representing the environment or sound that another individual is currently experiencing.
- **Congruency:** The combined audio-visual presentation can be either **congruent** (matching emotional valence) or **incongruent** (mismatched valence).

## Experimental Design & Tasks
Participants are required to perform emotional valence ratings based on dynamic perspective-shifting cues:
1. **Self-Perspective Trials:** Participants are explicitly cued to rate how pleasant or unpleasant the stimulus was for **themselves**.
2. **Other-Perspective Trials:** Participants are cued to infer and rate how pleasant or unpleasant the stimulus was for **the other person**.

### Recorded Behavioral Metrics
The task automatically logs high-precision behavioral responses for subsequent statistical analysis:
- **Valence Rating:** The participant's subjective evaluation on a continuous pleasantness/unpleasantness scale.
- **Reaction Time (RT):** The exact time elapsed (in milliseconds) between the onset of the rating scale and the participant's response registration.

## Repository Structure
The project directory is organized as follows to maintain a clean workflow and ensure correct path mapping within PsychoPy:

```text
├── Code/
│   ├── experiment.psyexp      # Core PsychoPy Builder file
│   └── experiment.py          # Auto-generated Python script from the Builder
├── stimulus/                  # Main folder containing all audio and image files together
└── loop.xlsx                  # Excel file containing trial conditions and design matrix
