# Data, licence and attribution

## Source

GutSignal Lab uses the public **Bowel sounds** dataset published by Robert
Nowak on Kaggle:

https://www.kaggle.com/datasets/robertnowak/bowel-sounds

The dataset is associated with:

Ficek J, Radzikowski K, Nowak JK, Yoshie O, Walkowiak J, Nowak R. *Analysis of
Gastrointestinal Acoustic Activity Using Deep Neural Networks.* Sensors.
2021;21(22):7602. https://doi.org/10.3390/s21227602

## Licence

The source dataset is published under the Creative Commons
Attribution-NonCommercial 4.0 International licence:

https://creativecommons.org/licenses/by-nc/4.0/

This project is therefore an independent, non-commercial research and
application portfolio experiment. The three WAV files in `demo_samples/` are
deterministic synthetic signals created for the interface. They contain no
participant audio. The raw dataset and row-level participant or recording
identifiers are not redistributed in this repository.

The original source code in this repository is licensed separately under the
MIT License. That software licence does not replace or broaden the permissions
attached to the source dataset or any trained artefact derived from it. The
bundled baseline model is retained for
this non-commercial portfolio demonstration; any reuse requires a separate
review of the dataset terms.

## Participant grouping caveat

The dataset reports 19 participants. Recording filenames contain suffixes from
`a` to `s`, also yielding 19 groups. GutSignal Lab uses those suffixes as the
participant-group key during grouped cross-validation.

That mapping is a reasonable inference from the public file structure, but it
has not been independently confirmed with the dataset authors. The project
therefore describes these as **inferred participant groups** and makes the
assumption visible wherever the validation result is discussed.

## Responsible interpretation

The labels describe annotated sound events in short recordings. They are not
diagnoses, symptom labels or measures of digestive health. Model probabilities
must not be presented as clinical risk, wellness scores or evidence that the
system generalises to consumer hardware or ordinary daily life.
