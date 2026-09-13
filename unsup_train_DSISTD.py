import subprocess
from dataclasses import dataclass
from typing import List

@dataclass
class PhaseConfig:
    start_epoch: int
    end_epoch: int
    step: int
    script: str

def generate_reset_epochs(phases: List[PhaseConfig]) -> List[int]:
    reset_epochs = []
    for phase in phases:
        phase_reset_epochs = list(range(phase.start_epoch, phase.end_epoch + phase.step, phase.step))
        phase_reset_epochs = [epoch for epoch in phase_reset_epochs if epoch <= phase.end_epoch]
        reset_epochs.extend(phase_reset_epochs)
    reset_epochs = sorted(set(reset_epochs))
    return reset_epochs

def generate_training_scripts(phases: List[PhaseConfig], reset_epochs: List[int], max_total_epochs: int) -> List[str]:
    training_scripts = []
    for phase in phases:
        num_steps = len([epoch for epoch in reset_epochs if phase.start_epoch <= epoch < phase.end_epoch])
        training_scripts.extend([phase.script] * num_steps)
    return training_scripts

def main():
    max_total_epochs = 100 
    phases = [
        PhaseConfig(start_epoch=0, end_epoch=5, step=5, script='train_phase0_DSISTD.py'),
        PhaseConfig(start_epoch=5, end_epoch=80, step=5, script='train_phase1_DSISTD.py'),
        PhaseConfig(start_epoch=80, end_epoch=100, step=20, script='train_sup_DSISTD.py'),
    ]
    reset_epochs = generate_reset_epochs(phases)
    print(f"Reset Epochs: {reset_epochs}")

    training_scripts = generate_training_scripts(phases, reset_epochs, max_total_epochs)
    print(f"Training Scripts: {training_scripts}")

    if len(training_scripts) != len(reset_epochs) - 1:
        raise ValueError("The length of training_scripts should be equal to the length of reset_epochs minus one.")

    total_epochs_trained = 0

    reset_epochs = sorted(reset_epochs)
    stages = reset_epochs + [max_total_epochs]

    for idx in range(len(reset_epochs)):
        start_epoch = reset_epochs[idx]
        end_epoch = stages[idx + 1]
        epochs_to_train = end_epoch - total_epochs_trained
        if epochs_to_train <= 0:
            continue  

        if total_epochs_trained + epochs_to_train > max_total_epochs:
            epochs_to_train = max_total_epochs - total_epochs_trained

        print(f"Starting training for {epochs_to_train} epochs until total epoch {total_epochs_trained + epochs_to_train}")

        try:
            script = training_scripts[idx] 
        except IndexError:
            raise IndexError(f"No training script has been specified for stage {idx}. Please ensure that the length of training_scripts equals the length of reset_epochs minus one.")

        print(f"Using {script} for epochs {total_epochs_trained} to {end_epoch}")

        result = subprocess.call([
            'python', script,
            '--stop_at_epoch', str(epochs_to_train),
            '--total_epochs_trained', str(total_epochs_trained)
        ])

        if result != 0:
            raise RuntimeError(f"The training script {script} failed during stage {total_epochs_trained} to {end_epoch}.")

        total_epochs_trained += epochs_to_train

        if total_epochs_trained >= max_total_epochs:
            print("Reached maximum total epochs.")
            break

if __name__ == "__main__":
    main()