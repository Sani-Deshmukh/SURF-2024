# Lunar Surface Image Registration

Iterative feature-based matching solution for Lunar surface image registration

## Installation

### Build Manually

```bash
# When using conda for convenience
conda env create --file environment.yml
conda activate lunarreg

# Build the package
python -m build

# Install with pip
pip install dist/<package wheel file>
```

## Usage

Currently no CLI tools are available.
A usage example can be found in `./tests/test.py` along with sample inputs.
