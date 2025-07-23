## Preparation
Insert your Hugging Face token into the `login(token="input your huggingface token")` function in the code.

The dataset and model will be loaded via Hugging Face.

## Performing the Attack

### SDBA
`python Fed_Backdoor_SDBA.py --resume --attack_rounds 1 5`

### Other Attacks for Comparison

**Baseline**\
`python Fed_Backdoor.py --resume --attack_rounds 1 5`

**Neurotoxin**\
`python Fed_Backdoor_Neurotoxin.py --resume --attack_rounds 1 5`

**SDBA Gaussian**\
`python Fed_Backdoor_SDBA_Gaussian.py --resume --attack_rounds 1 5`
