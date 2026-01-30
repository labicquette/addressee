from pathlib import Path


from spidr.models import spidr_base

def load_spidr():

    spidr_model = spidr_base().to("cuda")
    return spidr_model



