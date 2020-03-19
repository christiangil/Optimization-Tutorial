import numpy as np
import os
import pandas as pd
import sys
import rebound

def collision(reb_sim, col):
    reb_sim.contents._status = 5
    return 0

sim_names = "new_sims/"

nsim = int(sys.argv[1])
sa_name = ("id_%5.0d.bin"%nsim).replace(" ","0")

sim = rebound.SimulationArchive(sim_names + sa_name)[0]
P1 = sim.particles[1].P
try:
    sim.integrate(1e9 * P1, exact_finish_time=0)
    res = True
except:
    res = False

df = pd.DataFrame(data=[sa_name], columns=["sim"])
df["stability"] = res
df.to_csv(sim_names  + sa_name + "_stab.csv")
