import sys
import matplotlib.pyplot as plt
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path).astype({"epoch": "int32"})

fig, (ax_loss, ax_on_diag, ax_off_diag) = plt.subplots(3, 1)

ax_loss.plot(df["loss"])
ax_loss.set_title("Loss")

ax_on_diag.plot(df["on_diag"])
ax_on_diag.set_title("On Diagonal")

ax_off_diag.plot(df["off_diag"])
ax_off_diag.set_title("Off Diagonal")
plt.savefig("log.png")
plt.show()

