import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.typing as npt

path = sys.argv[1]
df: pd.DataFrame = pd.read_csv(path).astype({"epoch": "int32"})

fig, ax = plt.subplots(3, 1, figsize=(8, 10))
ax: npt.NDArray

columns = ["loss", "on_diag", "off_diag"]
titles = ["Loss", "On Diagonal", "Off Diagonal"]

for i, (col, title) in enumerate(zip(columns, titles)):
    a = ax[i]
    y = df[col].to_numpy()
    x = np.arange(len(y))
    log_y = np.log(y)

    # raw data on left axis
    line1, = a.plot(x, y, color="tab:blue", label=col)
    a.set_ylabel(col, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a.set_title(title)

    # log data + linear fit on right axis
    a2 = a.twinx()
    line2, = a2.plot(x, log_y, color="tab:orange", label=f"log({col})")

    coeffs = np.polyfit(x, log_y, 1)
    fit = np.polyval(coeffs, x)
    residuals = log_y - fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    line3, = a2.plot(x, fit, color="tab:green", linestyle="--",
                      label=f"linear fit (R²={r_squared:.3f})")

    a2.set_ylabel(f"log({col})", color="tab:orange")
    a2.tick_params(axis="y", labelcolor="tab:orange")

    a.legend(handles=[line1, line2, line3], loc="upper right", fontsize=8)

    print(f"{col}: slope={coeffs[0]:.4f}, intercept={coeffs[1]:.4f}, R²={r_squared:.4f}")

fig.tight_layout()
plt.savefig("log.png")
plt.show()
