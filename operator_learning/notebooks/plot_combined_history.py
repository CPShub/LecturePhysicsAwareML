# %% Imports
from matplotlib import pyplot as plt
import klax

colors = { 
    'green':        '#16a48a',
    'lightblue':    '#688fc6',
    'darkblue':     '#435384',
    'grey':         '#cccccc',
    'orange':       '#f6a315',
    'red':          '#c24c4c',
    'black':        '#000000',
}

# %% Load training history

history_with = klax.HistoryCallback.load("fno_history_with_enrichment.pkl")
history_without = klax.HistoryCallback.load("fno_history_no_enrichment.pkl")

_, ax = plt.subplots()
ax.set(
    xlabel="Step",
    ylabel="Loss",
    yscale="log",
    title="Training History",
)
ax.grid(True)
history_with.plot(ax=ax, 
                  loss_options={"color": colors['lightblue'], "label": "Loss with enrichment", "lw":3},
                  val_loss_options={"color": colors['orange'], "label": "Validation loss with enrichment", "lw":3},
                  )
history_without.plot(ax=ax,)
ax.legend()
plt.tight_layout()
plt.show()