# Reference EIS curves from the held-out test split

The `test_*.txt` files each contain one complete UART frame: 121 **raw** features,
ordered `r_1,i_1,...,r_60,i_60,temperature`, terminated by a newline. Do not send
`cell` or the reference `soh`. The ESP32 applies the fitted scaler and int8
conversion itself.

`reference_test_samples.json` records the dataset SHA-256, seed 42, split ratios
70%/15%/remaining, scaler coefficients, feature order, test index, original CSV
row index (zero based, excluding the header) and true SOH. These samples were
selected by running the actual training split function against `sample/df_clean.csv`,
using a dataset with `cell` removed and dropping missing rows. They are not arbitrary rows from the
original dataset.

These are reference examples for that exact split and dataset. For a trained
job, prefer **Connect ESP32** in the frontend: its examples and scaler
come from the actual job, including any different seed or dataset.
A true SOH of 80 implies a red reference label, but the LED is controlled by the
model's prediction, which may differ.

These frames require a model trained without `cell`. For other input columns,
use the samples exported by the matching training job.
