import numpy as np

def decode_to_coordinates(y_pred, orig_w, orig_h, IMGS_PER_INSTANCE, WIDTH, HEIGHT, GRID_ROWS, GRID_COLS):
    y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)

    conf, xoff, yoff = np.split(y_pred, 3, axis=-1)

    conf = np.squeeze(conf, -1)
    xoff = np.squeeze(xoff, -1)
    yoff = np.squeeze(yoff, -1)

    coords = []

    for b in range(conf.shape[0]):
        for t in range(conf.shape[1]):

            r, c = np.unravel_index(
                np.argmax(conf[b, t]),
                (GRID_ROWS, GRID_COLS)
            )

            if conf[b, t, r, c] < 0.5:
                coords.append((0, 0))
                continue

            x = (c + xoff[b, t, r, c]) * (WIDTH / GRID_COLS)
            y = (r + yoff[b, t, r, c]) * (HEIGHT / GRID_ROWS)

            coords.append((
                int(x / WIDTH * orig_w),
                int(y / HEIGHT * orig_h)
            ))

    return coords