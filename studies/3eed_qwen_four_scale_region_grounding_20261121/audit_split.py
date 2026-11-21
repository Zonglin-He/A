"""Verify that the fit-only tree contains no withheld raw pixel-cell draws."""
import numpy as np

from inputs import OUT, load_panel, write
from region_depth import camera, held_cells, sample_points, transform_to_front_view
from crossplatform.round7.camera_projection import project_model_points, raw_to_model_points


def run():
    counts = dict(sample_draws=0, in_image_in_both=0, valid_difference=0,
                  fit_held_pixel_parity_differences=0, raw_held_retained=0, raw_fit_removed=0)
    for row in load_panel():
        path = row["points"]
        samples, indices, _ = sample_points(dict(platform=row["target"], pointpath=path, seed=row["seed"]))
        source = (np.load(path, allow_pickle=False) if path.endswith(".npy")
                  else np.fromfile(path, "<f4").reshape(-1, 4))
        exact = raw_to_model_points(source[indices, :3], row["target"], row["suffix"])
        if row["target"] == "waymo":
            samples, _ = transform_to_front_view(samples, np.empty((0, 7)), row["waymo_view"])
            samples = samples.astype(np.float32)
        specification, size, _ = camera(row)
        raw_projection = project_model_points(exact, specification, size)
        sample_projection = project_model_points(samples, specification, size)
        a, b = raw_projection["in_image"], sample_projection["in_image"]
        both = a & b
        raw_hold = held_cells((raw_projection["uv"][both]//4).astype(np.int64))
        sample_hold = held_cells((sample_projection["uv"][both]//4).astype(np.int64))
        counts["sample_draws"] += len(samples)
        counts["in_image_in_both"] += int(both.sum())
        counts["valid_difference"] += int((a != b).sum())
        counts["fit_held_pixel_parity_differences"] += int((raw_hold != sample_hold).sum())
        counts["raw_held_retained"] += int((raw_hold & ~sample_hold).sum())
        counts["raw_fit_removed"] += int((~raw_hold & sample_hold).sum())
    assert counts["valid_difference"] == counts["raw_held_retained"] == 0
    write(OUT / "split_audit.json", counts)
    print(counts, flush=True)


if __name__ == "__main__":
    run()
