from origami_tube_design import (
    find_ff_tube_design, find_ne_tube_design,
    visualize_ff_tube, visualize_ne_tube,
)

TARGET_N = 4
R_TARGET = 2.0
G_TARGET = 0.3
NUM_RINGS = 10
INTERACTIVE = True

if __name__ == "__main__":
    print("=" * 60)
    print("FF tube design search + visualization")
    print("=" * 60)
    ff_result = find_ff_tube_design(TARGET_N, R_TARGET, G_TARGET)
    if ff_result["alphas_deg"] is not None:
        visualize_ff_tube(ff_result, num_rings=NUM_RINGS, interactive=INTERACTIVE)
    else:
        print("[skip] FF search did not converge; skipping visualization")

    print("=" * 60)
    print("NE tube design search + visualization")
    print("=" * 60)
    ne_result = find_ne_tube_design(TARGET_N, R_TARGET, G_TARGET)
    if ne_result["alphas_deg"] is not None:
        visualize_ne_tube(ne_result, num_rings=NUM_RINGS, interactive=INTERACTIVE)
    else:
        print("[skip] NE search did not converge; skipping visualization")
