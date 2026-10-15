import os
import numpy as np
import pandas as pd


NUM_TIME_STEPS = 128
PHI_ES = 0.3


def poly_area(xs: np.ndarray, ys: np.ndarray) -> float:


    xs = np.asarray(xs).flatten()
    ys = np.asarray(ys).flatten()
    if len(xs) < 3:
        return 0.0


    x_shift = np.roll(xs, -1)
    y_shift = np.roll(ys, -1)
    area = 0.5 * np.abs(np.sum(xs * y_shift - ys * x_shift))
    return float(area)


def compute_frame_areas(df_video: pd.DataFrame):


    frame_areas = {}

    for frame_id, df_frame in df_video.groupby("Frame"):

        xs1 = df_frame["X1"].values
        ys1 = df_frame["Y1"].values
        xs2 = df_frame["X2"].values
        ys2 = df_frame["Y2"].values

        area1 = poly_area(xs1, ys1)
        area2 = poly_area(xs2, ys2)

        frame_areas[frame_id] = {"A1": area1, "A2": area2}

    return frame_areas


def decide_cavity_contour(frame_areas: dict):


    frames = sorted(frame_areas.keys())
    if len(frames) != 2:
        raise ValueError(f"Expect 2 frames (ED/ES), but got {len(frames)} frames: {frames}")

    f1, f2 = frames
    A1_f1 = frame_areas[f1]["A1"]
    A1_f2 = frame_areas[f2]["A1"]
    A2_f1 = frame_areas[f1]["A2"]
    A2_f2 = frame_areas[f2]["A2"]

    delta1 = abs(A1_f1 - A1_f2)
    delta2 = abs(A2_f1 - A2_f2)

    if delta1 >= delta2:
        cavity_key = "A1"
    else:
        cavity_key = "A2"

    return cavity_key


def extract_ed_es_info(frame_areas: dict, cavity_key: str):


    frames = sorted(frame_areas.keys())
    if len(frames) != 2:
        raise ValueError(f"Expect 2 frames for ED/ES, but got {len(frames)} frames.")

    f1, f2 = frames
    A_f1 = frame_areas[f1][cavity_key]
    A_f2 = frame_areas[f2][cavity_key]


    if A_f1 >= A_f2:
        f_ED, A_ED = f1, A_f1
        f_ES, A_ES = f2, A_f2
    else:
        f_ED, A_ED = f2, A_f2
        f_ES, A_ES = f1, A_f1

    return f_ED, f_ES, A_ED, A_ES


def build_pseudo_ecg(num_steps: int,
                     A_ED: float,
                     A_ES: float,
                     phi_es: float = PHI_ES) -> np.ndarray:


    phi = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)


    g = np.zeros_like(phi)


    mask_systole = phi <= phi_es

    if phi_es > 0:
        phi_s = phi[mask_systole] / phi_es
        g[mask_systole] = 0.5 * (1.0 + np.cos(np.pi * phi_s))
    else:

        g[mask_systole] = 1.0


    mask_diastole = phi > phi_es
    if phi_es < 1.0:
        phi_d = (phi[mask_diastole] - phi_es) / (1.0 - phi_es)
        g[mask_diastole] = 0.5 * (1.0 - np.cos(np.pi * phi_d))
    else:

        g[mask_diastole] = 0.0


    A_t = A_ES + (A_ED - A_ES) * g


    if abs(A_ED - A_ES) > 1e-6:
        A_norm = (A_t - A_ES) / (A_ED - A_ES)
    else:

        A_norm = np.full_like(A_t, 0.5, dtype=np.float32)


    dA_norm = np.diff(A_norm, append=A_norm[-1])


    pseudo_ecg = np.stack([phi, A_norm.astype(np.float32), dA_norm.astype(np.float32)], axis=-1)
    return pseudo_ecg


def main(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {"FileName", "X1", "Y1", "X2", "Y2", "Frame"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV is missing required columns: {required_cols - set(df.columns)}")


    for file_name, df_video in df.groupby("FileName"):
        try:
            frame_areas = compute_frame_areas(df_video)
            cavity_key = decide_cavity_contour(frame_areas)
            f_ED, f_ES, A_ED, A_ES = extract_ed_es_info(frame_areas, cavity_key)

            pseudo_ecg = build_pseudo_ecg(
                num_steps=NUM_TIME_STEPS,
                A_ED=A_ED,
                A_ES=A_ES,
                phi_es=PHI_ES,
            )


            base_name = os.path.splitext(file_name)[0]
            out_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(out_path, pseudo_ecg)

            print(f"[OK] {file_name}: saved pseudo-ECG to {out_path}")
        except Exception as e:
            print(f"[WARN] Skip {file_name} due to error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo ECG from volume tracings.")
    parser.add_argument("--csv-path", "--csv_path", dest="csv_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    args = parser.parse_args()
    main(args.csv_path, args.output_dir)
