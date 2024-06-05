

def calc_stride(pose_df, band=5):

    vals = []
    for i in range(0, int(max(pose_df.X))):
        vals.append(pose_df[((i - band) < pose_df.X) &
                            (pose_df.X < (i+band))].shape[0])

    i = 0
    j = len(vals) - 1
    max_so_far = 0
    dist_at_max = 0
    while i < j:
        vol = (j - i) * min(vals[i], vals[j])
        if vol > max_so_far:
            max_so_far = vol
            dist_at_max = abs(i - j)
        if vals[i] > vals[j]:
            j -= 1
        else:
            i += 1

    return dist_at_max
